"""Shared training loop plus the CASC and LCR trainer wrappers."""
import logging
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report, f1_score
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer
from tqdm import trange

from .. import data
from ..config import (
    aspect_category_mapper,
    batch_size,
    bert_mapper,
    config,
    epochs,
    learning_rate,
    lcr_learning_rate,
    max_grad_norm,
    max_length,
    path_mapper,
    seed,
    sentiment_category_mapper,
    test_year,
    validation_data_size,
)
from ..losses import GCEQ
from .casc import BERTLinear
from .lcr import LCRRothopPP


def _maybe_parallel(model):
    """Wrap in DataParallel when >1 CUDA device is visible."""
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        return nn.DataParallel(model)
    return model


def _unwrap(model):
    """Return the underlying module for a possibly-DataParallel-wrapped model."""
    return model.module if hasattr(model, 'module') else model


def _load_resume(ckpt, model, optimizer):
    """Restore a checkpoint; returns (start_epoch, best_val_loss, patience_counter)."""
    _unwrap(model).load_state_dict(ckpt['model'])
    optimizer.load_state_dict(ckpt['optimizer'])
    return (ckpt['epoch'],
            ckpt.get('best_val_loss', float('inf')),
            ckpt.get('patience_counter', 0))


def _set_seeds(value):
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)


def run_training(model, train_step, evaluate_metrics, log_epoch, train_loader, val_loader,
                 optimizer, *, epochs, logger, max_grad_norm=1.0, start_epoch=0,
                 best_val_loss=None, patience_counter=0, patience=None,
                 save_best=None, save_checkpoint=None, resume=None):
    """Shared epoch loop for CASC and LCR.

    - train_step(*batch, device=device) -> per-batch loss tensor (run_training backprops it).
    - evaluate_metrics() -> dict of metrics that must include 'loss'.
    - log_epoch(epoch_no, train_avg, metrics)
    - save_best(model, metrics, epoch_no) on improvement (optional).
    - save_checkpoint(model, optimizer, metrics, epoch_no, best_val_loss, patience_counter)
      every epoch (optional).
    - early stopping: when patience is set, stop after `patience` non-improving epochs.

    Returns (best_val_loss, patience_counter) for the caller to persist.
    """
    if best_val_loss is None:
        best_val_loss = float('inf')
    device = next(_unwrap(model).parameters()).device

    if resume:
        ckpt = torch.load(resume, map_location='cpu')
        start_epoch, best_val_loss, patience_counter = _load_resume(ckpt, model, optimizer)
        logger.info(f'Resumed from epoch {start_epoch} (best val loss {best_val_loss:.4f})')

    for epoch in trange(start_epoch, epochs):
        model.train()
        running_loss, cnt = 0.0, 0
        for batch in train_loader:
            optimizer.zero_grad()
            loss = train_step(*batch, device=device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            running_loss += loss.item()
            cnt += 1
        train_avg = running_loss / max(cnt, 1)

        metrics = evaluate_metrics()

        log_epoch(epoch + 1, train_avg, metrics)

        improved = metrics['loss'] < best_val_loss - 1e-6
        if improved:
            best_val_loss = metrics['loss']
            patience_counter = 0
            if save_best:
                save_best(model, metrics, epoch + 1)
        elif patience is not None:
            patience_counter += 1

        if save_checkpoint:
            save_checkpoint(model, optimizer, metrics, epoch + 1, best_val_loss, patience_counter)

        if patience is not None and not improved and patience_counter >= patience:
            logger.info('Early stopping.')
            break

    return best_val_loss, patience_counter


class CASC:
    """CASC BERTLinear trainer."""

    def __init__(self):
        self.domain = config['domain']
        self.bert_type = bert_mapper[self.domain]
        self.device = torch.device(config['device'])
        self.tokenizer = AutoTokenizer.from_pretrained(self.bert_type)
        self.root_path = path_mapper[self.domain]
        self.gce_q = config['gce_q']
        self.run_dir = config.get('run_dir') or self.root_path
        os.makedirs(self.run_dir, exist_ok=True)
        self.logger = logging.getLogger(f'{__name__}.CASC')

        categories = aspect_category_mapper[self.domain]
        polarities = sentiment_category_mapper[self.domain]

        model = BERTLinear(self.bert_type, len(categories), len(polarities), q=self.gce_q)
        self.model = _maybe_parallel(model.to(self.device))

        self.aspect_dict = data.aspect_dict
        self.polarity_dict = data.polarity_dict

    def load_training_data(self):
        """Returns a TensorDataset of (labels_cat, labels_pol, input_ids, attention_mask)."""
        sentences, cats, pols = data.load_training_data()
        encoded = self.tokenizer(
            sentences, padding=True, return_tensors='pt',
            max_length=max_length, return_attention_mask=True, truncation=True)
        labels_cat = torch.tensor(cats)
        labels_pol = torch.tensor(pols)
        self.update_class_weights(labels_cat, labels_pol)
        return TensorDataset(labels_cat, labels_pol,
                             encoded['input_ids'], encoded['attention_mask'])

    def update_class_weights(self, labels_cat, labels_pol):
        model = _unwrap(self.model)
        num_cat = model.ff_cat.out_features
        num_pol = model.ff_pol.out_features
        model.loss_cat.set_weights(torch.bincount(labels_cat, minlength=num_cat).tolist())
        model.loss_pol.set_weights(torch.bincount(labels_pol, minlength=num_pol).tolist())

    def train_model(self, dataset=None, epochs=epochs, resume=None):
        if dataset is None:
            dataset = self.load_training_data()
        self.set_seed(seed)

        train_data, val_data = torch.utils.data.random_split(
            dataset, [len(dataset) - validation_data_size, validation_data_size])
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)

        model = self.model
        device = self.device
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        batch_losses = []

        def train_step(labels_cat, labels_pol, input_ids, attention_mask, device=device):
            encoded_dict = {
                'input_ids': input_ids.to(device),
                'attention_mask': attention_mask.to(device),
            }
            loss, _, _ = model(labels_cat.to(device), labels_pol.to(device), **encoded_dict)
            batch_losses.append(loss.item())
            if len(batch_losses) == 50:
                self.logger.info('Batch loss: %.6f', sum(batch_losses) / len(batch_losses))
                batch_losses.clear()
            return loss

        @torch.no_grad()
        def evaluate_metrics():
            model.eval()
            val_loss, iters = 0.0, 0
            for labels_cat, labels_pol, input_ids, attention_mask in val_loader:
                encoded_dict = {
                    'input_ids': input_ids.to(device),
                    'attention_mask': attention_mask.to(device),
                }
                loss, _, _ = model(labels_cat.to(device), labels_pol.to(device), **encoded_dict)
                val_loss += loss.item()
                iters += 1
            return {'loss': val_loss / max(iters, 1)}

        def log_epoch(epoch_no, train_avg, metrics):
            self.logger.info(
                f'epoch {epoch_no}/{epochs}: train_loss={train_avg:.6f} '
                f'val_loss={metrics["loss"]:.6f}')

        def save_best(m, _metrics, _epoch_no):
            self.save_model('casc_model')

        def save_checkpoint(m, opt, _metrics, epoch_no, best_val_loss, patience_counter):
            torch.save({
                'model': _unwrap(m).state_dict(),
                'optimizer': opt.state_dict(),
                'epoch': epoch_no,
                'best_val_loss': best_val_loss,
                'patience_counter': patience_counter,
            }, os.path.join(self.run_dir, 'casc_checkpoint.pth'))

        run_training(
            model, train_step, evaluate_metrics, log_epoch,
            train_loader, val_loader, optimizer,
            epochs=epochs, logger=self.logger, max_grad_norm=max_grad_norm,
            patience=None, save_best=save_best,
            save_checkpoint=save_checkpoint, resume=resume)

    def set_seed(self, value):
        _set_seeds(value)

    def save_model(self, name='casc_model'):
        """Save the model state dict into the current run directory."""
        path = os.path.join(self.run_dir, f'{name}.pth')
        torch.save(_unwrap(self.model).state_dict(), path)
        self.logger.info(f'Saved model to {path}')
        return path

    def load_model(self, name='model_CASC'):
        """Load a model; accepts a directory (run dir) or an explicit .pth path."""
        model = self.model
        if os.path.isdir(name):
            path = os.path.join(name, 'casc_model.pth')
            if not os.path.exists(path):
                path = os.path.join(name, 'best_model.pth')
        elif name.endswith('.pth'):
            path = name
        else:
            path = os.path.join(self.run_dir, f'{name}.pth')
            if not os.path.exists(path):
                path = os.path.join(self.run_dir, 'casc_model.pth')
            if not os.path.exists(path):
                path = os.path.join(self.run_dir, 'best_model.pth')
            if not os.path.exists(path):
                from .. import runs
                path = runs.latest_artifact('casc_model.pth', ('_casc', '_all'))
                if not path or not os.path.exists(path):
                    # Legacy shared-dir artifacts are the LCR model (trained last);
                    # only trust best_model.pth from dedicated CASC runs.
                    path = runs.latest_artifact('best_model.pth', ('_casc',))
            if not path or not os.path.exists(path):
                path = os.path.join(self.root_path, f'{name}.pth')
        model.load_state_dict(torch.load(path, map_location=config['device']))
        self.model = model
        self.logger.info(f'Loaded model from {path}')
        return self.model

    def evaluate(self, year=None, test_type='test', label_type='single'):
        if year is None:
            year = test_year
        test_sentences, test_cats, test_pols = data.load_semeval(year, test_type, label_type)

        model = self.model
        model.eval()
        device = self.device

        rows = []
        with torch.no_grad():
            for i in range(0, len(test_sentences), batch_size):
                batch = test_sentences[i:i + batch_size]
                batch_cats = test_cats[i:i + batch_size]
                batch_pols = test_pols[i:i + batch_size]
                cats = torch.tensor(batch_cats)
                pols = torch.tensor(batch_pols)

                encoded = self.tokenizer(
                    batch, padding='longest', return_tensors='pt',
                    return_attention_mask=True, max_length=max_length,
                    truncation=True).to(device)

                _, logits_cat, logits_pol = model(cats.to(device),
                                                  pols.to(device), **encoded)

                for sentence, cat, logit_cat, pol, logit_pol in zip(
                        batch, batch_cats, logits_cat, batch_pols, logits_pol):
                    rows.append([sentence, self.aspect_dict[cat],
                                 self.aspect_dict[logit_cat.argmax().item()],
                                 self.polarity_dict[pol],
                                 self.polarity_dict[logit_pol.argmax().item()]])

        df = pd.DataFrame(
            rows, columns=['sentence', 'actual category', 'predicted category',
                           'actual polarity', 'predicted polarity'])

        preds_path = os.path.join(self.run_dir, 'casc_predictions.csv')
        df.to_csv(preds_path, index=False)
        self.logger.info(f'Wrote predictions to {preds_path}')

        print('Polarity')
        print(classification_report(df['actual polarity'], df['predicted polarity'], digits=4))
        print()
        print('Aspect')
        print(classification_report(df['actual category'], df['predicted category'], digits=4))
        print()
        return df


class LCR:
    """Trains the two-tower LCRRothopPP model on pre-embedded combined tensors."""

    def __init__(self, embedding_dim=768, num_cat=None, num_pol=None, **model_kwargs):
        self.domain = config['domain']
        self.device = torch.device(config['device'])
        self.root_path = path_mapper[self.domain]
        self.run_dir = config.get('run_dir') or self.root_path
        os.makedirs(self.run_dir, exist_ok=True)
        self.num_cat = num_cat if num_cat is not None else len(aspect_category_mapper[self.domain])
        self.num_pol = num_pol if num_pol is not None else len(sentiment_category_mapper[self.domain])
        self.logger = logging.getLogger(f'{__name__}.LCR')

        self.model = _maybe_parallel(
            LCRRothopPP(embedding_dim=embedding_dim, num_cat=self.num_cat,
                        num_pol=self.num_pol, **model_kwargs).to(self.device))
        self.optimizer = None
        self.early_stopping_patience = 3
        self.best_val_loss = float('inf')
        self.patience_counter = 0

    def set_seeds(self, value=0):
        _set_seeds(value)

    @staticmethod
    def _to_tensors(X, cats, pols, device):
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X)
        cats = torch.as_tensor(cats, dtype=torch.long)
        pols = torch.as_tensor(pols, dtype=torch.long)
        return X.to(device), cats.to(device), pols.to(device)

    def _forward_loss(self, X, cats, pols, q):
        out = self.model(X)
        pol_loss = F.cross_entropy(out['pol'], pols)
        cat_loss = F.cross_entropy(out['cat'], cats)
        loss = (pol_loss + cat_loss) / 2 + _unwrap(self.model).regularization_loss()
        if q is not None and q > 0:
            gce = GCEQ(q)
            loss = loss + gce(out['pol'], pols) + gce(out['cat'], cats)
        return loss, out

    @torch.no_grad()
    def _evaluate(self, loader):
        self.model.eval()
        cat_preds, cat_true = [], []
        pol_preds, pol_true = [], []
        total_loss = 0.0
        iters = 0
        for X, cats, pols in loader:
            X, cats, pols = self._to_tensors(X, cats, pols, self.device)
            loss, out = self._forward_loss(X, cats, pols, q=None)
            total_loss += loss.item()
            cat_preds.append(out['cat'].argmax(1).cpu())
            pol_preds.append(out['pol'].argmax(1).cpu())
            cat_true.append(cats.cpu())
            pol_true.append(pols.cpu())
            iters += 1
        cat_pred = torch.cat(cat_preds).numpy()
        cat_true = torch.cat(cat_true).numpy()
        pol_pred = torch.cat(pol_preds).numpy()
        pol_true = torch.cat(pol_true).numpy()
        return {
            'loss': total_loss / max(iters, 1),
            'cat_acc': (cat_pred == cat_true).mean(),
            'pol_acc': (pol_pred == pol_true).mean(),
            'cat_f1': f1_score(cat_true, cat_pred, average='macro', zero_division=0),
            'pol_f1': f1_score(pol_true, pol_pred, average='macro', zero_division=0),
        }

    def train_model(self, X_train, cat_train, pol_train,
                    X_val=None, cat_val=None, pol_val=None,
                    epochs=epochs, batch_size=batch_size,
                    learning_rate=None, q=None,
                    l1=0.0, l2=0.0, early_stopping=True, resume=None, record=True):
        if learning_rate is None:
            learning_rate = lcr_learning_rate
        if q is None:
            q = config['gce_q']
        _unwrap(self.model).l1, _unwrap(self.model).l2 = l1, l2
        self.set_seeds(seed)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        train_ds = data.make_tensor_dataset(X_train, cat_train, pol_train)
        if X_val is not None:
            val_ds = data.make_tensor_dataset(X_val, cat_val, pol_val)
            val_loader = DataLoader(val_ds, batch_size=batch_size)
        else:
            train_ds, val_ds = torch.utils.data.random_split(
                train_ds, [len(train_ds) - validation_data_size, validation_data_size])
            val_loader = DataLoader(val_ds, batch_size=batch_size)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

        def train_step(X, cats, pols, device=self.device):
            X, cats, pols = self._to_tensors(X, cats, pols, device)
            loss, _ = self._forward_loss(X, cats, pols, q)
            return loss

        def evaluate_metrics(self=self):
            return self._evaluate(val_loader)

        def log_epoch(epoch_no, train_avg, metrics):
            self.logger.info(
                f'epoch {epoch_no}/{epochs} train_loss={train_avg:.6f} '
                f'val_loss={metrics["loss"]:.6f} '
                f'val_cat_f1={metrics["cat_f1"]:.4f} val_pol_f1={metrics["pol_f1"]:.4f}')

        def save_best(m, metrics, epoch_no):
            if record:
                self.save_model('lcr_model')

        def save_checkpoint(m, opt, _metrics, epoch_no, bvl, pc):
            if record:
                torch.save({
                    'model': _unwrap(m).state_dict(),
                    'optimizer': opt.state_dict(),
                    'epoch': epoch_no,
                    'best_val_loss': bvl,
                    'patience_counter': pc,
                }, os.path.join(self.run_dir, 'lcr_checkpoint.pth'))

        best_val_loss, patience_counter = run_training(
            self.model, train_step, evaluate_metrics, log_epoch,
            train_loader, val_loader, self.optimizer,
            epochs=epochs, logger=self.logger, max_grad_norm=max_grad_norm,
            best_val_loss=self.best_val_loss,
            patience_counter=self.patience_counter,
            patience=self.early_stopping_patience if early_stopping else None,
            save_best=save_best, save_checkpoint=save_checkpoint,
            resume=resume)
        # Persist final best/metrics back onto the trainer instance (needed by tune()).
        self.best_val_loss = best_val_loss
        self.patience_counter = patience_counter

    def evaluate(self, X_test, cat_test, pol_test):
        loader = DataLoader(data.make_tensor_dataset(X_test, cat_test, pol_test),
                            batch_size=batch_size)
        metrics = self._evaluate(loader)
        print(f"test_loss = {metrics['loss']:.6f}, "
              f"cat_acc = {metrics['cat_acc']:.4f}, cat_f1 = {metrics['cat_f1']:.4f}, "
              f"pol_acc = {metrics['pol_acc']:.4f}, pol_f1 = {metrics['pol_f1']:.4f}")
        return metrics

    def tune(self, X_train, cat_train, pol_train, X_val, cat_val, pol_val,
             max_epochs=20):
        """Hyperparameter search with Optuna."""
        import optuna
        Xv = np.asarray(X_val)
        Xtr = np.asarray(X_train)

        def objective(trial):
            l1 = trial.suggest_float('l1', 1e-9, 1e-3, log=True)
            l2 = trial.suggest_float('l2', 1e-9, 1e-3, log=True)
            lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
            drop_1 = trial.suggest_float('drop_1', 0.2, 0.6, step=0.1)
            drop_2 = trial.suggest_float('drop_2', 0.2, 0.6, step=0.1)
            hidden = trial.suggest_int('hidden_units', 200, 750, step=50)
            hop = trial.suggest_int('hops', 1, 8)
            q = trial.suggest_float('q', 0.1, 1.0, step=0.1)

            self.model = LCRRothopPP(num_cat=self.num_cat, num_pol=self.num_pol,
                                     hidden_units=hidden, hop=hop,
                                     drop_1=drop_1, drop_2=drop_2).to(self.device)
            self.best_val_loss = float('inf')
            self.patience_counter = 0
            self.train_model(Xtr, cat_train, pol_train, Xv, cat_val, pol_val,
                             epochs=max_epochs, learning_rate=lr, q=q, l1=l1, l2=l2,
                             early_stopping=True, record=False)
            return self.best_val_loss

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=20)

        # Retrain once with the best trial's params so the saved model matches.
        bp = study.best_trial.params
        self.model = LCRRothopPP(num_cat=self.num_cat, num_pol=self.num_pol,
                                 hidden_units=bp['hidden_units'], hop=bp['hops'],
                                 drop_1=bp['drop_1'], drop_2=bp['drop_2']).to(self.device)
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.train_model(Xtr, cat_train, pol_train, Xv, cat_val, pol_val,
                         epochs=max_epochs, learning_rate=bp['lr'], q=bp['q'],
                         l1=bp['l1'], l2=bp['l2'], early_stopping=True)
        self.save_model('lcr_model')
        return study

    def save_model(self, name='lcr_model'):
        """Save the model state dict into the current run directory."""
        path = os.path.join(self.run_dir, f'{name}.pth')
        torch.save(_unwrap(self.model).state_dict(), path)
        self.logger.info(f'Saved model to {path}')
        return path

    def _find_latest_trained(self, filename):
        from .. import runs
        return runs.latest_artifact(filename, ('_lcr', '_all'))

    def load_model(self, name='lcr_model'):
        """Load a model; accepts a directory (run dir) or an explicit .pth path."""
        if os.path.isdir(name):
            path = os.path.join(name, 'lcr_model.pth')
            if not os.path.exists(path):
                path = os.path.join(name, 'best_model.pth')
        elif name.endswith('.pth'):
            path = name
        else:
            path = os.path.join(self.run_dir, f'{name}.pth')
            if not os.path.exists(path):
                path = os.path.join(self.run_dir, 'lcr_model.pth')
            if not os.path.exists(path):
                path = os.path.join(self.run_dir, 'best_model.pth')
            if not os.path.exists(path):
                path = self._find_latest_trained('lcr_model.pth')
            if not os.path.exists(path):
                path = self._find_latest_trained('best_model.pth')
            if not path or not os.path.exists(path):
                path = os.path.join(self.root_path, f'{name}.pth')
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.logger.info(f'Loaded model from {path}')
        return self.model
