"""Unified AbsaTrainer for the CASC and LCR models (with CASC/LCR factories)."""
import logging
import os
import random

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report
from torch import nn, optim
from torch.nn.parallel import DataParallel
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer

from .. import data
from ..config import config, domain as active_domain, training
from ..losses import absa_loss, evaluate
from .casc import BERTLinear
from .lcr import LCRRothopPP


def _strip_targets(sentence):
    """Reconstruct the plain sentence by collapsing the ' [SEP] ' markers.

    The labeler stores ``before [SEP] target [SEP] after`` split from the
    original sentence, so replacing each marker with a space exactly
    restores the plain text (target word stays in its original position).
    CASC consumes plain sentences (target knowledge lives in the label
    columns); LCR keeps the markers for its left/target/right split.
    """
    return sentence.replace(' [SEP] ', ' ').strip()


def _unwrap(model):
    """Return the underlying module for a possibly-DataParallel-wrapped model."""
    return model.module if hasattr(model, 'module') else model


def _parallel(module):
    """Wrap in DataParallel when `config['device']` is bare 'cuda' on a multi-GPU box."""
    dev = config['device']
    if dev == 'cuda' and torch.cuda.device_count() > 1:
        return DataParallel(module)
    return module


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


def _format_val_metrics(epoch, train_avg, metrics):
    """One ``val_<name>=<value>`` per metric (keys other than 'loss')."""
    parts = [f'val_{k}={v:.4f}' for k, v in sorted(metrics.items()) if k != 'loss']
    return (' ' + ' '.join(parts)) if parts else ''


class AbsaTrainer:
    """Trainer owning the full loop for either model type.

    ``model_type`` is ``'casc'`` (BERTLinear) or ``'lcr'`` (LCRRothopPP).
    Per batch the trainer runs ``core(*inputs)``, computes the shared
    :func:`~lcr_plus_casc.losses.absa_loss` (GCE q, L1/L2 over the model's
    ``regularization_params()``, optional per-class weights), then backward
    itself (the models no longer expose ``loss``/``step``).
    Per epoch the validation loader is fed through ``core.predict`` + the shared
    :func:`~lcr_plus_casc.losses.evaluate`.
    """

    def _loss_fn(self, reg_params=()):
        """Build the batch loss closure used for training/evaluation."""
        def loss_fn(out, cats, pols):
            return absa_loss(out, cats, pols, q=self.q, l1=self.l1, l2=self.l2,
                             reg_params=reg_params,
                             cat_weights=self.cat_weights,
                             pol_weights=self.pol_weights)
        return loss_fn

    def __init__(self, model_type, num_cat=None, num_pol=None, **model_kwargs):
        if model_type not in ('casc', 'lcr'):
            raise ValueError(f"model_type must be 'casc' or 'lcr', got {model_type!r}")
        self.model_type = model_type
        self.domain = config['domain']
        self.device = torch.device(config['device'])
        self.root_path = active_domain().root_path
        self.run_dir = config.get('run_dir') or self.root_path
        os.makedirs(self.run_dir, exist_ok=True)
        self.logger = logging.getLogger(f'{__name__}.{model_type.upper()}')
        self.q = config['gce_q']
        self.l1 = 0.0
        self.l2 = 0.0
        self.cat_weights = None
        self.pol_weights = None
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.optimizer = None
        self.early_stopping_patience = 3

        if num_cat is None:
            num_cat = len(active_domain().categories)
        if num_pol is None:
            num_pol = len(active_domain().polarities)
        self.num_cat = num_cat
        self.num_pol = num_pol

        if model_type == 'casc':
            self.bert_type = active_domain().bert_model
            self.tokenizer = AutoTokenizer.from_pretrained(self.bert_type)
            self.aspect_dict = data.aspect_dict
            self.polarity_dict = data.polarity_dict
            model = BERTLinear(self.bert_type, num_cat, num_pol)
        else:
            model = LCRRothopPP(num_cat=num_cat, num_pol=num_pol, **model_kwargs)
        self.model = _parallel(model.to(self.device))

    # ------------------------------ data ------------------------------
    def load_training_data(self, path=None):
        """CASC: TensorDataset from the labeled training set (or a gold TSV path)."""
        sentences, cats, pols = data.load_training_data(training_path=path)
        return self._tokenize_casc(sentences, cats, pols)

    def _tokenize_casc(self, sentences, cats, pols):
        """Build a CASC TensorDataset from explicit (sentences, cats, pols)."""
        sentences = [_strip_targets(s) for s in sentences]
        encoded = self.tokenizer(
            sentences, padding=True, return_tensors='pt',
            max_length=config['max_length'], return_attention_mask=True, truncation=True)
        labels_cat = torch.tensor(cats)
        labels_pol = torch.tensor(pols)
        self.update_class_weights(labels_cat, labels_pol)
        return TensorDataset(encoded['input_ids'], encoded['attention_mask'],
                             labels_cat, labels_pol)

    def update_class_weights(self, labels_cat, labels_pol):
        """CASC: store per-class label counts used to weight the loss terms."""
        self.cat_weights = torch.bincount(labels_cat, minlength=self.num_cat).tolist()
        self.pol_weights = torch.bincount(labels_pol, minlength=self.num_pol).tolist()

    def build_dataset(self, sentences, cats, pols):
        """LCR: tokenize ' [SEP] '-separated sentences once into the LCR dataset."""
        return data.make_tensor_dataset(sentences, cats, pols)

    # ------------------------------ training ------------------------------
    def train_model(self, sent_or_dataset=None, cat_train=None, pol_train=None,
                    sent_val=None, cat_val=None, pol_val=None,
                    epochs=None, batch_size=None,
                    learning_rate=None, bert_lr=None, q=None,
                    l1=0.0, l2=0.0, early_stopping=True, record=True, resume=None):
        is_lcr = self.model_type == 'lcr'
        if epochs is None:
            epochs = training.epochs
        if batch_size is None:
            batch_size = training.batch_size
        if learning_rate is None:
            learning_rate = training.lcr_learning_rate if is_lcr else training.learning_rate
        if q is not None:
            self.q = q
        self.l1 = l1
        self.l2 = l2

        if not is_lcr:
            if cat_train is not None:
                # Explicit (sentences, cats, pols) triples, e.g. gold TSV rows.
                train_ds = self._tokenize_casc(sent_or_dataset, cat_train, pol_train)
            else:
                train_ds = (self.load_training_data()
                            if sent_or_dataset is None else sent_or_dataset)
            if sent_val is not None:
                val_ds = self._tokenize_casc(sent_val, cat_val, pol_val)
            else:
                val_ds = None  # no validation set -> train-only, no early stopping
            self.set_seed(training.seed)
            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
            val_loader = (DataLoader(val_ds, batch_size=batch_size, shuffle=True)
                          if val_ds is not None else None)
            self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
            self._fit(train_loader, val_loader, epochs=epochs, patience=None,
                      record=record, resume=resume,
                      train_total=len(train_ds),
                      val_total=len(val_ds) if val_ds is not None else None)
            return

        if bert_lr is None:
            bert_lr = training.lcr_bert_lr
        core = _unwrap(self.model)
        self.set_seed(training.seed)

        train_ds = self.build_dataset(sent_or_dataset, cat_train, pol_train)
        if sent_val is not None:
            val_ds = self.build_dataset(sent_val, cat_val, pol_val)
        else:
            val_ds = None  # no validation set -> train-only, no early stopping
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size) if val_ds is not None else None

        bert_params, rest_params = [], []
        for name, p in core.named_parameters():
            (bert_params if name.startswith('bert.') else rest_params).append(p)
        self.optimizer = optim.Adam([
            {'params': bert_params, 'lr': bert_lr},
            {'params': rest_params, 'lr': learning_rate},
        ])

        self.best_val_loss, self.patience_counter = self._fit(
            train_loader, val_loader, epochs=epochs,
            patience=self.early_stopping_patience if early_stopping else None,
            record=record, resume=resume,
            train_total=len(train_ds),
            val_total=len(val_ds) if val_ds is not None else None)

    def _fit(self, train_loader, val_loader, *, epochs, patience, record,
             resume=None, train_total=None, val_total=None):
        """Shared epoch loop; returns (best_val_loss, patience_counter)."""
        core = _unwrap(self.model)
        reg_params = core.regularization_params()
        start_epoch = 0
        if resume:
            ckpt = torch.load(resume, map_location='cpu')
            start_epoch, self.best_val_loss, self.patience_counter = \
                _load_resume(ckpt, self.model, self.optimizer)
            self.logger.info(
                f'Resumed from epoch {start_epoch} (best val loss {self.best_val_loss:.4f})')

        for epoch in range(start_epoch, epochs):
            self.model.train()
            running_loss, cnt = 0.0, 0
            samples_seen = 0
            last_log_block = 0
            for batch in train_loader:
                self.optimizer.zero_grad()
                inputs = [x.to(self.device) for x in batch[:-2]]
                cats = batch[-2].to(self.device)
                pols = batch[-1].to(self.device)
                out = core(*inputs)
                loss = absa_loss(out, cats, pols, q=self.q, l1=self.l1, l2=self.l2,
                                 reg_params=reg_params,
                                 cat_weights=self.cat_weights,
                                 pol_weights=self.pol_weights)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), training.max_grad_norm)
                self.optimizer.step()
                running_loss += loss.item()
                cnt += 1
                samples_seen += len(cats)
                log_block = samples_seen // training.log_every
                if log_block > last_log_block:
                    last_log_block = log_block
                    epoch_loss_avg = running_loss / max(cnt, 1)
                    pos = f'{samples_seen}/{train_total}' if train_total else f'{samples_seen}'
                    self.logger.info(
                        f'[epoch {epoch + 1}] {pos} samples | '
                        f'avg_loss={epoch_loss_avg:.6f}')

            train_avg = running_loss / max(cnt, 1)
            pos = f'{samples_seen}/{train_total}' if train_total else f'{samples_seen}'
            self.logger.info(
                f'[epoch {epoch + 1}] {pos} samples | avg_loss={train_avg:.6f}')
            if val_loader is not None:
                preds = core.predict(val_loader, device=self.device,
                                    on_progress=self._predict_progress(
                                        f'[val epoch {epoch + 1}]', val_total=val_total))
                n_val = int(sum(b['cats'].shape[0] for b in preds))
                val_pos = f'{n_val}/{val_total}' if val_total else f'{n_val}'
                self.logger.info(f'[val epoch {epoch + 1}] {val_pos} samples evaluated')
                metrics = evaluate(preds, self._loss_fn(), config['metrics'])

                extra = _format_val_metrics(epoch + 1, train_avg, metrics)
                self.logger.info(
                    f'epoch {epoch + 1}/{epochs}: train_loss={train_avg:.6f} '
                    f'val_loss={metrics["loss"]:.6f}{extra}')

                improved = metrics['loss'] < self.best_val_loss - 1e-6
                if improved:
                    self.best_val_loss = metrics['loss']
                    self.patience_counter = 0
                elif patience is not None:
                    self.patience_counter += 1
            else:
                self.logger.info(
                    f'epoch {epoch + 1}/{epochs}: train_loss={train_avg:.6f} '
                    f'(no validation set; saving each epoch)')
                improved = True

            if record:
                if improved:
                    self._save_best()
                self._save_checkpoint(epoch + 1)

            if val_loader is not None and patience is not None \
                    and not improved and self.patience_counter >= patience:
                self.logger.info('Early stopping.')
                break

        return self.best_val_loss, self.patience_counter

    def _save_best(self):
        path = os.path.join(self.run_dir, f'{self.model_type}_model.pth')
        torch.save(_unwrap(self.model).state_dict(), path)
        self.logger.info(f'Saved model to {path}')

    def _save_checkpoint(self, epoch_no):
        torch.save({
            'model': _unwrap(self.model).state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epoch': epoch_no,
            'best_val_loss': self.best_val_loss,
            'patience_counter': self.patience_counter,
        }, os.path.join(self.run_dir, f'{self.model_type}_checkpoint.pth'))

    def _predict_progress(self, label, val_total=None):
        log_every = max(1, training.log_every)
        state = {'last': 0}

        def _cb(samples_seen):
            block = samples_seen // log_every
            if block > state['last']:
                state['last'] = block
                pos = f'{samples_seen}/{val_total}' if val_total else f'{samples_seen}'
                self.logger.info(f'{label} {pos} samples evaluated')

        return _cb

    # ------------------------------ evaluation ------------------------------
    def evaluate(self, sent_or_dataset=None, cat_test=None, pol_test=None,
                 year=None, test_type='test', label_type='single'):
        if cat_test is not None and self.model_type == 'lcr':
            # LCR: raw (sentences, cats, pols) test split.
            test_ds = self.build_dataset(sent_or_dataset, cat_test, pol_test)
            n_total = len(test_ds)
            loader = DataLoader(test_ds, batch_size=training.batch_size)
            core = _unwrap(self.model)
            with torch.no_grad():
                preds = core.predict(loader, device=self.device,
                                     on_progress=self._predict_progress('[test]',
                                                                        val_total=n_total))
            n_test = int(sum(b['cats'].shape[0] for b in preds))
            test_pos = f'{n_test}/{n_total}' if n_total else f'{n_test}'
            self.logger.info(f'[test] {test_pos} samples evaluated')

            rows = []
            offset = 0
            for b in preds:
                n = b['cats'].shape[0]
                for i in range(n):
                    sent = _strip_targets(sent_or_dataset[offset + i])
                    true_c = int(b['cats'][i]); pred_c = int(b['cat'][i].argmax())
                    true_p = int(b['pols'][i]); pred_p = int(b['pol'][i].argmax())
                    rows.append([sent,
                                 data.aspect_dict[true_c], data.aspect_dict[pred_c],
                                 data.polarity_dict[true_p], data.polarity_dict[pred_p]])
                offset += n

            df = pd.DataFrame(
                rows, columns=['sentence', 'actual category', 'predicted category',
                               'actual polarity', 'predicted polarity'])
            preds_path = os.path.join(self.run_dir, 'lcr_predictions.csv')
            df.to_csv(preds_path, index=False)
            self.logger.info(f'Wrote predictions to {preds_path}')

            self.logger.info('Polarity')
            self.logger.info(classification_report(df['actual polarity'], df['predicted polarity'], digits=4))
            self.logger.info('Aspect')
            self.logger.info(classification_report(df['actual category'], df['predicted category'], digits=4))

            metrics = evaluate(preds, self._loss_fn(), config['metrics'])
            detail = ', '.join(f'{k}={v:.4f}' for k, v in sorted(metrics.items()) if k != 'loss')
            self.logger.info(f"test_loss = {metrics['loss']:.6f}, {detail}")
            return metrics

        # CASC: explicit test rows, else the default SemEval split.
        if cat_test is not None:
            test_sentences, test_cats, test_pols = sent_or_dataset, cat_test, pol_test
        else:
            if year is None:
                year = training.test_year
            test_sentences, test_cats, test_pols = data.load_semeval(year, test_type, label_type)

        model = self.model
        model.eval()
        device = self.device

        eval_batch = training.batch_size
        rows = []
        log_every = max(1, training.log_every)
        samples_seen = 0
        last_log_block = 0
        n_total = len(test_sentences)
        with torch.no_grad():
            for i in range(0, n_total, eval_batch):
                batch = [_strip_targets(s) for s in test_sentences[i:i + eval_batch]]
                batch_cats = test_cats[i:i + eval_batch]
                batch_pols = test_pols[i:i + eval_batch]

                encoded = self.tokenizer(
                    batch, padding='longest', return_tensors='pt',
                    return_attention_mask=True,                     return_token_type_ids=False,
                    max_length=config['max_length'], truncation=True).to(device)

                out = model(encoded['input_ids'], encoded['attention_mask'])
                logits_cat, logits_pol = out['cat'], out['pol']

                samples_seen += len(batch)
                log_block = samples_seen // log_every
                if log_block > last_log_block:
                    last_log_block = log_block
                    test_pos = f'{samples_seen}/{n_total}' if n_total else f'{samples_seen}'
                    self.logger.info(f'[test] {test_pos} samples evaluated')

                for sentence, cat, logit_cat, pol, logit_pol in zip(
                        batch, batch_cats, logits_cat, batch_pols, logits_pol):
                    rows.append([sentence, self.aspect_dict[cat],
                                 self.aspect_dict[logit_cat.argmax().item()],
                                 self.polarity_dict[pol],
                                 self.polarity_dict[logit_pol.argmax().item()]])

        test_pos = f'{samples_seen}/{n_total}' if n_total else f'{samples_seen}'
        self.logger.info(f'[test] {test_pos} samples evaluated')

        df = pd.DataFrame(
            rows, columns=['sentence', 'actual category', 'predicted category',
                           'actual polarity', 'predicted polarity'])

        preds_path = os.path.join(self.run_dir, 'casc_predictions.csv')
        df.to_csv(preds_path, index=False)
        self.logger.info(f'Wrote predictions to {preds_path}')

        self.logger.info('Polarity')
        self.logger.info(classification_report(df['actual polarity'], df['predicted polarity'], digits=4))
        self.logger.info('Aspect')
        self.logger.info(classification_report(df['actual category'], df['predicted category'], digits=4))
        return df

    # ------------------------------ tuning (LCR only) ------------------------------
    def tune(self, sent_train, cat_train, pol_train, sent_val, cat_val, pol_val,
             max_epochs=20):
        """Hyperparameter search with Optuna (LCR only)."""
        if self.model_type != 'lcr':
            raise NotImplementedError('tune is only supported for the LCR model')
        import optuna

        def objective(trial):
            l1 = trial.suggest_float('l1', 1e-9, 1e-3, log=True)
            l2 = trial.suggest_float('l2', 1e-9, 1e-3, log=True)
            lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
            drop_1 = trial.suggest_float('drop_1', 0.2, 0.6, step=0.1)
            drop_2 = trial.suggest_float('drop_2', 0.2, 0.6, step=0.1)
            hidden = trial.suggest_int('hidden_units', 200, 750, step=50)
            hop = trial.suggest_int('hops', 1, 8)
            q = trial.suggest_float('q', 0.1, 1.0, step=0.1)

            self.model = _parallel(LCRRothopPP(num_cat=self.num_cat, num_pol=self.num_pol,
                                               hidden_units=hidden, hop=hop,
                                               drop_1=drop_1, drop_2=drop_2).to(self.device))
            self.best_val_loss = float('inf')
            self.patience_counter = 0
            self.train_model(sent_train, cat_train, pol_train, sent_val, cat_val, pol_val,
                             epochs=max_epochs, learning_rate=lr, q=q, l1=l1, l2=l2,
                             early_stopping=True, record=False)
            return self.best_val_loss

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=20)

        # Retrain once with the best trial's params so the saved model matches.
        bp = study.best_trial.params
        self.model = _parallel(LCRRothopPP(num_cat=self.num_cat, num_pol=self.num_pol,
                                           hidden_units=bp['hidden_units'], hop=bp['hops'],
                                           drop_1=bp['drop_1'], drop_2=bp['drop_2']).to(self.device))
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.train_model(sent_train, cat_train, pol_train, sent_val, cat_val, pol_val,
                         epochs=max_epochs, learning_rate=bp['lr'], q=bp['q'],
                         l1=bp['l1'], l2=bp['l2'], early_stopping=True)
        self.save_model('lcr_model')
        return study

    # ------------------------------ persistence ------------------------------
    def save_model(self, name=None):
        """Save the model state dict into the current run directory."""
        name = name or f'{self.model_type}_model'
        path = os.path.join(self.run_dir, f'{name}.pth')
        torch.save(_unwrap(self.model).state_dict(), path)
        self.logger.info(f'Saved model to {path}')
        return path

    def load_model(self, name=None):
        """Load a model; accepts a directory (run dir) or an explicit .pth path."""
        name = name or f'{self.model_type}_model'
        tag = self.model_type
        if os.path.isdir(name):
            path = os.path.join(name, f'{tag}_model.pth')
        elif name.endswith('.pth'):
            path = name
        else:
            path = os.path.join(self.run_dir, f'{name}.pth')
            if not os.path.exists(path):
                path = os.path.join(self.run_dir, f'{tag}_model.pth')
            if not os.path.exists(path):
                from .. import runs
                path = runs.latest_artifact(f'{tag}_model.pth', (f'_{tag}', '_all'))
            if not path or not os.path.exists(path):
                raise FileNotFoundError(
                    f"no {tag!r} model found (tried {self.run_dir}, "
                    f"newest '{tag}_model.pth' under this recipe's artifact dir)")
        _unwrap(self.model).load_state_dict(torch.load(path, map_location=self.device))
        self.logger.info(f'Loaded model from {path}')
        return self.model

    # ------------------------------ misc ------------------------------
    def set_seed(self, value):
        _set_seeds(value)

    set_seeds = set_seed


def CASC(**kwargs):
    """Backward-compatible factory: CASC trainer (BERTLinear)."""
    return AbsaTrainer('casc', **kwargs)


def LCR(**kwargs):
    """Backward-compatible factory: LCR trainer (LCRRothopPP)."""
    return AbsaTrainer('lcr', **kwargs)
