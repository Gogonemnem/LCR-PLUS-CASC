"""LCR (RoTHop++ two-tower) training loop, GCE loss, and helpers."""
from ..config import *
import logging
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import trange
from torch.utils.data import DataLoader
from .lcr_model import LCRRothopPP
from .. import data
from torch import optim
import random
import numpy as np
from sklearn.metrics import classification_report, f1_score


def _maybe_parallel(model):
    """Wrap in DataParallel when >1 CUDA device is visible (LCR: single [B, L, D] tensor in / dict out)."""
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        return nn.DataParallel(model)
    return model


def _unwrap(model):
    """Return the underlying module for a possibly-DataParallel-wrapped model."""
    return model.module if hasattr(model, 'module') else model


class GCEQ(nn.Module):
    """Generalized cross entropy with exponent q, applied per-class on the true class."""

    def __init__(self, q=0.4):
        super().__init__()
        self.q = q

    def forward(self, y_pred, y_true):
        # y_pred: raw logits [B, C]; y_true: class indices [B]
        # GCE operates on the true-class probability, mirroring TF's y = y_pred[y_true==1].
        log_p = F.log_softmax(y_pred, dim=1)
        p_true = torch.gather(log_p, 1, y_true.unsqueeze(1)).squeeze(1).exp()
        return ((1 - p_true ** self.q) / self.q).mean()


class LCRTrainer:
    """Trains the two-tower LCRRothopPP model on pre-embedded combined tensors."""

    def __init__(self, embedding_dim=768, num_cat=3, num_pol=2, **model_kwargs):
        self.domain = config['domain']
        self.device = config['device']
        self.root_path = path_mapper[self.domain]
        self.run_dir = config.get('run_dir') or self.root_path
        os.makedirs(self.run_dir, exist_ok=True)
        self.num_cat = num_cat
        self.num_pol = num_pol
        self.logger = logging.getLogger(__name__)

        self.model = _maybe_parallel(
            LCRRothopPP(embedding_dim=embedding_dim, **model_kwargs).to(self.device))
        self.optimizer = None
        self.early_stopping_patience = 3
        self.best_val_loss = float('inf')
        self.patience_counter = 0

    def set_seeds(self, value=0):
        random.seed(value)
        np.random.seed(value)
        torch.manual_seed(value)
        torch.cuda.manual_seed_all(value)

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
        loss = (pol_loss + cat_loss) / 2 + self.model.regularization_loss()
        if q is not None and q > 0:
            loss = loss + GCEQ(q)(out['pol'], pols) + GCEQ(q)(out['cat'], cats)
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
                    learning_rate=learning_rate, q=0.4,
                    l1=0.0, l2=0.0, early_stopping=True, resume=None, record=True):
        _unwrap(self.model).l1, _unwrap(self.model).l2 = l1, l2
        self.set_seeds(0)
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

        start_epoch = 0
        if resume:
            ckpt = torch.load(resume, map_location='cpu')
            _unwrap(self.model).load_state_dict(ckpt['model'])
            self.optimizer.load_state_dict(ckpt['optimizer'])
            start_epoch = ckpt['epoch']
            self.best_val_loss = ckpt.get('best_val_loss', float('inf'))
            self.patience_counter = ckpt.get('patience_counter', 0)
            self.logger.info(f'Resumed from epoch {start_epoch} (best val loss {self.best_val_loss:.4f})')

        for epoch in trange(start_epoch, epochs):
            self.model.train()
            batch_loss, cnt = 0.0, 0
            for X, cats, pols in train_loader:
                X, cats, pols = self._to_tensors(X, cats, pols, self.device)
                self.optimizer.zero_grad()
                loss, _ = self._forward_loss(X, cats, pols, q)
                loss.backward()
                self.optimizer.step()
                batch_loss += loss.item()
                cnt += 1
            train_loss = batch_loss / max(cnt, 1)

            val_metrics = self._evaluate(val_loader)

            self.logger.info(f"epoch {epoch + 1}/{epochs} train_loss={train_loss:.6f} "
                             f"val_loss={val_metrics['loss']:.6f} "
                             f"val_cat_f1={val_metrics['cat_f1']:.4f} val_pol_f1={val_metrics['pol_f1']:.4f}")

            improved = False
            if val_metrics['loss'] < self.best_val_loss - 1e-6:
                self.best_val_loss = val_metrics['loss']
                self.patience_counter = 0
                if record:
                    self.save_model('lcr_model')
                improved = True

            if record:
                torch.save({
                    'model': _unwrap(self.model).state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'epoch': epoch + 1,
                    'best_val_loss': self.best_val_loss,
                    'patience_counter': self.patience_counter,
                }, os.path.join(self.run_dir, 'lcr_checkpoint.pth'))

            if early_stopping and not improved:
                self.patience_counter += 1
                if self.patience_counter >= self.early_stopping_patience:
                    self.logger.info('Early stopping.')
                    break

    def evaluate(self, X_test, cat_test, pol_test):
        loader = DataLoader(data.make_tensor_dataset(X_test, cat_test, pol_test), batch_size=batch_size)
        metrics = self._evaluate(loader)
        print(f"test_loss = {metrics['loss']:.6f}, "
              f"cat_acc = {metrics['cat_acc']:.4f}, cat_f1 = {metrics['cat_f1']:.4f}, "
              f"pol_acc = {metrics['pol_acc']:.4f}, pol_f1 = {metrics['pol_f1']:.4f}")
        return metrics

    def tune(self, X_train, cat_train, pol_train, X_val, cat_val, pol_val,
             search_space=None, max_epochs=20):
        """Hyperparameter search with Optuna, mirroring hypertrain.py."""
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

            self.model = LCRRothopPP(hidden_units=hidden, hop=hop,
                                     drop_1=drop_1, drop_2=drop_2).to(self.device)
            self.early_stopping_patience = 3
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
        self.model = LCRRothopPP(hidden_units=bp['hidden_units'], hop=bp['hops'],
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
