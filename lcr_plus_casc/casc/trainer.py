"""CASC training loop, losses, and helpers."""
from ..config import *
from ..filter_words import filter_words
import logging
import os
import torch
import torch.nn as nn
from transformers import AutoTokenizer
from tqdm import trange
from torch.utils.data import DataLoader, TensorDataset
from .model import BERTLinear
from .. import data
from torch import optim
import random
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report


def _maybe_parallel(model):
    """Wrap in DataParallel when >1 CUDA device is visible (CASC: (input_ids, attention_mask, cat, pol) in / (loss, logits_cat, logits_pol) out)."""
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        return nn.DataParallel(model)
    return model


def _unwrap(model):
    """Return the underlying module for a possibly-DataParallel-wrapped model."""
    return model.module if hasattr(model, 'module') else model


class Trainer:

    def __init__(self):
        self.domain = config['domain']
        self.bert_type = bert_mapper[self.domain]
        self.device = config['device']
        self.tokenizer = AutoTokenizer.from_pretrained(self.bert_type)
        self.root_path = path_mapper[self.domain]
        self.run_dir = config.get('run_dir') or self.root_path
        os.makedirs(self.run_dir, exist_ok=True)
        self.logger = logging.getLogger(__name__)

        categories = aspect_category_mapper[self.domain]
        polarities = sentiment_category_mapper[self.domain]

        self.model = _maybe_parallel(BERTLinear(
            self.bert_type, len(categories), len(polarities)).to(self.device))

        aspect_dict = {}
        inv_aspect_dict = {}
        for i, cat in enumerate(categories):
            aspect_dict[i] = cat
            inv_aspect_dict[cat] = i

        polarity_dict = {}
        inv_polarity_dict = {}
        for i, pol in enumerate(polarities):
            polarity_dict[i] = pol
            inv_polarity_dict[pol] = i

        self.aspect_dict = aspect_dict
        self.inv_aspect_dict = inv_aspect_dict
        self.polarity_dict = polarity_dict
        self.inv_polarity_dict = inv_polarity_dict

    def load_training_data(self):
        sentences, cats, pols = data.load_training_data()
        encoded_dict = self.tokenizer(
            sentences,
            padding=True,
            return_tensors='pt',
            max_length=128,
            return_attention_mask=True,
            truncation=True)
        labels_cat = torch.tensor(cats)
        labels_pol = torch.tensor(pols)

        self.update_class_weights(labels_cat, labels_pol)

        dataset = TensorDataset(
            labels_cat, labels_pol, encoded_dict['input_ids'], encoded_dict['attention_mask'])
        return dataset

    def update_class_weights(self, labels_cat, labels_pol):
        model = _unwrap(self.model)
        num_cat = model.ff_cat.out_features
        num_pol = model.ff_pol.out_features
        model.loss_cat.set_weights(torch.bincount(labels_cat, minlength=num_cat).tolist())
        model.loss_pol.set_weights(torch.bincount(labels_pol, minlength=num_pol).tolist())

    def set_seed(self, value):
        random.seed(value)
        np.random.seed(value)
        torch.manual_seed(value)
        torch.cuda.manual_seed_all(value)

    def train_model(self, dataset, epochs=epochs, resume=None):
        self.set_seed(0)

        # Prepare dataset
        train_data, val_data = torch.utils.data.random_split(
            dataset, [len(dataset) - validation_data_size, validation_data_size])
        dataloader = DataLoader(train_data, batch_size=batch_size)
        val_dataloader = DataLoader(val_data, batch_size=batch_size)

        model = self.model
        device = self.device

        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        start_epoch = 0
        best_val_loss = float('inf')
        if resume:
            ckpt = torch.load(resume, map_location='cpu')
            _unwrap(model).load_state_dict(ckpt['model'])
            optimizer.load_state_dict(ckpt['optimizer'])
            start_epoch = ckpt['epoch']
            best_val_loss = ckpt.get('best_val_loss', float('inf'))
            self.logger.info(f'Resumed from epoch {start_epoch} (best val loss {best_val_loss:.4f})')

        for epoch in trange(start_epoch, epochs):
            model.train()
            print_loss = 0
            batch_loss = 0
            cnt = 0
            for labels_cat, labels_pol, input_ids, attention_mask in dataloader:
                optimizer.zero_grad()
                encoded_dict = {
                    'input_ids': input_ids.to(device),
                    'attention_mask': attention_mask.to(device)
                }
                loss, _, _ = model(labels_cat.to(device),
                                   labels_pol.to(device), **encoded_dict)
                loss.backward()
                optimizer.step()
                print_loss += loss.item()
                batch_loss += loss.item()
                cnt += 1
                if cnt % 50 == 0:
                    self.logger.info('Batch loss: %.6f', batch_loss / 50)
                    batch_loss = 0

            print_loss /= cnt
            model.eval()
            with torch.no_grad():
                val_loss = 0
                iters = 0
                for labels_cat, labels_pol, input_ids, attention_mask in val_dataloader:
                    encoded_dict = {
                        'input_ids': input_ids.to(device),
                        'attention_mask': attention_mask.to(device)
                    }
                    loss, _, _ = model(labels_cat.to(
                        device), labels_pol.to(device), **encoded_dict)
                    val_loss += loss.item()
                    iters += 1
                val_loss /= iters
            self.logger.info(f'epoch {epoch + 1}/{epochs}: train_loss={print_loss:.6f} val_loss={val_loss:.6f}')

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_model('casc_model')
                torch.save({
                    'model': _unwrap(model).state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch + 1,
                    'best_val_loss': best_val_loss,
                }, os.path.join(self.run_dir, 'casc_checkpoint.pth'))

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

    def evaluate(self, test_year=2016, test_type='test', label_type='single'):
        test_sentences, test_cats, test_pols = data.load_semeval(test_year, test_type, label_type)

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
                    batch,
                    padding='longest',
                    return_tensors='pt',
                    return_attention_mask=True,
                    max_length=128,
                    truncation=True).to(device)

                _, logits_cat, logits_pol = model(cats.to(device),
                                                  pols.to(device), **encoded)

                for sentence, cat, logit_cat, pol, logit_pol in zip(
                        batch, batch_cats, logits_cat, batch_pols, logits_pol):
                    rows.append([sentence, self.aspect_dict[cat],
                                 self.aspect_dict[logit_cat.argmax().item()],
                                 self.polarity_dict[pol],
                                 self.polarity_dict[logit_pol.argmax().item()]])

        df = pd.DataFrame(rows, columns=['sentence', 'actual category', 'predicted category', 'actual polarity', 'predicted polarity'])

        preds_path = os.path.join(self.run_dir, 'predictions.csv')
        df.to_csv(preds_path)
        self.logger.info(f'Wrote predictions to {preds_path}')

        print('Polarity')
        print(classification_report(df['actual polarity'], df['predicted polarity'], digits=4))
        print()

        print('Aspect')
        print(classification_report(df['actual category'], df['predicted category'], digits=4))
        print()
