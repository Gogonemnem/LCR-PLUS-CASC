# LCR+CASC
Code for Left-Center-Right Context-aware Aspect Category and Sentiment Classification. The code extends the CASC code from https://www.github.com/Raghu150999/UnsupervisedABSA.

## Setup
Python 3.10+ with the dependencies in `requirements.txt` (PyTorch is the only deep-learning framework; the original TensorFlow parts have been ported).

## Pipeline
Everything is driven from a single CLI. A named YAML recipe (`--config <name>`, defaulting to `casc.yaml` or `lcr.yaml`) supplies the hyperparameters; unlisted fields keep their dataclass defaults, and per-run CLI flags override the recipe.

1. `prep` — CASC-style unsupervised labeling: build the domain vocabularies (`datasets/<domain>/lexicons/`), extract aspect spans, score them with MLM replacements, and threshold into `datasets/<domain>/label.txt`.
2. `casc` — train/evaluate the baseline `BERTLinear` CASC model.
3. `lcr` — train/evaluate the two-tower `LCRRothopPP` model (with optional `--tune` Optuna search) using the unified GCE + L1/L2 loss.

Which phases run is implied by the data paths: `--train-path` trains, `--val-path` adds a validation split (else no validation), `--test-path` evaluates. Give both train and test to train then evaluate; give only `--test-path` to evaluate a saved checkpoint.

```
python main.py prep
python main.py casc --train-path <train.txt> --test-path <test.txt>
python main.py lcr --train-path <train.txt> --test-path <test.txt>
python main.py lcr --tune --train-path <train.txt>
python main.py lcr --test-path <test.txt> --load <run_dir>   # eval-only
```

Each run writes artifacts to `output/<config-name>/<timestamp>_<kind>/`: `config.json`, `run.log`, `*_predictions.csv`, plus `casc_model.pth` / `casc_checkpoint.pth` (CASC) or `lcr_model.pth` / `lcr_checkpoint.pth` (LCR). Resume a crashed run with `--resume` (newest such checkpoint), `--resume <run-dir>`, or `--resume <path/to/checkpoint.pth>`. `evaluate_test.py` gives a standalone sklearn classification report over the saved predictions.

Devices: by default every torch module is wrapped with `DataParallel` and fans batches out across all visible GPUs (labeling scoring and training alike). `--device cuda:N` pins one GPU; `--device cpu` forces CPU.

## Dataset layout
Per domain under `datasets/<domain>/`:
- `raw/` — original SemEval/Tang training data (XML + root `train.txt`/`test.txt`)
- `lexicons/` — auto-built seed lexicons (`dict_*.txt`)
- `intermediate/` — `scores.txt` and training embeddings
- `label.txt` — thresholded unsupervised aspect/polarity labels (training rows)
- `2015/`, `2016/` — per-year `train_{single,multiple}.txt` and `test_{single,multiple}.txt`

Regenerate the per-year files from the raw XMLs with `lcr_plus_casc/labeling/semeval_reader.py`, or re-run `prep` (delete `lexicons/dict_*.txt` and `intermediate/scores.txt` first for a full re-label).

## Config recipes
Under `configs/`:
- `casc.yaml` — BERTLinear defaults (GCE q, learning rate, batch size).
- `lcr.yaml` — LCR paper parameters (hidden 768, hop 6, q 0.4, l1=l2 1e-7, drop1 0.5 / drop2 0.3).
- `lcr_q01_lr3.yaml` — LCR Table-7 variant (q 0.1, lcr_learning_rate 1e-3).

## Layout
```
lcr_plus_casc/
  config.py          # TrainingConfig / DomainConfig dataclasses, recipe loader, CKPT names
  data.py            # dataset loaders: SemEval TSVs, unlabeled rows, tensor datasets
  filter_words.py    # shared stop-word list
  losses.py          # unified absa_loss (CE + GCE + L1/L2), metrics
  runs.py            # per-run output dirs and best/checkpoint artifact resolution
  labeling/          # unsupervised CASC preprocessing
    vocab.py, dictionary.py, split_file.py
    extracter.py     #   span extraction (spaCy + regex + heuristic)
    score_computer.py#   MLM top-k scoring + z-score aggregation
    labeler.py       #   thresholding -> label.txt
    mlm.py           #   masked-LM scorer (DataParallel-aware)
    semeval_reader.py, pipeline.py
  classifiers/
    casc.py          # BERTLinear baseline
    lcr.py           # LCRRothopPP two-tower model
    training.py      # AbsaTrainer (unified CASC/LCR train/tune/evaluate/progress)
main.py              # single CLI entry point
evaluate_test.py     # standalone sklearn report over saved predictions
configs/             # named YAML recipes (casc, lcr, lcr_q01_lr3)
```
