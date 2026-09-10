# LCR+CASC
Code for Left-Center-Right Context-aware Aspect Category and Sentiment Classification. The code extends the CASC code from https://www.github.com/Raghu150999/UnsupervisedABSA.

## Setup
Python 3.10+ with the dependencies in `requirements.txt` (PyTorch is the only deep-learning framework; the original TensorFlow parts have been ported).

## Pipeline
1. `prep` — CASC-style unsupervised preprocessing: build the domain vocabulary, extract aspect terms, compute scores, and label polarity/aspect (Huang et al. 2020 Yelp restaurant reviews, built on the Tang et al. 2016 SemEval task).
2. `embed` — embed all (aspect, context) pairs with BERT-DK so the neural models read pre-computed tensors from disk (no mixing of frameworks at train time).
3. `casc` — train/evaluate the baseline `BERTLinear` CASC model.
4. `lcr` — train/evaluate (or Optuna-tune) the two-tower `LCRRothopPP` model with the GCE loss.

Everything is driven from a single CLI:
```
python main.py --domain restaurant [--device cuda:0] prep
python main.py --domain restaurant [--device cuda:0] embed --split train
python main.py --domain restaurant [--device cuda:0] casc   [--epochs ... --batch_size ...]
python main.py --domain restaurant [--device cuda:0] lcr    [--tune | --eval]
python main.py --domain restaurant [--device cuda:0] all
```
`--device cpu` forces CPU; on a multi-GPU box the trainers auto-wrap in `DataParallel` when >1 CUDA device is visible. `evaluate_test.py` gives a standalone sklearn classification report for a saved model.

## Layout
```
lcr_plus_casc/
  config.py          # shared paths/splits/label maps
  filter_words.py    # shared stop-word list
  data.py            # shared dataset loaders (both pipelines)
  casc/              # CASC pipeline (BERT-DK + Linear baseline)
    model.py         #   BERTLinear, LQLoss
    vocab_generator.py / extracter.py / score_computer.py / labeler.py
    semeval_reader.py
    trainer.py       #   Trainer for BERTLinear
  lcr/               # LCR pipeline (two-tower RoTHop++)
    lcr_model.py     #   LCRRothopPP
    embedding.py     #   tokenize/embed helpers
    trainer.py       #   LCRTrainer, GCEQ loss
```
