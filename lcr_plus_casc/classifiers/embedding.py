"""Tokenization and BERT embedding of left/target/right separated sentences."""

import torch
from transformers import AutoTokenizer, BertModel

from ..config import bert_mapper, config, left_max_length, right_max_length, target_max_length


def _bert_type():
    return bert_mapper[config['domain']]


def _cat(tok, key, *segments):
    tok[key] = torch.cat(segments, dim=1)


def _pad(length, device):
    return torch.zeros(1, length).int().to(device)


def embed_separated(tokens):
    """Mean-pool the last 4 hidden layers of BERT on a tokenized sentence."""
    bert = BertModel.from_pretrained(_bert_type(), output_hidden_states=True).to(config['device'])

    hidden_states = bert(**tokens)
    mean = torch.mean(torch.stack(hidden_states[2][-4:]), dim=0)
    return mean.cpu().detach().numpy()


def tokenize_separated(separated_sentence):
    """Tokenize a '[SEP]' separated sentence and pad each segment.

    Returns a BatchEncoding on ``config['device']`` whose ``input_ids`` are laid
    out as [CLS] left(left_max_length) [SEP] target(target_max_length) [SEP]
    right(right_max_length), with matching token_type_ids / attention_mask.
    """
    tokenizer = AutoTokenizer.from_pretrained(_bert_type())
    device = config['device']
    tok = tokenizer(separated_sentence, return_tensors='pt').to(device)

    # separate left, target, right
    # the target segment is marked with [SEP] (id 102)
    target_mask = tok['input_ids'] == 102
    target_start, target_end = target_mask.nonzero()[:-1, 1]
    tok['token_type_ids'][:, target_start:target_end+1] = 1

    # Pad the left segment to left_max_length
    left_padding = left_max_length - target_start + 1
    if left_padding > 0:
        padding = _pad(left_padding, device)
        _cat(tok, 'input_ids', tok['input_ids'][:, :1], padding, tok['input_ids'][:, 1:])
        _cat(tok, 'token_type_ids', tok['token_type_ids'][:, :1], padding, tok['token_type_ids'][:, 1:])
        _cat(tok, 'attention_mask', tok['attention_mask'][:, :1], padding, tok['attention_mask'][:, 1:])
    else:
        _cat(tok, 'input_ids', tok['input_ids'][:, :1], tok['input_ids'][:, target_start-left_max_length:target_start], tok['input_ids'][:, target_start:])
        _cat(tok, 'token_type_ids', tok['token_type_ids'][:, :1], tok['token_type_ids'][:, target_start-left_max_length:target_start], tok['token_type_ids'][:, target_start:])
        _cat(tok, 'attention_mask', tok['attention_mask'][:, :1], tok['attention_mask'][:, target_start-left_max_length:target_start], tok['attention_mask'][:, target_start:])

    target_start += left_padding
    target_end += left_padding
    target_padding = target_max_length - (target_end - target_start) + 1

    # Pad the target segment to target_max_length
    if target_padding > 0:
        padding = _pad(target_padding, device)
        _cat(tok, 'input_ids', tok['input_ids'][:, :target_start+1], padding, tok['input_ids'][:, target_start+1:])
        _cat(tok, 'token_type_ids', tok['token_type_ids'][:, :target_start+1], padding + 1, tok['token_type_ids'][:, target_start+1:])
        _cat(tok, 'attention_mask', tok['attention_mask'][:, :target_start+1], padding, tok['attention_mask'][:, target_start+1:])
    else:
        _cat(tok, 'input_ids', tok['input_ids'][:, :target_start+1], tok['input_ids'][:, target_end-target_max_length:target_end], tok['input_ids'][:, target_end:])
        _cat(tok, 'token_type_ids', tok['token_type_ids'][:, :target_start+1], tok['token_type_ids'][:, target_end-target_max_length:target_end], tok['token_type_ids'][:, target_end:])
        _cat(tok, 'attention_mask', tok['attention_mask'][:, :target_start+1], tok['attention_mask'][:, target_end-target_max_length:target_end], tok['attention_mask'][:, target_end:])

    target_end += target_padding
    end = tok['input_ids'].size()[1]
    # Pad the right segment to right_max_length
    right_padding = right_max_length - (end - target_end) + 2  # 2, 1 for first token, second for last

    if right_padding > 0:
        padding = _pad(right_padding, device)
        _cat(tok, 'input_ids', tok['input_ids'][:, :-1], padding, tok['input_ids'][:, -1:])
        _cat(tok, 'token_type_ids', tok['token_type_ids'][:, :-1], padding, tok['token_type_ids'][:, -1:])
        _cat(tok, 'attention_mask', tok['attention_mask'][:, :-1], padding, tok['attention_mask'][:, -1:])
    else:
        _cat(tok, 'input_ids', tok['input_ids'][:, :target_end+right_max_length+1], tok['input_ids'][:, -1:])
        _cat(tok, 'token_type_ids', tok['token_type_ids'][:, :target_end+right_max_length+1], tok['token_type_ids'][:, -1:])
        _cat(tok, 'attention_mask', tok['attention_mask'][:, :target_end+right_max_length+1], tok['attention_mask'][:, -1:])

    return tok
