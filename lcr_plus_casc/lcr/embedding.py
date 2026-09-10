from transformers import AutoTokenizer, BertModel
import torch

from ..config import *

def _bert_type():
    return bert_mapper[config['domain']]


def _cat(tok, key, *segments):
    tok[key] = torch.cat(segments, dim=1)


def _pad(length, device):
    return torch.zeros(1, length).int().to(device)


def embed_separated(tokens):
    bert = BertModel.from_pretrained(_bert_type(), output_hidden_states=True).to(config['device'])

    hidden_states = bert(**tokens)
    mean = torch.mean(torch.stack(hidden_states[2][-4:]), dim=0)
    return mean.cpu().detach().numpy()


def tokenize_separated(separated_sentence):
    tokenizer = AutoTokenizer.from_pretrained(_bert_type())
    device = config['device']
    tok = tokenizer(separated_sentence, return_tensors='pt').to(device)

    # separate left, target, right
    # the target segment is marked with [SEP] (id 102)
    target_mask = tok['input_ids'] == 102
    target_start, target_end = target_mask.nonzero()[:-1, 1]
    tok['token_type_ids'][:, target_start:target_end+1] = 1

    # Pad the left segment to left_max_length
    left_padding = left_max_length-target_start+1
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
    target_padding = target_max_length-(target_end-target_start)+1

    # Pad the target segment to target_max_length
    if target_padding > 0:
        padding = _pad(target_padding, device)
        _cat(tok, 'input_ids', tok['input_ids'][:, :target_start+1], padding, tok['input_ids'][:, target_start+1:])
        _cat(tok, 'token_type_ids', tok['token_type_ids'][:, :target_start+1], padding+1, tok['token_type_ids'][:, target_start+1:])
        _cat(tok, 'attention_mask', tok['attention_mask'][:, :target_start+1], padding, tok['attention_mask'][:, target_start+1:])
    else:
        _cat(tok, 'input_ids', tok['input_ids'][:, :target_start+1], tok['input_ids'][:, target_end-target_max_length:target_end], tok['input_ids'][:, target_end:])
        _cat(tok, 'token_type_ids', tok['token_type_ids'][:, :target_start+1], tok['token_type_ids'][:, target_end-target_max_length:target_end], tok['token_type_ids'][:, target_end:])
        _cat(tok, 'attention_mask', tok['attention_mask'][:, :target_start+1], tok['attention_mask'][:, target_end-target_max_length:target_end], tok['attention_mask'][:, target_end:])

    target_end += target_padding
    end = tok['input_ids'].size()[1]
    # Pad the right segment to right_max_length
    right_padding = right_max_length-(end-target_end)+2  # 2, 1 for first token, second for last

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

def main():
    print('-----------')
    embedding = tokenize_separated('hi there [SEP] Who are you [SEP] i dont know')
    print(embedding['input_ids'])
    print(embedding['input_ids'][0, 1:left_max_length+1])
    print(embedding['input_ids'][0, left_max_length+2:left_max_length+target_max_length+2])
    print(embedding['input_ids'][0, left_max_length+target_max_length+3:left_max_length+target_max_length+right_max_length+3])

    # import sys
    # import tensorflow as tf
    # print(sys.getsizeof(embedding))
    # print(sys.getsizeof(tf.convert_to_tensor(embedding)))

if __name__ == '__main__':
    main()
