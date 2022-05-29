from transformers import AutoTokenizer, BertModel
import torch

from config import *

domain = config['domain']
bert_type = bert_mapper[domain]

# Find padding
# attention_mask = input == 0
# tok['attention_mask'][attention_mask] = 0
# # print(torch.where(attention_mask, tok['attention_mask'], 0))

def embed_separated(tokens):
    bert = BertModel.from_pretrained(bert_type, output_hidden_states=True)

    bidden_states = bert(**tokens)
    mean = torch.mean(torch.stack(bidden_states[2][-4:]), dim=0)
    return mean.detach().numpy()

def tokenize_separated(separated_sentence):
    tokenizer = AutoTokenizer.from_pretrained(bert_type)
    tok = tokenizer(separated_sentence,
                    return_tensors='pt',)

    # print(tok)
    # separate left, target, right
    target_mask = tok['input_ids'] == 102
    target_start, target_end = target_mask.nonzero()[:-1, 1]
    tok['token_type_ids'][:, target_start:target_end+1] = 1

    # Add padding
    left_padding = left_max_length-target_start+1
    if left_padding > 0:
        padding = torch.zeros(1, left_padding).int()
        tok['input_ids'] = torch.cat((tok['input_ids'][:, :1], padding, tok['input_ids'][:, 1:]), dim=1)
        tok['token_type_ids'] = torch.cat((tok['token_type_ids'][:, :1], padding, tok['token_type_ids'][:, 1:]), dim=1)
        tok['attention_mask'] = torch.cat((tok['attention_mask'][:, :1], padding, tok['attention_mask'][:, 1:]), dim=1)

    else:
        tok['input_ids'] = torch.cat((tok['input_ids'][:, :1], tok['input_ids'][:, target_start-left_max_length:target_start], tok['input_ids'][:, target_start:]), dim=1) 
        tok['token_type_ids'] = torch.cat((tok['token_type_ids'][:, :1], tok['token_type_ids'][:, target_start-left_max_length:target_start], tok['token_type_ids'][:, target_start:]), dim=1) 
        tok['attention_mask'] = torch.cat((tok['attention_mask'][:, :1], tok['attention_mask'][:, target_start-left_max_length:target_start], tok['attention_mask'][:, target_start:]), dim=1) 

    target_start += left_padding
    target_end += left_padding
    target_padding = target_max_length-(target_end-target_start)+1

    if target_padding > 0:
        padding = torch.zeros(1, target_padding).int()
        tok['input_ids'] = torch.cat((tok['input_ids'][:, :target_start+1], padding, tok['input_ids'][:, target_start+1:]), dim=1)
        tok['token_type_ids'] = torch.cat((tok['token_type_ids'][:, :target_start+1], padding+1, tok['token_type_ids'][:, target_start+1:]), dim=1)
        tok['attention_mask'] = torch.cat((tok['attention_mask'][:, :target_start+1], padding, tok['attention_mask'][:, target_start+1:]), dim=1)

    else:
        tok['input_ids'] = torch.cat((tok['input_ids'][:, :target_start+1], tok['input_ids'][:, target_end-target_max_length:target_end], tok['input_ids'][:, target_end:]), dim=1)
        tok['token_type_ids'] = torch.cat((tok['token_type_ids'][:, :target_start+1], tok['token_type_ids'][:, target_end-target_max_length:target_end], tok['token_type_ids'][:, target_end:]), dim=1)
        tok['attention_mask'] = torch.cat((tok['attention_mask'][:, :target_start+1], tok['attention_mask'][:, target_end-target_max_length:target_end], tok['attention_mask'][:, target_end:]), dim=1)

    target_end += target_padding
    end = tok['input_ids'].size()[1]
    right_padding = right_max_length-(end-target_end)+2 # 2, 1 for first token, second for last

    # print(tok)
    if right_padding > 0:
        padding = torch.zeros(1, right_padding).int()
        tok['input_ids'] = torch.cat((tok['input_ids'][:, :-1], padding, tok['input_ids'][:, -1:]), dim=1)
        tok['token_type_ids'] = torch.cat((tok['token_type_ids'][:, :-1], padding, tok['token_type_ids'][:, -1:]), dim=1)
        tok['attention_mask'] = torch.cat((tok['attention_mask'][:, :-1], padding, tok['attention_mask'][:, -1:]), dim=1)
    else:
        tok['input_ids'] = torch.cat((tok['input_ids'][:, :target_end+right_max_length+1], tok['input_ids'][:, -1:]), dim=1)
        tok['token_type_ids'] = torch.cat((tok['token_type_ids'][:, :target_end+right_max_length+1], tok['token_type_ids'][:, -1:]), dim=1)
        tok['attention_mask'] = torch.cat((tok['attention_mask'][:, :target_end+right_max_length+1], tok['attention_mask'][:, -1:]), dim=1)

    # print(len(tok['input_ids'][0]))
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
