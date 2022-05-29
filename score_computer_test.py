from re import S
from transformers import AutoTokenizer, BertForMaskedLM
from config import *
from tqdm import tqdm
import torch
from filter_words import filter_words

class ScoreComputer:
    '''
    Computes unnormalised overlap scores for each aspect category and sentiment polarity and saves in "scores.txt" file
    '''
    def __init__(self, aspect_vocabularies, sentiment_vocabularies):
        self.domain = config['domain']
        self.bert_type = bert_mapper[self.domain]
        self.device = config['device']
        self.mlm_model = BertForMaskedLM.from_pretrained(self.bert_type).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.bert_type)
        self.root_path = path_mapper[self.domain]
        self.aspect_vocabularies = aspect_vocabularies
        self.sentiment_vocabularies = sentiment_vocabularies
    
    def __call__(self, sentences, aspects, opinions):
        categories = aspect_category_mapper[self.domain]
        polarities = sentiment_category_mapper[self.domain]
        K = K_2

        aspect_sets = self.load_vocabulary(self.aspect_vocabularies, M[self.domain])
        polarity_sets = self.load_vocabulary(self.sentiment_vocabularies, M[self.domain])

        f = open(f'{self.root_path}/scores.txt', 'w', encoding='utf-8')
        
        for sentence, aspect, opinion in tqdm(zip(sentences, aspects, opinions)):
            if opinion == '##' or aspect == '##': # or len(aspect) == 0:
                continue
            
            aspect_words = set(aspect.split())
            total_aspects = len(aspect_words)
            aspect_words = set(aspect_words)
            opinion_words = set(opinion.split())
            total_opinions = len(opinion_words)
            opinion_words = set(opinion_words)
            
            cat_scores = {cat: -1 for cat in categories}
            cat_words = {cat: '##' for cat in categories}
            pol_scores = {pol: -1 for pol in polarities}
            pol_words = {pol: '##' for pol in polarities}

            # tokenize and predict words from input sentence
            ids = self.tokenizer(sentence, return_tensors='pt', truncation=True)['input_ids']
            tokens = self.tokenizer.convert_ids_to_tokens(ids[0])
            word_predictions = self.mlm_model(ids.to(self.device))[0]
            word_scores, word_ids = torch.topk(word_predictions, K, -1)
            word_ids = word_ids.squeeze(0)

            for idx, token in enumerate(tokens):
                if token in aspect_words:
                    replacements = self.tokenizer.convert_ids_to_tokens(word_ids[idx])
                    for cat in categories:
                        asp_score = 0
                        
                        for repl in replacements:
                            if repl in filter_words or '##' in repl:
                                continue
                            if repl in aspect_sets[cat]:
                                asp_score += 1
                    
                        if asp_score > cat_scores[cat]:
                            cat_scores[cat] = asp_score
                            cat_words[cat] = token

                if token in opinion_words:
                    replacements = self.tokenizer.convert_ids_to_tokens(word_ids[idx])
                    for pol in polarities:
                        opi_score = 0

                        for repl in replacements:
                            if repl in filter_words or '##' in repl:
                                continue
                            if repl in polarity_sets[pol]:
                                opi_score += 1

                        if opi_score > pol_scores[pol]:
                            pol_scores[pol] = opi_score
                            pol_words[pol] = token
            summary = f'{sentence}\n'
            # for cat in categories:
            #     val = cat_scores[cat]
            #     target = cat_scores[cat]
            for cat in categories:
                val = cat_scores[cat] #/ total_aspects
                word = cat_words[cat]
                summary = summary + f' {cat}: {val}, {word}'
            
            for pol in polarities:
                val = pol_scores[pol] #/ total_opinions
                word = pol_words[pol]
                summary = summary + f' {pol}: {val}, {word}'

            f.write(summary)
            f.write('\n')
        f.close()
        

    def load_vocabulary(self, source, limit):
        target = {}
        for key in source:
            words = []
            for freq, word in source[key][:limit]:
                words.append(word)
            target[key] = set(words)
        return target
