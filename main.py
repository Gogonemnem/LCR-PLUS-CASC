from vocab_generator import VocabGenerator
from extracter import Extracter
from score_computer import ScoreComputer
from labeler import Labeler
from trainer import Trainer

# import torch
# print(torch.cuda.is_available())

vocabGenerator = VocabGenerator()
# aspect_vocabularies, sentiment_vocabularies = vocabGenerator()
aspect_vocabularies, sentiment_vocabularies = vocabGenerator.from_folder()

extracter = Extracter()
sentences, aspects, opinions = extracter()

scoreComputer = ScoreComputer(aspect_vocabularies, sentiment_vocabularies)
scoreComputer(sentences, aspects, opinions)

labeler = Labeler()
labeler()

trainer = Trainer()
dataset = trainer.load_training_data()
trainer.train_model(dataset)
trainer.save_model('model')
# trainer.load_model('model')
trainer.evaluate()








