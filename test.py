from config import *

domain = config['domain']
root = path_mapper[domain]
# import tensorflow as tf
# import numpy as np

# from embedding import embed_separated


# class LCRRothopPP(tf.keras.Model):
#     def __init__(self):
#         super().__init__()

#         self.probabilities = tf.keras.layers.Dense(3, 
#             tf.keras.layers.Activation('softmax'),
#             bias_initializer='zeros')
        
#     def call(self, inputs):
#         for i in inputs:
#             sentence = embed_separated(i.numpy())
#             print(sentence)

#         return [1, 2]


# if __name__ == '__main__':
#     x_train = ['hi [SEP] there [SEP] you', 'hoe [SEP] gaat [SEP] het']
#     y_train = [1, 2]
#     model = LCRRothopPP()
#     model.compile(optimizer='adam', 
#                   loss='categorical_crossentropy',
#                 #   metrics='acc',
#                   run_eagerly=True,
#     )
#     model.fit(x_train, y_train)

# import numpy as np
# from sklearn.model_selection import KFold
# X = np.array([[1, 2], [3, 4], [1, 2], [3, 4], [1, 2], [3, 4], [1, 2], [3, 4]])
# y = np.array([1, 2, 3, 4, 1, 2, 3, 4])
# kf = KFold(n_splits=3)
# print(kf.get_n_splits(X))
# print(kf)
# # KFold(n_splits=2, random_state=None, shuffle=False)
# for train_index, test_index in kf.split(X):
#     print("TRAIN:", train_index, "TEST:", test_index)
#     X_train, X_test = X[train_index], X[test_index]
#     y_train, y_test = y[train_index], y[test_index]
# # TRAIN: [2 3] TEST: [0 1]
# # TRAIN: [0 1] TEST: [2 3]

# import re

# sent = "merchant .is amazing . you truly ca n't amazing go wrong with the cocktails . the food is delicious , too . monday burger nights are the best . friendly staff and warm ambiance . definitely one of my favorite spots to grab a bite and a drink or two or three in madison ."

# pattern = r'amazing'

# split_sent = re.split(pattern, sent, maxsplit=1)

# separated_sentence = f"{split_sent[0]} [SEP] {pattern} [SEP] {split_sent[1]}"

# print(separated_sentence)

# import glob
# files = glob.iglob(f'{root}/training_embedding/*.np[yz]')
# for file in files:
#     print(file)

from lcr_rot_hop_plus_plus import LCRRothopPP
import data
from tensorflow_addons.metrics.f_scores import F1Score
import numpy as np
import tensorflow as tf

model = LCRRothopPP()
X, _, y = data.load_embedded_training()
y = tf.one_hot(y, 2, dtype='int32')
# print(y)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X, y, epochs=10, batch_size=16)
