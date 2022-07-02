import random
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
import keras_tuner as kt

def GCEQ(q=0.4):

    @tf.function
    def GCE(y_true, y_pred):
        # print(y_true, y_pred)
        # y_true = y_true[y_true == 1]
        y = y_pred[y_true == 1]
        # print(y)
        loss = (1 - (y ** q)) / q
        # print(loss)
        return loss
    return GCE

# X_train, cat_train, pol_train = data.load_embedded(data.load_training_data, path=f'{root}/training_embedding')
# pol_train = tf.one_hot(pol_train, 2, dtype='int32')
# cat_train = tf.one_hot(cat_train, 3, dtype='int32')
# y_train = {'cat': cat_train, 'pol': pol_train}

X_val, cat_val, pol_val = data.load_embedded(data.load_semeval, year=2015, data_type='val', label_type='multi')
pol_val = tf.one_hot(pol_val, 2, dtype='int32')
cat_val = tf.one_hot(cat_val, 3, dtype='int32')
y_val = {'cat': cat_val, 'pol': pol_val}

f1_pol = F1Score(num_classes=2, average='macro', name='f1')
f1_cat = F1Score(num_classes=3, average='macro', name='f1')

# model = LCRRothopPP()
# model.compile(optimizer='adam', loss=GCEQ(.1), metrics={'cat': ['acc', f1_cat], 'pol': ['acc', f1_pol]}, run_eagerly=False)
# model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=16)

# model1 = LCRRothopPP()
# model1.compile(optimizer='adam', loss=GCEQ(.1), metrics={'cat': ['acc', f1_cat], 'pol': ['acc', f1_pol]}, run_eagerly=False)
# model1.fit(X_train, y_train, validation_split=.2, epochs=10, batch_size=32)

# # # model.evaluate(X_test, y_test, batch_size=16)
# model1.evaluate(X_test, y_test, batch_size=16)

def build_model(hp):
    tf.random.set_seed(0)
    random.seed(0)
    np.random.seed(0)

    # Tune regularizers rate for L1 regularizer with values from 0.001, 0.0001, 1e-05, 1e-06, 1e-07, 1e-08 or 1e-09
    hp_l1_rates = hp.Choice("l1_regularizer", values=[10**-i for i in range(3, 10)])

    # Tune regularizers rate for L2 regularizer with values from 0.001, 0.0001, 1e-05, 1e-06, 1e-07, 1e-08 or 1e-09
    hp_l2_rates = hp.Choice("l2_regularizer", values=[10**-i for i in range(3, 10)])

    regularizer = tf.keras.regularizers.L1L2(l1=hp_l1_rates, l2=hp_l2_rates)


    # Tune learning rate for Adam optimizer with values from 0.01, 0.001 & 0.0001
    hp_learning_rate = hp.Choice("learning_rate", values=[10**-i for i in range(2, 5)])

    optimizer = tf.keras.optimizers.Adam(learning_rate=hp_learning_rate)

    # Tune dropout layers with values from 0.2 - 0.7 with stepsize of 0.1.
    drop_rate_1 = hp.Float("dropout_1", 0.2, 0.6, step=0.1)
    drop_rate_2 = hp.Float("dropout_2", 0.2, 0.6, step=0.1)

    # Tune number of hidden layers for the BiLSTMs
    hidden_units = hp.Int("hidden_units", min_value=200, max_value=750, step=50)
    
    # Tune number of hops
    hidden_units = hp.Int("hops", min_value=1, max_value=8)

    # Tune qloss
    q = hp.Float("q", 0, 1, step=0.1)

    f1_pol = F1Score(num_classes=2, average='macro', name='f1')
    f1_cat = F1Score(num_classes=3, average='macro', name='f1')

    model = LCRRothopPP(hop=3, hierarchy=(False, True), drop_1=drop_rate_1, drop_2=drop_rate_2, hidden_units=hidden_units, regularizer=regularizer)
    # model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['acc', f1])
    model.compile(optimizer=optimizer, loss=GCEQ(q), metrics={'cat': ['acc', f1_cat], 'pol': ['acc', f1_pol]})

    return model

stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
objective = [kt.Objective("val_cat_acc", direction="max"), kt.Objective("val_pol_acc", direction="max"),
             kt.Objective("val_cat_f1", direction="max"), kt.Objective("val_pol_f1", direction="max")]

# Instantiate the tuner
tuner = kt.Hyperband(build_model,
                    objective=objective,
                    max_epochs=20,
                    factor=10,
                    hyperband_iterations=5,
                    directory="logs/hp",
                    project_name="2015_LCR_MULTI",)

# tuner.search(X_train, y_train, validation_data=(X_val, y_val), batch_size=32, callbacks=[stop_early], verbose=1)
tuner.search(X_val, y_val, validation_split=0.2, batch_size=32, callbacks=[stop_early], verbose=1)
print('test')
models = tuner.get_best_models(num_models=5)
print(tuner.get_best_hyperparameters(1)[0].__dict__)
best_model = models[0]


X_test, cat_test, pol_test = data.load_embedded(data.load_semeval, year=2016, data_type='test', label_type='multi')
pol_test = tf.one_hot(pol_test, 2, dtype='int32')
cat_test = tf.one_hot(cat_test, 3, dtype='int32')
y_test = {'cat': cat_test, 'pol': pol_test}
best_model.evaluate(X_test, y_test, batch_size=16)

X_test, cat_test, pol_test = data.load_embedded(data.load_semeval, year=2015, data_type='test', label_type='multi')
pol_test = tf.one_hot(pol_test, 2, dtype='int32')
cat_test = tf.one_hot(cat_test, 3, dtype='int32')
y_test = {'cat': cat_test, 'pol': pol_test}
best_model.evaluate(X_test, y_test, batch_size=16)
