import random
from config import *

domain = config['domain']
root = path_mapper[domain]
pol_classes = 2
asp_classes = 3



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

    f1_pol = F1Score(num_classes=pol_classes, average='macro', name='f1')
    f1_cat = F1Score(num_classes=asp_classes, average='macro', name='f1')

    model = LCRRothopPP(hop=3, hierarchy=(False, True), drop_1=drop_rate_1, drop_2=drop_rate_2, hidden_units=hidden_units, regularizer=regularizer)
    # model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['acc', f1])
    model.compile(optimizer=optimizer, loss=GCEQ(q), metrics={'cat': ['acc', f1_cat], 'pol': ['acc', f1_pol]})

    return model


def train_hyper(project_name, training_data: dict=None, val_data: dict=None, test_data: dict=None):
    # Instantiate the tuner
    tuner = kt.Hyperband(build_model,
                        objective=objective,
                        max_epochs=20,
                        factor=10,
                        hyperband_iterations=5,
                        directory="logs/hp",
                        project_name=project_name,)

    if training_data is not None:
        X_train, cat_train, pol_train = data.load_embedded(data.load_training_data, path=training_data['path'])
        pol_train = tf.one_hot(pol_train, pol_classes, dtype='int32')
        cat_train = tf.one_hot(cat_train, asp_classes, dtype='int32')
        y_train = {'cat': cat_train, 'pol': pol_train}
    
    if val_data is not None:
        if 'year' in val_data:
            X_val, cat_val, pol_val = data.load_embedded(data.load_semeval, year=val_data['year'], data_type='val', label_type=val_data['label_type'])
        elif 'path' in val_data:
            X_val, cat_val, pol_val = data.load_embedded(data.load_training_data, path=val_data['path'])
        pol_val = tf.one_hot(pol_val, pol_classes, dtype='int32')
        cat_val = tf.one_hot(cat_val, asp_classes, dtype='int32')
        y_val = {'cat': cat_val, 'pol': pol_val}
    
    if test_data is not None:
        X_test, cat_test, pol_test = data.load_embedded(data.load_semeval, year=test_data['year'], data_type='test', label_type=test_data['label_type'])
        # X_test, cat_test, pol_test = data.load_embedded(data.load_training_data2, path=f'{root}/test_embedding_2015', training_path=r'datasets\restaurant\label 2015 single.txt')
        pol_test = tf.one_hot(pol_test, pol_classes, dtype='int32')
        cat_test = tf.one_hot(cat_test, asp_classes, dtype='int32')
        y_test = {'cat': cat_test, 'pol': pol_test}
        print(X_test)

    if training_data is not None:
        if training_data == val_data:
            print(len(X_train), len(pol_train))
            tuner.search(X_train, y_train, validation_split=0.2, batch_size=32, callbacks=[stop_early], verbose=1)
        elif val_data is not None:
            tuner.search(X_train, y_train, validation_data=(X_val, y_val), batch_size=32, callbacks=[stop_early], verbose=1)
        else: # No validation
            tuner.search(X_train, y_train, validation_data=(X_val, y_val), batch_size=32, callbacks=[stop_early], verbose=1)
    elif val_data is not None:
        tuner.search(X_val, y_val, validation_split=0.2, batch_size=32, callbacks=[stop_early], verbose=1)
    
    if test_data is not None:
        models = tuner.get_best_models(num_models=5)
        best_model = models[0]
        print(tuner.get_best_hyperparameters(1)[0].__dict__)
        # best_model.evaluate(X_test, y_test, batch_size=16)
        

f1_pol = F1Score(num_classes=pol_classes, average='macro', name='f1')
f1_cat = F1Score(num_classes=asp_classes, average='macro', name='f1')

stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
objective = [kt.Objective("val_cat_acc", direction="max"), kt.Objective("val_pol_acc", direction="max"),
             kt.Objective("val_cat_f1", direction="max"), kt.Objective("val_pol_f1", direction="max")]

training_data={"path": f'{root}/training_embedding'}
train_hyper("LCR_MAX_SEP", training_data=training_data, val_data=training_data, test_data={"year": 2016, "label_type": 'multi'})
# train_hyper("2015_LCR_MULTI", val_data={"year": 2015, "label_type": 'multi'}, test_data={"year": 2015, "label_type": 'multi'})
# train_hyper("2016_LCR_MULTI", val_data={"year": 2016, "label_type": 'multi'}, test_data={"year": 2016, "label_type": 'multi'})

# train_hyper("2015_LCR_SINGLE", val_data={"year": 2015, "label_type": 'single'}, test_data={"year": 2015, "label_type": 'single'})
# train_hyper("2016_LCR_SINGLE", val_data={"year": 2016, "label_type": 'single'}, test_data={"year": 2016, "label_type": 'single'})

