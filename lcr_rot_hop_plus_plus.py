import tensorflow as tf
from tensorflow_addons.layers import AdaptiveAveragePooling1D
from attention import BilinearAttention, HierarchicalAttention
from config import *


class LCRRothopPP(tf.keras.Model):
    def __init__(self, invert: bool = False, hop: int = 1, hierarchy: tuple = None, drop_1: float = 0.2, drop_2: float = 0.5, hidden_units: int = None, regularizer=None):
        """Creates a new LCR-Rot-hop++ model described in Trusca's paper.

        Args:
            data_paths (list, optional): List of paths to the training & validation data for GloVe Embeddings. Defaults to None.
            embedding_path (str, optional): Path to the GloVe embeddings. Defaults to None.
            invert (bool, optional): If true, invert the order of 2-step approach. Defaults to False (Target2Context then Context2Target).
            hop (int, optional): Number of hops. Defaults to 1. If 0 hops, there is no attention layer applied.
            hierarchy (tuple, optional): Tuple which decides which of the 4 hierarchical attention methods is used. First item: whether to combine the softmaxlayers. Second item: whether to apply the layer iteratively. Defaults to None, no hierarchical attention used.
            drop_1 (float, optional): Float between 0 and 1. Fraction of the input units to drop. Defaults to 0.2.
            drop_2 (float, optional): Float between 0 and 1. Fraction of the output neurons to drop. Defaults to 0.5.
            hidden_units (int, optional): Number of hidden units in the LSTM layer. Thus half the number of hidden units in the BiLSTM layer. If None, number is equal to the embedding dimension. Defaults to None.
            regularizer (_type_, optional): A tensorflow/keras regularizer. E.g., L1, L2 or L1 & L2. Defaults to no regularizer.

        Make sure to use sensible inputs, no checks are made.
        """
        super().__init__()

        # Determines which of the described methods in the papers from Wallaart & Trusca
        self.invert = invert
        self.hop = hop
        self.hierarchy = hierarchy

        # According to https://arxiv.org/pdf/1207.0580.pdf
        self.drop_input = tf.keras.layers.Dropout(drop_1)
        self.drop_output = tf.keras.layers.Dropout(drop_2)

        
        self.hidden_units = hidden_units
        self.regularizer = regularizer

        self.average_pooling = AdaptiveAveragePooling1D(1)

        # 'MLP' layer
        self.probabilities = tf.keras.layers.Dense(2,
            tf.keras.layers.Activation('softmax'),
            bias_initializer='zeros',
            kernel_regularizer=regularizer,
            bias_regularizer=regularizer
        )

        if self.hidden_units is None:
            self.hidden_units = 768
        
        # BiLSTM layers
        self.left_bilstm = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(self.hidden_units, return_sequences=True)
        )
        self.target_bilstm = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(self.hidden_units, return_sequences=True)
        )
        self.right_bilstm = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(self.hidden_units, return_sequences=True)
        )

        # Attention layers
        self.attention_left = BilinearAttention(
            2*self.hidden_units, self.regularizer)
        self.attention_right = BilinearAttention(
            2*self.hidden_units, self.regularizer)
        self.attention_target_left = BilinearAttention(
            2*self.hidden_units, self.regularizer)
        self.attention_target_right = BilinearAttention(
            2*self.hidden_units, self.regularizer)

        # Hierarchical attention layers
        if self.hierarchy is not None:
            if self.hierarchy[0]:  # combine all
                self.hierarchical = HierarchicalAttention(
                    2*self.hidden_units, self.regularizer)
            else:  # separate inner & outer
                self.hierarchical_inner = HierarchicalAttention(
                    2*self.hidden_units, self.regularizer)
                self.hierarchical_outer = HierarchicalAttention(
                    2*self.hidden_units, self.regularizer)

    # def build(self, input_shape):
    #     if self.hidden_units is None:
    #         self.hidden_units = input_shape[-1]
        
    #     # BiLSTM layers
    #     self.left_bilstm = tf.keras.layers.Bidirectional(
    #         tf.keras.layers.LSTM(self.hidden_units, return_sequences=True)
    #     )
    #     self.target_bilstm = tf.keras.layers.Bidirectional(
    #         tf.keras.layers.LSTM(self.hidden_units, return_sequences=True)
    #     )
    #     self.right_bilstm = tf.keras.layers.Bidirectional(
    #         tf.keras.layers.LSTM(self.hidden_units, return_sequences=True)
    #     )

    #     # Attention layers
    #     self.attention_left = BilinearAttention(
    #         2*self.hidden_units, self.regularizer)
    #     self.attention_right = BilinearAttention(
    #         2*self.hidden_units, self.regularizer)
    #     self.attention_target_left = BilinearAttention(
    #         2*self.hidden_units, self.regularizer)
    #     self.attention_target_right = BilinearAttention(
    #         2*self.hidden_units, self.regularizer)

    #     # Hierarchical attention layers
    #     if self.hierarchy is not None:
    #         if self.hierarchy[0]:  # combine all
    #             self.hierarchical = HierarchicalAttention(
    #                 2*self.hidden_units, self.regularizer)
    #         else:  # separate inner & outer
    #             self.hierarchical_inner = HierarchicalAttention(
    #                 2*self.hidden_units, self.regularizer)
    #             self.hierarchical_outer = HierarchicalAttention(
    #                 2*self.hidden_units, self.regularizer)

    def call(self, inputs):
        """Describes the model by relating the layers

        Args:
            inputs: List of the left context, target, and right context

        Returns: Probabilities per class
        """
        # Separate inputs: LCR
        # input_left, input_target, input_right = inputs[0], inputs[1], inputs[2]
        
        input_left, input_target, input_right = inputs[0], inputs[1], inputs[2]
        input_left = inputs[:, 1:left_max_length+1]
        input_target = inputs[:, left_max_length+2:left_max_length+target_max_length+2]
        input_right = inputs[:, left_max_length+target_max_length+3:left_max_length+target_max_length+right_max_length+3]

        # BiLSTMs
        # input_left = self.drop_input(input_left)
        left_bilstm = self.left_bilstm(input_left)

        # input_target = self.drop_input(input_target)
        target_bilstm = self.target_bilstm(input_target)

        # input_right = self.drop_input(input_right)
        right_bilstm = self.right_bilstm(input_right)

        # Representations
        representation_target_left = representation_target_right = self.average_pooling(
            target_bilstm)[:, 0, :]

        representation_left = self.average_pooling(left_bilstm)[:, 0, :]
        representation_right = self.average_pooling(right_bilstm)[:, 0, :]

        # Attention layers
        # for hop == 0, this loop is skipped -> LCR model (no attention, no rot)
        for _ in range(self.hop):
            representation_left, representation_target_left, representation_target_right, representation_right = self._apply_bilinear_attention(
                left_bilstm, target_bilstm, right_bilstm, representation_left, representation_target_left, representation_target_right, representation_right)

            if self.hierarchy is not None and self.hierarchy[1]:
                representation_left, representation_target_left, representation_target_right, representation_right = self._apply_hierarchical_attention(
                    representation_left, representation_target_left, representation_target_right, representation_right)

        if self.hierarchy is not None and (not self.hierarchy[1] or self.hop == 0):
            representation_left, representation_target_left, representation_target_right, representation_right = self._apply_hierarchical_attention(
                representation_left, representation_target_left, representation_target_right, representation_right)

        # MLP
        v = tf.concat([representation_left, representation_target_left,
                      representation_target_right, representation_right], axis=1)
        v = self.drop_output(v)

        pred = self.probabilities(v)
        return pred

    def _apply_bilinear_attention(self, left_bilstm, target_bilstm, right_bilstm, representation_left, representation_target_left, representation_target_right, representation_right):
        """Applies the attention layer described by in the paper"""
        if self.invert:
            representation_target_left = self.attention_target_left(
                [target_bilstm, representation_left])
            representation_target_right = self.attention_target_right(
                [target_bilstm, representation_right])

        representation_left = self.attention_left(
            [left_bilstm, representation_target_left])
        representation_right = self.attention_right(
            [right_bilstm, representation_target_right])

        if not self.invert:
            representation_target_left = self.attention_target_left(
                [target_bilstm, representation_left])
            representation_target_right = self.attention_target_right(
                [target_bilstm, representation_right])

        return representation_left, representation_target_left, representation_target_right, representation_right

    def _apply_hierarchical_attention(self, representation_left, representation_target_left, representation_target_right, representation_right):
        """Applies the hierarchical attention layer described by in the paper"""
        if self.hierarchy[0]:  # combine all
            representations = tf.stack(
                [representation_left, representation_target_left, representation_target_right, representation_right], axis=1)
            representation_left, representation_target_left, representation_target_right, representation_right = tf.unstack(
                self.hierarchical(representations), axis=1)
        else:  # separate inner & outer
            representations = tf.stack(
                [representation_left, representation_right], axis=1)
            representation_left, representation_right = tf.unstack(
                self.hierarchical_outer(representations), axis=1)

            representations = tf.stack(
                [representation_target_left, representation_target_right], axis=1)
            representation_target_left, representation_target_right = tf.unstack(
                self.hierarchical_inner(representations), axis=1)
        return representation_left, representation_target_left, representation_target_right, representation_right
