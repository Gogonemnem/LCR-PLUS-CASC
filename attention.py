import tensorflow as tf

class Attention(tf.keras.layers.Layer):
    def __init__(self, dim, regularizer, **kwargs):
        self.dim = dim
        self.regularizer = regularizer
        self.tanh = tf.keras.layers.Activation('tanh')
        self.softmax = tf.keras.layers.Softmax(axis=1)
        super().__init__(**kwargs)


class BilinearAttention(Attention):
    def __init__(self, dim, regularizer, **kwargs):
        super().__init__(dim, regularizer, **kwargs)

    def build(self, input_shape):
        self.weight = self.add_weight(
            name="bi_att_weight",
            shape=(self.dim, self.dim),
            initializer=tf.keras.initializers.RandomUniform(-0.1, 0.1),
            trainable=True,
            regularizer=self.regularizer
        )

        self.bias = self.add_weight(
            name="bi_att_bias",
            shape=(1, ),
            initializer='zeros',
            trainable=True,
            regularizer=self.regularizer
        )

        return super().build(input_shape)

    def call(self, inputs):
        hidden, pool_target = inputs[0], inputs[1]
        length = tf.shape(hidden)[1]

        # sizes: batch x L x 1 = batch x L x 2d @ 2d x 2d @ batch x 2d x 1
        # first_term = hidden @ self.weight @ pool_target
        first_term = tf.einsum('bik, bk -> bi', hidden @
                               self.weight, pool_target)
        # sizes: batch x L x 1 = 1 x 1 * batch x L x 1
        second_term = self.bias * tf.ones([length, ])

        func = self.tanh(first_term + second_term)
        alpha = self.softmax(func)

        # basically hidden^T @ alpha, but these are 3 dimensional tensors
        r = tf.einsum('bki,bk->bi', hidden, alpha)
        return r


class HierarchicalAttention(Attention):
    def __init__(self, dim, regularizer, **kwargs):
        super().__init__(dim, regularizer, **kwargs)

    def build(self, input_shape):
        self.weight = self.add_weight(
            name="hi_att_weight",
            shape=(self.dim, 1),
            initializer=tf.keras.initializers.RandomUniform(-0.1, 0.1),
            trainable=True,
            regularizer=self.regularizer
        )

        self.bias = self.add_weight(
            name="hi_att_bias",
            shape=(1, ),
            initializer='zeros',
            trainable=True,
            regularizer=self.regularizer
        )

        return super().build(input_shape)

    def call(self, representations):
        length = tf.shape(representations)[1]

        first_term = representations @ self.weight
        second_term = self.bias * tf.ones([length, 1]) # , 1 needed bc tensorflow weird

        func = self.tanh(first_term + second_term)
        alpha = self.softmax(func)

        representations *= alpha
        return representations
