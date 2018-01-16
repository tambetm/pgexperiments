from keras import backend as K

def sample_multinomial(x, from_logits=False):
    if K.backend() == 'theano':
        from theano.tensor.shared_randomstreams import RandomStreams
        random = RandomStreams()
        if from_logits:
            # TODO: there is a more direct way from logits
            return random.multinomial(pvals=K.softmax(x))
        else:
            return random.multinomial(pvals=x)
    elif K.backend() == 'tensorflow':
        import tensorflow as tf
        shape = K.shape(x)
        if not from_logits:
            x = tf.clip_by_value(x, K.epsilon(), 1 - K.epsilon())
            x = tf.log(x)
        return K.reshape(tf.one_hot(tf.multinomial(K.reshape(x, (-1, shape[-1])), 1), shape[-1]), shape)
    else:
        raise NotImplementedError

def sparse_sample_multinomial(x, from_logits=False):
    if K.backend() == 'theano':
        from theano.tensor.shared_randomstreams import RandomStreams
        random = RandomStreams()
        if from_logits:
            # TODO: there is a more direct way from logits
            return K.argmax(random.multinomial(pvals=K.softmax(x)))
        else:
            return K.argmax(random.multinomial(pvals=x))
    elif K.backend() == 'tensorflow':
        import tensorflow as tf
        shape = K.shape(x)
        if not from_logits:
            x = tf.clip_by_value(x, K.epsilon(), 1 - K.epsilon())
            x = tf.log(x)
        return K.reshape(tf.multinomial(K.reshape(x, (-1, shape[-1])), 1), shape[:-1])
    else:
        raise NotImplementedError


def sample_binomial(x, from_logits=False):
    if K.backend() == 'theano':
        from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
        random = RandomStreams()
        s = random.uniform(x.shape, low=0, high=1)
    elif K.backend() == 'tensorflow':
        import tensorflow as tf
        s = tf.random_uniform(tf.shape(x), minval=0, maxval=1)
    else:
        raise NotImplementedError

    if from_logits:
        # TODO: there might be more direct way from logits
        return K.cast(K.sigmoid(x) > s, K.floatx())
    else:
        return K.cast(x > s, K.floatx())


if __name__ == '__main__':
    import numpy as np
    x = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    y = K.eval(sample_multinomial(K.variable(x)))
    assert np.all(y == x), y
    y = K.eval(sparse_sample_multinomial(K.variable(x)))
    assert np.all(y == [0, 1, 2]), y
