import tensorflow as tf
import numpy as np
from math import floor, e


class Classifier(tf.keras.Model):
    def __init__(self, shape, num_cls=10, model='cnn',  regularization = 1.0, threshold=None, tau_method='exp'):
        super(Classifier, self).__init__()
        self.shape = shape
        self.num_cls = num_cls
        self.lam = regularization
        self.cap = -1.9999998 / e
        if (threshold == None):
            self.threshold = tf.Variable([0 for i in range (num_cls)], trainable=False)
        else:
            self.threshold = tf.Variable(threshold, trainable=False)
        self.tau_method = tau_method
        if self.tau_method == 'exp':
            self.tau = tf.Variable(0.0, trainable=False)
        else:
            self.tau = self.tau_method
        self.cap = tf.constant(self.cap)
        if (model == 'cnn'):
            self.model = tf.keras.Sequential(
                [
                    tf.keras.layers.InputLayer(input_shape=(self.shape)),
                    tf.keras.layers.Conv2D(
                        filters=64, kernel_size=3, strides=(2, 2), activation='relu'),
                    tf.keras.layers.MaxPool2D(2,2),
                    tf.keras.layers.Conv2D(
                        filters=128, kernel_size=3, strides=(2, 2), activation='relu'),
                    tf.keras.layers.Flatten(),
                    tf.keras.layers.Dense(128, activation='relu'),
                    # No activation
                    tf.keras.layers.Dense(self.num_cls),
                ]
            )
        elif (model == 'mlp'):
            self.model = tf.keras.Sequential(
                (
                    [
                        tf.keras.layers.InputLayer(input_shape=self.shape),
                        tf.keras.layers.Flatten(),
                        tf.keras.layers.Dense(
                            256),
                        tf.keras.layers.LeakyReLU(alpha=0.1),
                        tf.keras.layers.Dense(
                            128),
                        tf.keras.layers.LeakyReLU(alpha=0.1),

                        # No activation
                        tf.keras.layers.Dense(self.num_cls),
                    ]
                )
            )

    def projection(self, X):
        return self.model(X)

    def call(self, X):
        return self.model(X)

    def _accumulate_tau(self, loss, on_train=True):
        if self.tau_method == 'exp' and (on_train):
            self.tau.assign(self.tau -  0.1 * (self.tau - tf.reduce_mean(loss)))
        return self.tau

    def _accumulate_threshold(self, cls, value):
        self.threshold[cls].assign(self.threshold[cls] - value)



    def mnist_score(self, X, n_split=10, eps=1E-16):
        scores = list()
        n_part = floor(X.shape[0] / n_split)
        for i in range(n_split):
            # retrieve images
            ix_start, ix_end = i * n_part, (i + 1) * n_part
            subset = X[ix_start:ix_end]
            # convert from uint8 to float32
            subset = tf.cast(subset, tf.float32)
            p_yx = self.model.predict(subset)
            # calculate p(y)
            p_y = np.expand_dims(p_yx.mean(axis=0), 0)
            # calculate KL divergence using log probabilities
            kl_d = p_yx * (np.log(p_yx + eps) - np.log(p_y + eps))
            # sum over classes
            sum_kl_d = kl_d.sum(axis=1)
            # average over images
            avg_kl_d = np.mean(sum_kl_d)
            # undo the log
            is_score = np.exp(avg_kl_d)
            # store
            scores.append(is_score)
        # average across images
        return scores


class F_VAE(tf.keras.Model):
    def __init__ (self, data, shape=[28,28,1], beta=4, latent_dim=8, num_cls=10, model='cnn'):
        super(F_VAE, self).__init__()
        self.beta = beta
        self.data = data
        self.shape = shape
        self.num_cls = num_cls
        self.latent_dim = latent_dim
        self.output_f = int(shape[0] / 4)
        self.output_l = shape[2]
        self.output_s = shape[1]
        if (model == 'cnn'):
            self.encoder = tf.keras.Sequential(
                [
                    tf.keras.layers.InputLayer(input_shape=shape),
                    tf.keras.layers.Conv2D(
                        filters=32, kernel_size=5, strides=(1, 1), padding='same', use_bias=False),
                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.LeakyReLU(),
                    tf.keras.layers.Conv2D(
                        filters=64, kernel_size=5, strides=(2, 2), padding='same', use_bias=False),
                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.LeakyReLU(),
                    tf.keras.layers.Conv2D(
                        filters=128, kernel_size=5, strides=(2, 2), padding='same', use_bias=False),
                    tf.keras.layers.Flatten(),
                    # No activation
                    tf.keras.layers.Dense(latent_dim - 1 + latent_dim - 1),
                ]
            )

            self.decoder = tf.keras.Sequential(
                [
                    tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
                    tf.keras.layers.Dense(units=self.output_f * self.output_f * 32, activation=tf.nn.relu),
                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.LeakyReLU(),
                    tf.keras.layers.Reshape(target_shape=(self.output_f, self.output_f, 32)),
                    tf.keras.layers.Conv2DTranspose(
                        filters=128, kernel_size=5, strides=1, padding='same', use_bias=False),
                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.LeakyReLU(),
                    tf.keras.layers.Conv2DTranspose(
                        filters=64, kernel_size=3, strides=2, padding='same', use_bias=False),
                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.LeakyReLU(),
                    tf.keras.layers.Conv2DTranspose(
                        filters=32, kernel_size=3, strides=2, padding='same', use_bias=False),
                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.LeakyReLU(),
                    tf.keras.layers.Conv2DTranspose(
                        filters=self.output_l, kernel_size=3, strides=1, padding='same'),
                ]
            )
        elif (model == "mlp"):
            self.encoder = tf.keras.Sequential(
                [
                    tf.keras.layers.InputLayer(input_shape=shape),
                    tf.keras.layers.Dense(
                        64, activation='relu'),
                    tf.keras.layers.Dense(
                        32, activation='relu'),
                    tf.keras.layers.Flatten(),
                    # No activation
                    tf.keras.layers.Dense(latent_dim-1 + latent_dim-1),
                ]
            )
            self.decoder = tf.keras.Sequential(
                [
                    tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
                    tf.keras.layers.Dense(latent_dim * latent_dim, activation=tf.nn.relu),
                    tf.keras.layers.Dense(
                        512, activation='relu'),
                    tf.keras.layers.Dense(
                        self.output_f * 4 * self.output_s * 3,
                        activation='relu'),
                    # No activation
                    tf.keras.layers.Reshape(target_shape=[self.output_f*4, self.output_s, 3]),
                    tf.keras.layers.Dense(
                        1)
                ]
            )

    def sample(self, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(100, self.latent_dim))
        return self.decode(eps, apply_sigmoid=True)

    def encode(self, x):
        mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        z = eps * tf.exp(logvar * .5) + mean
        return z

    def decode(self, z, apply_sigmoid=False):
        logits = self.decoder(z)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs
        return logits

