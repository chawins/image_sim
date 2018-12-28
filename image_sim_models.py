import tensorflow as tf
import numpy as np
import keras
import keras.backend as K
from keras.layers import (Activation, Dense, Flatten, Lambda, Conv2D, Input,
                          MaxPooling2D, Reshape, Concatenate, Cropping2D, Add,
                          Dropout)
from keras.preprocessing.image import ImageDataGenerator

from stn.spatial_transformer import SpatialTransformer
from stn.conv_model import locnet_v3

# This class is adapted from https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly

import numpy as np
import keras


class DataGenerator(keras.utils.Sequence):
    """
    Generates data for Keras

    Args:
        list_IDs: (list of str) list of paths to file to load
        labels: (list of int) list of labels correspond to <list_IDs>
    """

    def __init__(self, X, y, batch_size=128, shuffle=True):
        'Initialization'
        self.batch_size = batch_size
        self.X = X
        self.y = y
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.X) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index + 1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.X))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        """
        Generates data containing batch_size samples 
        X : (n_samples, *dim)
        """
        # Initialization
        X = np.empty((self.batch_size, *self.dim))
        y = np.empty((self.batch_size), dtype=np.int32)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # TODO: Load samples from files
            X[i,] = np.load('data/' + ID + '.npy')

            # Store class
            y[i] = self.labels[ID]

        return X, y


class SiameseNetwork(object):

    def __init__(self, scope, input_shape, output_shape=100, margin=1,
                 learning_rate=1e-3, 
                 reg=0, load_model=True, save_path="model/siam.h5"):
        """
        Args:
            output_shape: (int) embedding dimension
        """

        self.scope = scope
        self.save_path = save_path
        self.output_shape = output_shape
        self.height, self.width, self.channel = input_shape

        # Create placeholders
        self.x1 = tf.placeholder(tf.float32, [None, ] + input_shape, name="x1")
        self.x2 = tf.placeholder(tf.float32, [None, ] + input_shape, name="x2")
        # y is 1 if x1 and x2 are from the same class. y is 0 otherwise
        self.y = tf.placeholder(tf.float32, [None, 1], name="y")
        
        # =========================== Build model =========================== #
        inpt = Input(shape=input_shape)
        
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):

            u = Conv2D(32, (3, 3), activation="relu")(inpt)
            u = Conv2D(64, (3, 3), activation="relu")(u)
            # u = Conv2D(128, (3, 3), activation="relu")(u)
            u = Flatten()(u)
            # u = Dense(1024, activation="relu")(u)
            # u = Dropout(0.5)(u)
            u = Dense(256, activation="relu")(u)
            u = Dropout(0.25)(u)
            u = Dense(output_shape, activation="relu")(u)

            self.model = keras.models.Model(inputs=inpt, outputs=u)
            self.embed1 = self.model(self.x1)
            self.embed2 = self.model(self.x2)

        # Weight regularization
        self.reg_loss = 0
        for l in self.model.layers:
            w = l.weights
            if len(w) != 0:
                self.reg_loss += tf.reduce_sum(tf.square(w[0]))

        # Calculate loss
        self.dist = tf.sqrt(tf.reduce_sum(tf.square(self.embed1 - self.embed2), -1))
        # Square of L2 distance
        loss = self.y*tf.square(self.dist) + \
               (1 - self.y)*tf.square(tf.maximum(0., margin - self.dist))
        # loss = self.y*tf.pow(self.dist, 3) + \
        #        (1 - self.y)*tf.pow(tf.maximum(0., margin - self.dist), 3)
        self.loss = tf.reduce_mean(loss)
        self.total_loss = self.loss + reg*self.reg_loss

        var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)

        # Set up optimizer
        with tf.variable_scope(scope + "_opt"):
            optimizer = tf.train.AdamOptimizer(learning_rate)
            self.train_op = optimizer.minimize(self.total_loss, var_list=var_list)

        opt_var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                         scope=scope + "_opt")
        self.init = tf.variables_initializer(var_list=var_list + opt_var_list)

        if load_model:
            try:
                self.model.load_weights(self.save_path)
            except OSError:
                print("Saved weights not found...")
                print("Model was built, but no weight was loaded")

    def get_output(self, x):
        return self.model(x)

    def train_model(self, sess, data, dataaug=False, n_epoch=10, batch_size=128):

        x_train, y_train, x_val, y_val = data
        n_train, n_val = len(x_train), len(x_val)

        datagen = ImageDataGenerator(
            rotation_range=5,
            width_shift_range=0.05,
            height_shift_range=0.05,
            zoom_range=0.05,
            channel_shift_range=0.1,
            brightness_range=(0.9, 1.1))

        # Initilize all network variables
        sess.run(self.init)

        best_val_loss = 1e9
        n_step = np.ceil(n_train / float(batch_size)).astype(np.int32)
        ind1 = np.arange(n_train)
        ind2 = np.arange(n_train)

        for epoch in range(n_epoch):
            print("============= EPOCH: {} =============".format(epoch))
            # Need to set learning phase to 1 every epoch because model_eval()
            # is also called at the end of every epoch
            K.set_learning_phase(1)

            # Training steps
            if not dataaug:
                np.random.shuffle(ind1)
                np.random.shuffle(ind2)
                for step in range(n_step):
                    start = step * batch_size
                    end = (step + 1) * batch_size
                    feed_dict = {self.x1: x_train[ind1[start:end]], 
                                 self.x2: x_train[ind2[start:end]], 
                                 self.y: (y_train[ind1[start:end]] == 
                                          y_train[ind2[start:end]])}
                    _, loss = sess.run([self.train_op, self.loss], 
                                    feed_dict=feed_dict)
                    if step % 50 == 0:
                        print("STEP: {} \tLoss: {:.4f}".format(step, loss))
            else:
                step = 0
                for x_batch, y_batch in datagen.flow(x_train, y_train, batch_size=2*batch_size):
                    # Using brightness_range in datagen rescales to 255
                    x_batch /= 255.
                    x_batch1, y_batch1 = x_batch[:batch_size], y_batch[:batch_size]
                    x_batch2, y_batch2 = x_batch[batch_size:], y_batch[:batch_size]
                    _, loss = sess.run([self.train_op, self.loss], 
                                        feed_dict={self.x1: x_batch1, 
                                                   self.x2: x_batch2,
                                                   self.y: y_batch1 == y_batch2})
                    if step % 50 == 0:
                        print("STEP: {} \tLoss: {:.4f}".format(step, loss))
                    if step > n_step:
                        break
                    step += 1

            # Print progress
            _, dist_same, dist_diff, loss = self.eval_model(sess, (x_train, y_train))
            print("Train dist_same|dist_diff|loss:\t{:.4f}|{:.4f}|{:.4f}".format(
                dist_same, dist_diff, loss))
            _, dist_same, dist_diff, loss = self.eval_model(sess, (x_val, y_val))
            print("Val dist_same|dist_diff|loss:\t{:.4f}|{:.4f}|{:.4f}".format(
                dist_same, dist_diff, loss))

            if loss < best_val_loss:
                best_val_loss = loss
                # Save model
                self.model.save_weights(self.save_path)

        # Restore to the best saved model
        self.model.load_weights(self.save_path)
    
    def get_embed(self, sess, x, batch_size=128):

        n_samples = len(x)
        n_step = np.ceil(n_samples / float(batch_size)).astype(np.int32)
        output = np.zeros([n_samples, self.output_shape])
        for step in range(n_step):
            start = step * batch_size
            end = (step + 1) * batch_size
            feed_dict = {self.x1: x[start:end]}
            output[start:end] = sess.run(self.embed1, feed_dict=feed_dict)
        return output

    def predict_model(self, sess, x, y=None, batch_size=128):
        """
        Args:
            sess:
            x: (list of two numpy array) a list of two numpy arrays (x1, x2)
            y: (numpy array of 0 or 1)
        """
        assert x[0].shape == x[1].shape
        assert x[0].shape[0] == y.shape[0]

        K.set_learning_phase(0)
        n_samples = len(x[0])
        output = np.zeros([n_samples, ])
        loss = 0
        n_step = np.ceil(n_samples / float(batch_size)).astype(np.int32)

        for step in range(n_step):
            start = step * batch_size
            end = (step + 1) * batch_size
            if y is None:
                feed_dict = {self.x1: x[0][start:end], self.x2: x[1][start:end]}
                output[start:end] = sess.run(self.dist, feed_dict=feed_dict)
            else:
                feed_dict = {self.x1: x[0][start:end], self.x2: x[1][start:end], 
                             self.y: y[start:end]}
                output[start:end], l = sess.run([self.dist, self.loss], 
                                                feed_dict=feed_dict)
                loss += l * len(x[0][start:end])

        if y is None:
            return output 
        else:
            return output, loss / n_samples

    def eval_model(self, sess, data, batch_size=128):

        x, y = data

        assert x.shape[0] == y.shape[0]

        ind = np.arange(len(x))
        np.random.shuffle(ind)
        x1 = x
        x2 = x[ind]
        y_eval = y == y[ind]
        dist, loss = self.predict_model(sess, [x1, x2], y_eval, batch_size=batch_size)
        dist_same = np.mean(dist[np.where(y_eval == 1)[0]])
        dist_diff = np.mean(dist[np.where(y_eval == 0)[0]])
        return dist, dist_same, dist_diff, loss