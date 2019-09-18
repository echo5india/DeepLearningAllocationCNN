import tensorflow as tf
from utils import log
from config.config import DEFAULT_LEARNING_RATE, DEFAULT_TF_OPTIMIZER, DEFAULT_LOG_ENV


class Net:
    """
    a mother class for all types of Neural Networks
    """

    def __init__(self, shape_x: iter, shape_y: iter, hyperparams: dict, name: str):
        self._hyperparams = hyperparams
        self.name = name

        self._shape_x = shape_x
        self._shape_y = shape_y
        self._pre_thresholded_output = None

        self.x = None
        self.y = None
        self.phase_train = None
        self.dropout = None

        self.init = None
        self.output = None
        self.global_step = None
        self.optimizer = None
        self.loss = None
        self.accuracy = None

        self.summary_op = None

        self.saver = None

    def _inference(self, **kwargs):
        """
        instanciate: pre_thresholded_output, output
        """
        raise NotImplementedError('Should be implemented in the child class')

    @staticmethod
    def dense_layer(inputs, units, name=None):
        """
        A wrapped up dense layer function (defines dense layer operation)
        """
        with tf.variable_scope(name):
            inputs_shape = tf.Tensor.get_shape(inputs).as_list()[-1]

            w = tf.get_variable(name='weight',
                                trainable=True,
                                shape=[inputs_shape, units],
                                # [filter_height, filter_width, in_channels, out_channels]
                                initializer=tf.truncated_normal_initializer(stddev=0.01))
            b = tf.get_variable(name='bias',
                                trainable=True,
                                shape=[units],
                                initializer=tf.constant_initializer(0.0))

            outputs = tf.matmul(inputs, w)
            outputs = tf.nn.bias_add(outputs, b)
            return outputs

    @staticmethod
    def batch_normalization(inputs, training):
        """Performs a batch normalization using a standard set of parameters."""
        return tf.compat.v1.layers.batch_normalization(
            inputs=inputs, axis=-1,
            momentum=0.99, epsilon=1e-5, center=True,
            scale=True, training=training, fused=True)



    @staticmethod
    def conv2d(layer_name: str, inputs, kernel_shape: iter, strides=1, padding='SAME',
               activation_func=tf.nn.tanh, kernel_shape_pre=None):
        """
        Performs 2D convolution, enables a 'pre-convolution,
         the one that is used within the inception module for GoogLeNet
        """
        # in_channels = inputs.get_shape()[-1]

        with tf.variable_scope(layer_name):
            if kernel_shape_pre:
                with tf.variable_scope('pre_convolution'):
                    # mask weights and biases
                    w_pre = tf.get_variable(name='weights_pre',
                                            trainable=True,
                                            shape=kernel_shape_pre,
                                            # [filter_height, filter_width, in_channels, out_channels]
                                            initializer=tf.contrib.layers.xavier_initializer())
                    inputs = tf.nn.conv2d(inputs, filter=w_pre, strides=[1, strides, strides, 1], padding=padding,
                                          name='conv')
            with tf.variable_scope('convolution'):
                # mask weights and biases
                w = tf.get_variable(name='weights',
                                    trainable=True,
                                    shape=kernel_shape,
                                    # [filter_height, filter_width, in_channels, out_channels]
                                    initializer=tf.contrib.layers.xavier_initializer())
                b = tf.get_variable(name='biases',
                                    trainable=True,
                                    shape=[kernel_shape[-1]],
                                    initializer=tf.constant_initializer(0.0))
                inputs = tf.nn.conv2d(inputs, filter=w, strides=[1, strides, strides, 1], padding=padding, name='conv')
                inputs = tf.nn.bias_add(inputs, b, name='bias_add')
                outputs = activation_func(inputs,
                                          name='activation_func')
                return outputs

    @staticmethod
    def __cost_sensitive_loss_func(y_true, y_pred, expected_penalty):
        """
        Cost sensitive loss function with cross-entropy
        :param expected_penalty: penalty on confusion matrix
        :return: adjusted loss function
        """
        y_pred = tf.clip_by_value \
            ((y_pred * expected_penalty) / tf.reduce_sum(y_pred * expected_penalty, axis=-1, keepdims=True), 1e-7,
             1 - 1e-7)
        loss = tf.keras.backend.categorical_crossentropy(y_true, y_pred)
        return tf.reduce_mean(loss)

    def _xentropy_loss_func(self, cost_sensitive_loss=False, expected_penalty=None, **kwargs):
        """
        :param output: if cost_sensitive_loss is False output should be the pre_thresholded_output
        :param y_batch: y_batch
        :param expected_penalty: expected penalty of misclassifying i-th class.
        :return: loss function
        """
        y_batch = self.y
        with tf.name_scope('loss'):
            if cost_sensitive_loss:
                assert expected_penalty is not None, 'expected penalty in x_entropy_loss_function must not be None when cost_sensitive_loss = True'
                xentropy_loss = self.__cost_sensitive_loss_func(y_batch, self.output, expected_penalty)
            else:
                xentropy_loss = tf.reduce_mean(
                    tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_batch, logits=self._pre_thresholded_output))
            self.loss = xentropy_loss

    def _optimize(self):
        """
        Instantiates an tensorflow minimize operation
        """
        try:
            learning_rate = self._hyperparams['learning_rate']
        except KeyError:
            log('learning_rate was not specified in hyperparams using default:{}'.format(DEFAULT_LEARNING_RATE),
                loglevel='warning', environment=DEFAULT_LOG_ENV)
            learning_rate = DEFAULT_LEARNING_RATE

        try:
            tf_optimizer = self._hyperparams['tf_optimizer']
        except KeyError:
            log('tf_optimizer was not specisefied in hyperparams, using default:{}'.format(DEFAULT_TF_OPTIMIZER),
                loglevel='warning', environment=DEFAULT_LOG_ENV)
            tf_optimizer = DEFAULT_TF_OPTIMIZER
        if tf_optimizer.lower() == 'adam':
            tf_optimizer = tf.train.AdamOptimizer
        elif tf_optimizer.lower() == 'sgd':
            tf_optimizer = tf.train.GradientDescentOptimizer
        elif tf_optimizer.lower() == 'rmsprop':
            tf_optimizer = tf.train.RMSPropOptimizer
        else:
            raise NotImplementedError(
                'You need to modify the code to use another optimizer so far we have sgd, rmsprop and adam')

        tf.summary.scalar('training_loss', self.loss)
        self.optimizer = tf_optimizer(learning_rate=learning_rate).minimize(self.loss, global_step=self.global_step,
                                                                            name='minimize')

    def save(self, sess: tf.Session, model_ckpt_path: str, verbose=False, epoch=None, write_meta_graph=True):
        """
        Saves the model on disk
        :param sess: current session
        :param model_ckpt_path: path for the checkpoints' files on disk
        :param verbose: for full details
        :param epoch: number of the epoch for printing
        :param write_meta_graph: True to rewrite a .meta file for the network at each save
        """

        if verbose:
            msg = 'Saving Model to {}'.format(model_ckpt_path)
            msg += ', global step = {}'.format(self.global_step.eval(session=sess))
            msg = msg + ' epoch: {}'.format(epoch) if epoch else msg
            log(msg, environment=DEFAULT_LOG_ENV)
        self.saver.save(sess, model_ckpt_path, self.global_step, write_meta_graph=write_meta_graph)
        if verbose:
            log('Model Saved', environment=DEFAULT_LOG_ENV)

    def _accuracy_func(self):
        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(
                tf.cast(tf.equal(tf.argmax(self.output, axis=1), tf.argmax(self.y, axis=1)), tf.float32))
            tf.summary.scalar("validation_error", (1.0 - accuracy))
            self.accuracy = accuracy

    def restore_importants_ops(self, sess, model_ckpt_path_to_restore):
        raise NotImplementedError('Should be implemented in the child class since this is model dependent')


    def build_operations(self, **kwargs):
        """
        Builds all operations that will be run within the session and instantiates them

        :param kwargs: union (as a dict) of all arguments necessary for the functions below
        """
        self.x = tf.placeholder(tf.float32, self._shape_x, name='x')
        # tf.add_to_collection('x',self.x)
        self.y = tf.placeholder(tf.float32, self._shape_y, name='y')
        self.phase_train = tf.placeholder(tf.bool, name='phase_train')
        self.dropout = tf.placeholder(tf.float32, name='dropout')

        self._inference(**kwargs)  # example: dropout
        self._xentropy_loss_func(**kwargs)  # example: expected_penalty

        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self._optimize()  # todo see if we can do in a different way for the optimizer
        self._accuracy_func()
        self.summary_op = tf.summary.merge_all()
        self.init = tf.global_variables_initializer()

        max_save_to_keep = kwargs.get('max_save_to_keep', 4)
        self.saver = tf.train.Saver(max_to_keep=kwargs.get('max_save_to_keep', 4))
        log('Saver will keep the last {} latest models'.format(max_save_to_keep), DEFAULT_LOG_ENV)

    def get_dropout(self):
        try:
            dropout = self._hyperparams['dropout']
        except KeyError as e:
            print('dropout not found in kwargs, dropout 0.15 will be used')
            dropout = 0.15
        return dropout
