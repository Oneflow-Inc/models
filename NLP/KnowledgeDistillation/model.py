import tensorflow as tf
import oneflow as flow
import os


class TeacherModel:
    def __init__(self, args, model_type, X, Y):
        self.X = X
        self.Y = Y
        self.learning_rate = 0.001
        self.num_steps = args.num_steps
        self.batch_size = args.batch_size
        self.display_step = args.display_step
        self.num_input = 784  # MNIST data input (img shape: 28*28)
        self.num_classes = 10
        self.dropoutprob = args.dropoutprob
        self.softmax_temperature = args.temperature
        self.model_type = model_type
        self.initializer = flow.random_normal_initializer(stddev=0.1)

        # Store layers weight & bias
        self.weights = {
            'wd1': flow.get_variable(
                shape=[2 * 7 * 64, 1024],
                dtype=flow.float,
                initializer=flow.random_normal_initializer(mean=0.0, stddev=0.02, seed=None, dtype=None),
                name="%s_%s" % (self.model_type, "wd1")
            ),
            'wout': flow.get_variable(
                shape=[1024, self.num_classes],
                dtype=flow.float,
                initializer=flow.random_normal_initializer(mean=0.0, stddev=0.02, seed=None, dtype=None),
                name="%s_%s" % (self.model_type, "wout")
            )
        }

        self.biases = {
            'bc1': flow.get_variable(
                shape=[32],
                dtype=flow.float,
                initializer=flow.random_normal_initializer(mean=0.0, stddev=0.02, seed=None, dtype=None),
                name="%s_%s" % (self.model_type, "bc1")
            ),
            'bc2': flow.get_variable(
                shape=[64],
                dtype=flow.float,
                initializer=flow.random_normal_initializer(mean=0.0, stddev=0.02, seed=None, dtype=None),
                name="%s_%s" % (self.model_type, "bc2")
            ),
            'bd1': flow.get_variable(
                shape=[1024],
                dtype=flow.float,
                initializer=flow.random_normal_initializer(mean=0.0, stddev=0.02, seed=None, dtype=None),
                name="%s_%s" % (self.model_type, "bd1")
            ),
            'bout': flow.get_variable(
                shape=[self.num_classes],
                dtype=flow.float,
                initializer=flow.random_normal_initializer(mean=0.0, stddev=0.02, seed=None, dtype=None),
                name="%s_%s" % (self.model_type, "bout")
            )
        }

        return self.build_model()

    def conv2d(self, x, filters: int, b, kernel_size: int, strides=1, name=''):
        # Conv2D wrapper, with bias and relu activation
        with flow.scope.namespace("%sconv2d" % (self.model_type)):
            x = flow.layers.conv2d(
                x,
                filters=filters,
                kernel_size=kernel_size,
                strides=[strides, strides],
                data_format="NCHW",
                kernel_initializer=self.initializer,
                padding='SAME',
                name="{}-{}".format(self.model_type, name)
            )
            x = flow.nn.bias_add(x, b)
            return flow.nn.relu(x)

    def maxpool2d(self, x, k=2):
        # MaxPool2D wrapper
        with flow.scope.namespace("%smaxpool2d" % (self.model_type)):
            return flow.nn.max_pool2d(x, ksize=[k, k], strides=[k, k],
                                  padding='SAME')

    # Create model
    def build_model(self):
        # MNIST data input is a 1-D vector of 784 features (28*28 pixels)
        # Reshape to match picture format [Height x Width x Channel]
        # Tensor input become 4-D: [Batch Size, Height, Width, Channel]
        with flow.scope.namespace("%sinputreshape" % (self.model_type)):
            x = flow.reshape(self.X, shape=[-1, 28, 28, 1])

        # Convolution Layer
        with flow.scope.namespace("%sconvmaxpool" % (self.model_type)):
            # print('x.shape=', x.shape)
            conv1 = self.conv2d(x, 32, self.biases['bc1'], 5, name='conv1')
            # Max Pooling (down-sampling)
            conv1 = self.maxpool2d(conv1, k=2)

            # Convolution Layer
            conv2 = self.conv2d(x, 64, self.biases['bc2'], 5, name='conv2')
            # Max Pooling (down-sampling)
            conv2 = self.maxpool2d(conv2, k=2)

        # Fully connected layer
        # Reshape conv2 output to fit fully connected layer input
        with flow.scope.namespace("%sfclayer" % (self.model_type)):
            # print("conv2.shape=", conv2.shape) # [100, 64, 14, 1]
            fc1 = flow.reshape(conv2, [-1, self.weights['wd1'].shape[0]]) # 2 * 7 * 64

            fc1 = flow.nn.bias_add(flow.matmul(fc1, self.weights['wd1']), self.biases['bd1'])
            fc1 = flow.nn.relu(fc1)
            # Apply Dropout
            fc1 = flow.nn.dropout(fc1, self.dropoutprob)
            self.logits = flow.nn.bias_add(flow.matmul(fc1, self.weights['wout']), self.biases['bout']) / self.softmax_temperature

        with flow.scope.namespace("%sprediction" % (self.model_type)):
            self.prediction = flow.nn.softmax(self.logits)

        with flow.scope.namespace("%soptimization" % (self.model_type)):
            self.loss = flow.math.reduce_mean(flow.nn.sparse_softmax_cross_entropy_with_logits(
                logits=self.logits, labels=self.Y))

    def get_res(self):
        return self.loss, self.logits, self.prediction



class StudentModel:
    def __init__(self, args, model_type, X, Y, soft_Y = None, flag: bool = False):
        self.X = X
        self.Y = Y
        self.soft_Y = soft_Y
        self.flag = flag
        self.learning_rate = 0.001
        self.num_steps = args.num_steps
        self.batch_size = args.batch_size
        self.display_step = args.display_step
        self.n_hidden_1 = 256  # 1st layer number of neurons
        self.n_hidden_2 = 256  # 2nd layer number of neurons
        self.num_input = 784  # MNIST data input (img shape: 28*28)
        self.num_classes = 10
        self.softmax_temperature = args.temperature
        self.model_type = model_type
        self.initializer = flow.random_normal_initializer(stddev=0.1)

        # Store layers weight & bias
        self.weights = {
            'h1': flow.get_variable(
                shape=[self.num_input, self.n_hidden_1],
                dtype=flow.float,
                initializer=flow.random_normal_initializer(mean=0.0, stddev=0.02, seed=None, dtype=None),
                name="%s_%s" % (self.model_type, "h1")
            ),
            'h2': flow.get_variable(
                shape=[self.n_hidden_1, self.n_hidden_2],
                dtype=flow.float,
                initializer=flow.random_normal_initializer(mean=0.0, stddev=0.02, seed=None, dtype=None),
                name="%s_%s" % (self.model_type, "h2")
            ),
            'wout': flow.get_variable(
                shape=[self.n_hidden_2, self.num_classes],
                dtype=flow.float,
                initializer=flow.random_normal_initializer(mean=0.0, stddev=0.02, seed=None, dtype=None),
                name="%s_%s" % (self.model_type, "wout")
            ),
            'wlinear': flow.get_variable(
                shape=[self.num_input, self.num_classes],
                dtype=flow.float,
                initializer=flow.random_normal_initializer(mean=0.0, stddev=0.02, seed=None, dtype=None),
                name="%s_%s" % (self.model_type, "wlinear")
            )
        }

        self.biases = {
            'b1': flow.get_variable(
                shape=[self.n_hidden_1],
                dtype=flow.float,
                initializer=flow.random_normal_initializer(mean=0.0, stddev=0.02, seed=None, dtype=None),
                name="%s_%s" % (self.model_type, "b1")
            ),
            'b2': flow.get_variable(
                shape=[self.n_hidden_2],
                dtype=flow.float,
                initializer=flow.random_normal_initializer(mean=0.0, stddev=0.02, seed=None, dtype=None),
                name="%s_%s" % (self.model_type, "b2")
            ),
            'bout': flow.get_variable(
                shape=[self.num_classes],
                dtype=flow.float,
                initializer=flow.random_normal_initializer(mean=0.0, stddev=0.02, seed=None, dtype=None),
                name="%s_%s" % (self.model_type, "bout")
            ),
            'blinear': flow.get_variable(
                shape=[self.num_classes],
                dtype=flow.float,
                initializer=flow.random_normal_initializer(mean=0.0, stddev=0.02, seed=None, dtype=None),
                name="%s_%s" % (self.model_type, "blinear")
            )
        }

        self.build_model()

    # Create model
    def build_model(self):
        with flow.scope.namespace("%sfclayer" % (self.model_type)):
            # Hidden fully connected layer with 256 neurons
            x = flow.reshape(self.X, shape=[-1, self.X.shape[-1] * self.X.shape[-2]])
            layer_1 = flow.nn.bias_add(flow.matmul(x, self.weights['h1']), self.biases['b1'])
            # # Hidden fully connected layer with 256 neurons
            layer_2 = flow.nn.bias_add(flow.matmul(layer_1, self.weights['h2']), self.biases['b2'])
            # # Output fully connected layer with a neuron for each class
            self.logits = flow.nn.bias_add(flow.matmul(layer_2, self.weights['wout']), self.biases['bout'])

        with flow.scope.namespace("%sprediction" % (self.model_type)):
            self.prediction = flow.nn.softmax(self.logits)

        with flow.scope.namespace("%soptimization" % (self.model_type)):
            # Define loss and optimizer
            self.loss_standard = flow.math.reduce_mean(flow.nn.sparse_softmax_cross_entropy_with_logits(
                logits=self.logits, labels=self.Y))

            self.total_loss = self.loss_standard

            self.loss_soft = 0.0

            if self.flag:
                self.loss_soft = flow.math.reduce_mean(flow.nn.softmax_cross_entropy_with_logits(
                                            logits=self.logits / self.softmax_temperature, labels=self.soft_Y))

            self.total_loss += self.softmax_temperature * self.softmax_temperature * self.loss_soft

    def get_res(self):
        return self.total_loss, self.logits, self.prediction