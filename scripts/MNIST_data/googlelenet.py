import tensorflow as tf
import tensorflow.contrib.layers as layers
import tensorflow.contrib.framework as ops
import numpy as np
from tensorflow.keras.datasets.cifar10 import load_data

tf.set_random_seed(777)

learning_rate = 0.001
training_epochs = 200
batch_size = 128

class google_model:
    def __init__(self, sess, name):
        self.sess = sess
        self.name = name
        self.googlenet()

    def get_inception_layer(self, inputs, conv11_size, conv33_11_size, conv33_size, conv55_11_size, conv55_size, pool11_size):
        with tf.variable_scope("conv_1x1"):
            conv11 = layers.conv2d(inputs, conv11_size, [1, 1])
        with tf.variable_scope("conv_3x3"):
            conv33_11 = layers.conv2d(inputs, conv33_11_size, [1, 1])
            conv33 = layers.conv2d( conv33_11, conv33_size, [3, 3])
        with tf.variable_scope("conv_5x5"):
            conv55_11 = layers.conv2d(inputs, conv55_11_size, [1, 1])
            conv55 = layers.conv2d( conv55_11, conv55_size, [5, 5])
        with tf.variable_scope("pool_proj"):
            pool_proj = layers.max_pool2d(inputs, [3, 3], stride=1)
            pool11 = layers.conv2d(pool_proj, pool11_size, [1, 1])
        if tf.__version__ == '0.11.0rc0':
            return tf.concat(3, [conv11, conv33, conv55, pool11])

        return tf.concat([conv11, conv33, conv55, pool11], 3)

    def aux_logit_layer(self, inputs, num_classes,is_training):
        with tf.variable_scope("pool2d"):
            pooled = layers.avg_pool2d(inputs, [ 5, 5 ], stride = 3 )
        with tf.variable_scope("conv11"):
            conv11 = layers.conv2d( pooled, 128, [1, 1] )
        with tf.variable_scope("flatten"):
            flat = tf.reshape( conv11, [-1, 2048] )
        with tf.variable_scope("fc"):
            fc = layers.fully_connected( flat, 1024, activation_fn=None )
        with tf.variable_scope("drop"):
            drop = layers.dropout( fc, 0.3, is_training = is_training )
        with tf.variable_scope( "linear" ):
            linear = layers.fully_connected( drop, num_classes, activation_fn=None )
        with tf.variable_scope("soft"):
            soft = tf.nn.softmax( linear )
        return soft

    def googlenet(self, dropout_keep_prob=0.4, num_classes=10, is_training=True, scope=''):
        '''
        Implementation of https://arxiv.org/pdf/1409.4842.pdf
        '''
        self.training = tf.placeholder(tf.bool)
        self.X = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
        self.Y = tf.placeholder(tf.float32, shape=[None, num_classes])
        x_img = tf.reshape(self.X, [-1, 32, 32, 3])
        print(x_img.shape)
        self.end_points = {}
        with tf.name_scope(scope, "googlenet", [x_img]):
            with ops.arg_scope( [ layers.max_pool2d ], padding = 'SAME' ):
                self.end_points['conv0'] = layers.conv2d(x_img, 64, [3, 3], padding="SAME", scope='conv0')
                print(self.end_points['conv0'])
                self.end_points['pool0'] = layers.max_pool2d(self.end_points['conv0'], [2, 2], stride=2, padding='SAME', scope='pool0')
                print(self.end_points['pool0'])
                self.end_points['conv1_a'] = layers.conv2d( self.end_points['pool0'], 64, [1, 1], scope='conv1_a')
                print(self.end_points['conv1_a'])
                self.end_points['conv1_b'] = layers.conv2d( self.end_points['conv1_a'], 128, [3, 3], scope='conv1_b')
                print(self.end_points['conv1_b'])
                self.end_points['pool1'] = layers.max_pool2d(self.end_points['conv1_b'], [2, 2], stride=2, padding="SAME", scope='pool1')
                print(self.end_points['pool1'])

                with tf.variable_scope("inception_3a"):
                    self.end_points['inception_3a'] = self.get_inception_layer(self.end_points['pool1'], 64, 96, 128, 16, 32, 32 )

                with tf.variable_scope("inception_3b"):
                    self.end_points['inception_3b'] = self.get_inception_layer( self.end_points['inception_3a'], 128, 128, 192, 32, 96, 64 )

                print(self.end_points['inception_3b'])

                self.end_points['pool2'] = layers.max_pool2d(self.end_points['inception_3b'], [ 3, 3 ], scope='pool2')

                print(self.end_points['pool2'])

                with tf.variable_scope("inception_4a"):
                    self.end_points['inception_4a'] = self.get_inception_layer( self.end_points['pool2'], 192, 96, 208, 16, 48, 64 )

                #with tf.variable_scope("aux_logits_1"):
                #    end_points['aux_logits_1'] = self.aux_logit_layer( end_points['inception_4a'], num_classes, is_training)

                with tf.variable_scope("inception_4b"):
                    self.end_points['inception_4b'] = self.get_inception_layer( self.end_points['inception_4a'], 160, 112, 224, 24, 64, 64 )

                with tf.variable_scope("inception_4c"):
                    self.end_points['inception_4c'] = self.get_inception_layer( self.end_points['inception_4b'], 128, 128, 256, 24, 64, 64 )

                with tf.variable_scope("inception_4d"):
                    self.end_points['inception_4d'] = self.get_inception_layer( self.end_points['inception_4c'], 112, 144, 288, 32, 64, 64 )

                #with tf.variable_scope("aux_logits_2"):
                #    end_points['aux_logits_2'] = self.aux_logit_layer( end_points['inception_4d'], num_classes, is_training)

                with tf.variable_scope("inception_4e"):
                    self.end_points['inception_4e'] = self.get_inception_layer( self.end_points['inception_4d'], 256, 160, 320, 32, 128, 128 )

                print(self.end_points['inception_4e'])

                self.end_points['pool3'] = layers.max_pool2d(self.end_points['inception_4e'], [3, 3], scope='pool3')

                print(self.end_points['pool3'])

                with tf.variable_scope("inception_5a"):
                    self.end_points['inception_5a'] = self.get_inception_layer( self.end_points['pool3'], 256, 160, 320, 32, 128, 128 )

                with tf.variable_scope("inception_5b"):
                    self.end_points['inception_5b'] = self.get_inception_layer( self.end_points['inception_5a'], 384, 192, 384, 48, 128, 128 )

                print(self.end_points['inception_5b'].shape)

#                self.end_points['pool4'] = layers.avg_pool2d(self.end_points['inception_5b'], [3, 3], scope='pool4')

#                print(self.end_points['pool4'])

                initializer = tf.contrib.layers.xavier_initializer()

                self.end_points['flat'] = tf.contrib.layers.flatten(self.end_points['inception_5b'])
                self.end_points['reshape']= tf.layers.dense(inputs=self.end_points['flat'], units=1024, activation=tf.nn.relu,
                                     bias_initializer=initializer, kernel_initializer=initializer)
                print(self.end_points['reshape'])
                self.end_points['dropout'] = layers.dropout( self.end_points['reshape'], dropout_keep_prob, is_training=self.training)


                self.end_points['logits'] = tf.layers.dense(inputs=self.end_points['dropout'], units=num_classes)
                print(self.end_points['logits'])

                self.end_points['cost'] = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.end_points['logits'], labels=self.Y))

                self.end_points['optimizer'] = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.end_points['cost'])

                correct_prediction = tf.equal(tf.argmax(self.end_points['logits'], 1), tf.argmax(self.Y, 1))
                self.end_points['accuracy'] = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def predict(self, x_test, training=False):
        return self.sess.run(self.end_points['logits'], feed_dict={self.X: x_test, self.training:training})

    def get_accuracy(self, x_test, y_test, batch_size=512, training=False):
        N = x_test.shape[0]
        correct_sample = 0

        for i in range(0, N, batch_size):
            X_batch = x_test[i: i + batch_size]
            y_batch = y_test[i: i + batch_size]
            N_batch = X_batch.shape[0]

            feed = {
                self.X: X_batch,
                self.Y: y_batch,
                self.training: training
            }

            correct_sample += sess.run(self.end_points['accuracy'], feed_dict=feed) * N_batch

        return correct_sample / N

    def train(self, x_data, y_data, training=True):
        feed = {
            self.X: x_data,
            self.Y: y_data,
            self.training: training
        }
        c, _ = self.sess.run([self.end_points['cost'], self.end_points['optimizer']], feed_dict=feed)
        acc = self.sess.run(self.end_points['accuracy'], feed_dict=feed)

        return c, acc;

def next_batch(num, data, labels):
  '''
  `num` 개수 만큼의 랜덤한 샘플들과 레이블들을 리턴합니다.
  '''
  idx = np.arange(0, len(data))
  np.random.shuffle(idx)
  idx = idx[:num]
  data_shuffle = [data[i] for i in idx]
  labels_shuffle = [labels[i] for i in idx]

  return np.asarray(data_shuffle), np.asarray(labels_shuffle)



(x_train, y_train), (x_test, y_test) = load_data()

y_train_one_hot = tf.squeeze(tf.one_hot(y_train, 10), axis=1)
y_test_one_hot = tf.squeeze(tf.one_hot(y_test, 10), axis=1)

sess = tf.Session()
model = google_model(sess, "sex")

sess.run(tf.global_variables_initializer())

print("start")

for epoch in range(training_epochs):
    avg_cost = 0
    collect_acc = 0.0
    total_batch = int(len(x_train)/batch_size)

    for i in range(total_batch):
        batch_xs, batch_ys = next_batch(batch_size, x_train, y_train_one_hot.eval(session=sess))
        c, acc = model.train(batch_xs, batch_ys)

        collect_acc += acc / total_batch
        avg_cost += c / total_batch

    #collect_acc = collect_acc/total_batch

    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))
    print('Train Accuracy:', collect_acc)
    print('Test Accuracy:', model.get_accuracy(x_test, y_test_one_hot.eval(session=sess)))
    print()

print('Learning Finished!')

# Test model and check accuracy
print('Accuracy:', model.get_accuracy(x_test, y_test_one_hot.eval(session=sess)))