import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets.cifar100 import load_data

tf.set_random_seed(777)

learning_rate = 0.001
training_epochs = 200
batch_size = 128

class MyCIFAR10Model:
    def __init__(self, sess, name):
        self.sess = sess
        self.name = name
        self._build_net()

    def _build_net(self):
        with tf.variable_scope(self.name):
            self.training = tf.placeholder(tf.bool)
            self.X = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
            x_img = tf.reshape(self.X, [-1, 32, 32, 3])

            self.Y = tf.placeholder(tf.float32, shape=[None, 100])

            initializer = tf.contrib.layers.xavier_initializer()

            self.conv1 = tf.layers.conv2d(inputs=x_img, filters=64, kernel_size=[3, 3], padding="SAME",
                                     activation=tf.nn.relu, bias_initializer=initializer, kernel_initializer=initializer)
            self.pool1 = tf.layers.max_pooling2d(inputs=self.conv1, pool_size=[2, 2], padding="SAME", strides=2)
            self.dropout1 = tf.layers.dropout(inputs=self.pool1, rate=0.3, training=self.training)

            self.conv2 = tf.layers.conv2d(inputs=self.dropout1, filters=128, kernel_size=[3, 3], padding="SAME",
                                     activation=tf.nn.relu, bias_initializer=initializer, kernel_initializer=initializer)
            self.pool2 = tf.layers.max_pooling2d(inputs=self.conv2, pool_size=[2, 2], padding="SAME", strides=2)
            self.dropout2 = tf.layers.dropout(inputs=self.pool2, rate=0.3, training=self.training)

            # Convolutional Layer #2 and Pooling Layer #2
            self.conv3 = tf.layers.conv2d(inputs=self.dropout2, filters=256, kernel_size=[3, 3], padding="SAME",
                                     activation=tf.nn.relu, bias_initializer=initializer, kernel_initializer=initializer)
            self.pool3 = tf.layers.max_pooling2d(inputs=self.conv3, pool_size=[2, 2], padding="SAME", strides=2)
            self.dropout3 = tf.layers.dropout(inputs=self.pool3, rate=0.3, training=self.training)

            self.conv4 = tf.layers.conv2d(inputs=self.dropout3, filters=512, kernel_size=[3, 3], padding="SAME",
                                     activation=tf.nn.relu, bias_initializer=initializer, kernel_initializer=initializer)
            self.pool4 = tf.layers.max_pooling2d(inputs=self.conv4, pool_size=[2, 2], padding="SAME", strides=2)
            #self.dropout4 = tf.layers.dropout(inputs=self.pool4, rate=0.5, training=self.training)

            # Dense Layer with Relu
            self.flat = tf.contrib.layers.flatten(self.pool4)
            self.dense5 = tf.layers.dense(inputs=self.flat, units=1024, activation=tf.nn.relu,
                                     bias_initializer=initializer, kernel_initializer=initializer)
            self.dropout5 = tf.layers.dropout(inputs=self.dense5, rate=0.5, training=self.training)


            # Logits (no activation) Layer: L5 Final FC 625 inputs -> 10 outputs
            self.logits = tf.layers.dense(inputs=self.dropout7, units=100)

        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.Y))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost)

        correct_prediction = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.Y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def predict(self, x_test, training=False):
        return self.sess.run(self.logits, feed_dict={self.X: x_test, self.training: training})

    def get_accuracy(self, x_test, y_test, batch_size=512, training=False):
        #return self.sess.run(self.accuracy, feed_dict={self.X: x_test, self.Y: y_test, self.training: training})
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

            correct_sample += sess.run(self.accuracy, feed_dict=feed) * N_batch

        return correct_sample / N

    def train(self, x_data, y_data, training=True):
        feed = {
            self.X: x_data,
            self.Y: y_data,
            self.training: training
        }
        c, _ = self.sess.run([self.cost, self.optimizer], feed_dict=feed)
        acc = self.sess.run(self.accuracy, feed_dict=feed)

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

y_train_one_hot = tf.squeeze(tf.one_hot(y_train, 100), axis=1)
y_test_one_hot = tf.squeeze(tf.one_hot(y_test, 100), axis=1)

sess = tf.Session()
model = MyCIFAR10Model(sess, "sex")

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