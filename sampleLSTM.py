# -*- coding: utf-8 -*-

from __future__ import print_function

from vibration_generator import generator
import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
import random
import pickle

# Training Parameters
learning_rate = 0.001
training_steps = 100000
batch_size = 120
display_step = 200
num_of_kinds = 5
accMatrix = np.zeros((training_steps))
testMatrix = np.zeros((training_steps))
lossMatrix = np.zeros((training_steps))

# 5種の機器の振動データ作成クラス
generator = generator(num_of_kinds)
dataset = []
row_dataset = []
# 相関行列は適当、センサー5種類
matrix = np.random.rand(5, 5)
"""
for i in range(num_of_kinds + 1):
    if i == num_of_kinds:
        dataset.append(np.dot(matrix, generator.generate_normal()))
        row_dataset.append(np.dot(matrix, generator.generate_normal()))
    else:
        dataset.append(np.dot(matrix, generator.generate_abnormal(i)))
        row_dataset.append(np.dot(matrix, generator.generate_abnormal(i)))
"""
dataset = pickle.load(open('LSTMdict.pickle', 'rb'))['dataset']
row_dataset = pickle.load(open('LSTMdict.pickle', 'rb'))['row_dataset']
matrix = pickle.load(open('LSTMdict.pickle', 'rb'))['matrix']
# Network Parameters
num_input = 5 # MNIST data input (img shape: 28*28)
timesteps = 100 # timesteps
num_hidden = 32 # hidden layer num of features
num_classes = 6 # MNIST total classes (0-9 digits)

# バッチ作成
def make_batch(batch_size):
    each_size = int(batch_size/6)
    batch = np.zeros((batch_size, timesteps, num_input))
    batch = np.array(batch, dtype=np.float32)
    output = np.zeros((batch_size, 6))
    output = np.array(output, dtype=np.int32)
    
    for i in range(each_size):
        index = random.randint(0, 9000)
        batch[i, :] = dataset[0][:, index:index + 100].T
        output[i, 0] = 1
        
    for i in range(each_size, each_size*2):
        index = random.randint(0, 9000)
        batch[i, :] = dataset[1][:, index:index + 100].T
        output[i, 1] = 1
        
    for i in range(each_size*2, each_size*3):
        index = random.randint(0, 9000)
        batch[i, :] = dataset[2][:, index:index + 100].T
        output[i, 2] = 1
        
    for i in range(each_size*3, each_size*4):
        index = random.randint(0, 9000)
        batch[i, :] = dataset[3][:, index:index + 100].T
        output[i, 3] = 1
        
    for i in range(each_size*4, each_size*5):
        index = random.randint(0, 9000)
        batch[i, :] = dataset[4][:, index:index + 100].T
        output[i, 4] = 1
        
    for i in range(each_size*5, each_size*6):
        index = random.randint(0, 9000)
        batch[i, :] = dataset[5][:, index:index + 100].T
        output[i, 5] = 1
    
    return batch, output

# バッチ作成
def make_batch_test(batch_size):
    each_size = int(batch_size/6)
    batch = np.zeros((batch_size, timesteps, num_input))
    batch = np.array(batch, dtype=np.float32)
    output = np.zeros((batch_size, 6))
    output = np.array(output, dtype=np.int32)
    
    for i in range(each_size):
        index = random.randint(9000, 9900)
        batch[i, :] = dataset[0][:, index:index + 100].T
        output[i, 0] = 1
        
    for i in range(each_size, each_size*2):
        index = random.randint(9000, 9900)
        batch[i, :] = dataset[1][:, index:index + 100].T
        output[i, 1] = 1
        
    for i in range(each_size*2, each_size*3):
        index = random.randint(9000, 9900)
        batch[i, :] = dataset[2][:, index:index + 100].T
        output[i, 2] = 1
        
    for i in range(each_size*3, each_size*4):
        index = random.randint(9000, 9900)
        batch[i, :] = dataset[3][:, index:index + 100].T
        output[i, 3] = 1
        
    for i in range(each_size*4, each_size*5):
        index = random.randint(9000, 9900)
        batch[i, :] = dataset[4][:, index:index + 100].T
        output[i, 4] = 1
        
    for i in range(each_size*5, each_size*6):
        index = random.randint(9000, 9900)
        batch[i, :] = dataset[5][:, index:index + 100].T
        output[i, 5] = 1
    
    return batch, output

# tf Graph input
X = tf.placeholder("float", [None, timesteps, num_input])
Y = tf.placeholder("float", [None, num_classes])

# Define weights
weights = {
    'out': tf.Variable(tf.random_normal([num_hidden, num_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([num_classes]))
}


def RNN(x, weights, biases):

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, timesteps, n_input)
    # Required shape: 'timesteps' tensors list of shape (batch_size, n_input)

    # Unstack to get a list of 'timesteps' tensors of shape (batch_size, n_input)
    x = tf.unstack(x, timesteps, 1)

    # Define a lstm cell with tensorflow
    lstm_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)

    # Get lstm cell output
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

    # Linear activation, using rnn inner loop last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']

logits = RNN(X, weights, biases)
prediction = tf.nn.softmax(logits)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(1e-4)
train_op = optimizer.minimize(loss_op)

# Evaluate model (with test logits, for dropout to be disabled)
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)


    for step in range(1, training_steps+1):
        batch_x, batch_y = make_batch(batch_size)
        # Reshape data to get 28 seq of 28 elements
        # Run optimization op (backprop)
        prediction = sess.run([prediction], feed_dict={X: batch_x, Y: batch_y})
        sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
        # Calculate batch loss and accuracy
        loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
                                                                 Y: batch_y})
        lossMatrix[step-1] = loss
        accMatrix[step-1] = acc
        batch_x, batch_y = make_batch_test(batch_size)
        prediction_test = sess.run([prediction], feed_dict={X: batch_x, Y: batch_y})
        loss_test, acc_test = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
                                                                 Y: batch_y})
        testMatrix[step-1] = acc_test
        if step % display_step == 0 or step == 1:
            
            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss) + ", Training Accuracy= " + \
                  "{:.3f}".format(acc) + ", Testing Accuracy= " + \
                  "{:.3f}".format(acc_test))

    print("Optimization Finished!")
    
    saver = tf.train.Saver()
    saver.save(sess, "./LSTMmodel.ckpt")
LSTMdict = {"accMatrix":accMatrix, "testMatrix":testMatrix, "dataset":dataset, "row_dataset":row_dataset, "matrix":matrix, "prediction":prediction, "prediction_test":prediction_test}
pickle.dump(LSTMdict, open('LSTMdict.pickle', mode='wb'))

