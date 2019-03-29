# -*- coding: utf-8 -*-

import tensorflow as tf
from vibration_generator_abnormal_5level import generator
import numpy as np
import random
import pickle

batch_size = 100
training_epochs = 50000
num_of_kinds = 5
accMatrix = np.zeros((training_epochs))
testMatrix = np.zeros((training_epochs))

# 5種の機器の振動データ作成クラス
generator = generator(num_of_kinds)
dataset = []
row_dataset = []
# 相関行列は適当、センサー5種類
matrix = np.array([[0.9, 0.4, 0.2, 0.1, 0.05], 
                   [0.05, 0.9, 0.4, 0.2, 0.1], 
                   [0.1, 0.05, 0.9, 0.4, 0.2], 
                   [0.2, 0.1, 0.05, 0.9, 0.4], 
                   [0.4, 0.2, 0.1, 0.05, 0.9]])
dataset.append(np.dot(matrix, generator.generate_normal()))
row_dataset.append(generator.generate_normal())

dataset.append(np.dot(matrix, generator.generate_abnormal(0)))
row_dataset.append(generator.generate_abnormal(0))

# eq1 に異常があるかないか
sess = tf.Session()
x = tf.placeholder("float", shape=[None, 100])
y_ = tf.placeholder("float", shape=[None, 2])

# 荷重作成
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

# バイアス作成
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d_pad(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')

def max_pool_2_2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')
def max_pool_1_4(x):
    return tf.nn.max_pool(x, ksize=[1, 1, 4, 1],
                          strides=[1, 1, 4, 1], padding='SAME')

def max_pool_1_5(x):
    return tf.nn.max_pool(x, ksize=[1, 1, 5, 1],
                          strides=[1, 1, 5, 1], padding='SAME')

def max_pool_1_2(x):
    return tf.nn.max_pool(x, ksize=[1, 1, 2, 1],
                          strides=[1, 1, 2, 1], padding='SAME')
    
# バッチ作成
def make_batch(batch_size):
    batch = np.zeros((batch_size, 100))
    batch = np.array(batch, dtype=np.float32)
    output = np.zeros((batch_size, 2))
    output = np.array(output, dtype=np.int32)
    
    for i in range(50):
        index = random.randint(0, 45000)
        batch[i, :] = np.reshape(dataset[0][0, index:index + 100], (1, 100))
        output[i, 0] = 1
        
    for i in range(50, 60):
        index = random.randint(0, 8900)
        batch[i, :] = np.reshape(dataset[1][0, index:index + 100], (1, 100))
        output[i, 1] = 1
    for i in range(60, 70):
        index = random.randint(10000, 18900)
        batch[i, :] = np.reshape(dataset[1][0, index:index + 100], (1, 100))
        output[i, 1] = 1
    for i in range(70, 80):
        index = random.randint(20000, 28900)
        batch[i, :] = np.reshape(dataset[1][0, index:index + 100], (1, 100))
        output[i, 1] = 1
    for i in range(80, 90):
        index = random.randint(30000, 38900)
        batch[i, :] = np.reshape(dataset[1][0, index:index + 100], (1, 100))
        output[i, 1] = 1
    for i in range(90, 100):
        index = random.randint(40000, 48900)
        batch[i, :] = np.reshape(dataset[1][0, index:index + 100], (1, 100))
        output[i, 1] = 1
    return batch, output

# testバッチ作成
def make_batch_test(batch_size):
    batch = np.zeros((batch_size, 100))
    batch = np.array(batch, dtype=np.float32)
    output = np.zeros((batch_size, 2))
    output = np.array(output, dtype=np.int32)
    
    for i in range(50):
        index = random.randint(45000, 49900)
        batch[i, :] = np.reshape(dataset[0][0, index:index + 100], (1, 100))
        output[i, 0] = 1
        
    for i in range(50, 60):
        index = random.randint(9000, 9900)
        batch[i, :] = np.reshape(dataset[1][0, index:index + 100], (1, 100))
        output[i, 1] = 1
    for i in range(60, 70):
        index = random.randint(19000, 19900)
        batch[i, :] = np.reshape(dataset[1][0, index:index + 100], (1, 100))
        output[i, 1] = 1
    for i in range(70, 80):
        index = random.randint(29000, 29900)
        batch[i, :] = np.reshape(dataset[1][0, index:index + 100], (1, 100))
        output[i, 1] = 1
    for i in range(80, 90):
        index = random.randint(39000, 39900)
        batch[i, :] = np.reshape(dataset[1][0, index:index + 100], (1, 100))
        output[i, 1] = 1
    for i in range(90, 100):
        index = random.randint(49000, 49900)
        batch[i, :] = np.reshape(dataset[1][0, index:index + 100], (1, 100))
        output[i, 1] = 1
    return batch, output

# testバッチ作成
def make_batch_final(batch_size):
    batch = np.zeros((1000, 100))
    batch = np.array(batch, dtype=np.float32)
    output = np.zeros((1000, 2))
    output = np.array(output, dtype=np.int32)
    
    for i in range(500):
        index = random.randint(45000, 49900)
        batch[i, :] = np.reshape(dataset[0][0, index:index + 100], (1, 100))
        output[i, 0] = 1
        
    for i in range(500, 600):
        index = random.randint(9000, 9900)
        batch[i, :] = np.reshape(dataset[1][0, index:index + 100], (1, 100))
        output[i, 1] = 1
    for i in range(600, 700):
        index = random.randint(19000, 19900)
        batch[i, :] = np.reshape(dataset[1][0, index:index + 100], (1, 100))
        output[i, 1] = 1
    for i in range(700, 800):
        index = random.randint(29000, 29900)
        batch[i, :] = np.reshape(dataset[1][0, index:index + 100], (1, 100))
        output[i, 1] = 1
    for i in range(800, 900):
        index = random.randint(39000, 39900)
        batch[i, :] = np.reshape(dataset[1][0, index:index + 100], (1, 100))
        output[i, 1] = 1
    for i in range(900, 1000):
        index = random.randint(49000, 49900)
        batch[i, :] = np.reshape(dataset[1][0, index:index + 100], (1, 100))
        output[i, 1] = 1
    return batch, output

W_conv1 = weight_variable([1, 5, 1, 16])
b_conv1 = bias_variable([16])
x_image = tf.reshape(x, [-1, 1, 100, 1])
h_conv1 = tf.nn.relu(conv2d_pad(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_1_5(h_conv1)

W_conv2 = weight_variable([1, 5, 16, 32])
b_conv2 = bias_variable([32])
h_conv2 = tf.nn.relu(conv2d_pad(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_1_4(h_conv2)

W_conv3 = weight_variable([1, 2, 32, 64])
b_conv3 = bias_variable([64])
h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
h_pool3 = max_pool_1_2(h_conv3)

W_conv4 = weight_variable([1, 2, 64, 128])
b_conv4 = bias_variable([128])
h_conv4 = tf.nn.relu(conv2d(h_pool3, W_conv4) + b_conv4)

W_fc1 = weight_variable([128, 32])
b_fc1 = bias_variable([32])
h_flat = tf.reshape(h_conv4, [-1, 128])
h_fc1 = tf.nn.relu(tf.matmul(h_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([32, 2])
b_fc2 = bias_variable([2])
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

cross_entropy = -tf.reduce_sum(y_ * tf.log(tf.clip_by_value(y_conv,1e-10,1.0)))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
sess.run(tf.initialize_all_variables())

for i in range(training_epochs):
    batch, output = make_batch(batch_size)
    train_accuracy = accuracy.eval(session=sess, feed_dict={x: batch, y_: output, keep_prob: 1.0})
    accMatrix[i] = train_accuracy
    
    train_step.run(session=sess, feed_dict={x: batch, y_: output, keep_prob: 0.5})

    batch, output = make_batch_test(batch_size)
    test_accuracy = accuracy.eval(session=sess, feed_dict={x: batch, y_: output, keep_prob: 1.0})
    testMatrix[i] = test_accuracy

    if i%100 ==0:
        loss = cross_entropy.eval(session=sess, feed_dict={x: batch, y_: output, keep_prob: 1.0})
        print(i)
        print(train_accuracy)
        print(test_accuracy)
        print(loss)
        
batch, output = make_batch_final(batch_size)
y_conv = y_conv.eval(session=sess, feed_dict={x: batch, y_: output, keep_prob: 1.0})
CNNdict = {"accMatrix":accMatrix, "testMatrix":testMatrix, "dataset":dataset, "row_dataset":row_dataset, "matrix":matrix, "y_conv":y_conv}
pickle.dump(CNNdict, open('CNNsingledict.pickle', mode='wb'))
