# -*- coding: utf-8 -*-

import tensorflow as tf
from vibration_generator import generator
import numpy as np
import random
import pickle

batch_size = 120
training_epochs = 100000
num_of_kinds = 5
accMatrix = np.zeros((training_epochs))
testMatrix = np.zeros((training_epochs))

# 5種の機器の振動データ作成クラス
generator = generator(num_of_kinds)
dataset = []
row_dataset = []
# 相関行列は適当、センサー5種類
matrix = np.random.rand(5, 5)
'''
for i in range(num_of_kinds + 1):
    if i == num_of_kinds:
        dataset.append(np.dot(matrix, generator.generate_normal()))
        row_dataset.append(np.dot(matrix, generator.generate_normal()))
    else:
        dataset.append(np.dot(matrix, generator.generate_abnormal(i)))
        row_dataset.append(np.dot(matrix, generator.generate_abnormal(i)))
'''
dataset = pickle.load(open('LSTMdict.pickle', 'rb'))['dataset']
row_dataset = pickle.load(open('LSTMdict.pickle', 'rb'))['row_dataset']
matrix = pickle.load(open('LSTMdict.pickle', 'rb'))['matrix']
sess = tf.Session()
x = tf.placeholder("float", shape=[None, 500])
y_ = tf.placeholder("float", shape=[None, num_of_kinds + 1])


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
    
# バッチ作成
def make_batch(batch_size):
    each_size = int(batch_size/6)
    batch = np.zeros((batch_size, 500))
    batch = np.array(batch, dtype=np.float32)
    output = np.zeros((batch_size, 6))
    output = np.array(output, dtype=np.int32)
    
    for i in range(each_size):
        index = random.randint(0, 9000)
        batch[i, :] = np.reshape(dataset[0][:, index:index + 100], (1, 500))
        output[i, 0] = 1
        
    for i in range(each_size, each_size*2):
        index = random.randint(0, 9000)
        batch[i, :] = np.reshape(dataset[1][:, index:index + 100], (1, 500))
        output[i, 1] = 1
        
    for i in range(each_size*2, each_size*3):
        index = random.randint(0, 9000)
        batch[i, :] = np.reshape(dataset[2][:, index:index + 100], (1, 500))
        output[i, 2] = 1
        
    for i in range(each_size*3, each_size*4):
        index = random.randint(0, 9000)
        batch[i, :] = np.reshape(dataset[3][:, index:index + 100], (1, 500))
        output[i, 3] = 1
        
    for i in range(each_size*4, each_size*5):
        index = random.randint(0, 9000)
        batch[i, :] = np.reshape(dataset[4][:, index:index + 100], (1, 500))
        output[i, 4] = 1
        
    for i in range(each_size*5, each_size*6):
        index = random.randint(0, 9000)
        batch[i, :] = np.reshape(dataset[5][:, index:index + 100], (1, 500))
        output[i, 5] = 1
    
    return batch, output

# testバッチ作成
def make_batch_test(batch_size):
    each_size = int(batch_size/6)
    batch = np.zeros((batch_size, 500))
    batch = np.array(batch, dtype=np.float32)
    output = np.zeros((batch_size, 6))
    output = np.array(output, dtype=np.int32)
    
    for i in range(each_size):
        index = random.randint(9000, 9900)
        batch[i, :] = np.reshape(dataset[0][:, index:index + 100], (1, 500))
        output[i, 0] = 1
        
    for i in range(each_size, each_size*2):
        index = random.randint(9000, 9900)
        batch[i, :] = np.reshape(dataset[1][:, index:index + 100], (1, 500))
        output[i, 1] = 1
        
    for i in range(each_size*2, each_size*3):
        index = random.randint(9000, 9900)
        batch[i, :] = np.reshape(dataset[2][:, index:index + 100], (1, 500))
        output[i, 2] = 1
        
    for i in range(each_size*3, each_size*4):
        index = random.randint(9000, 9900)
        batch[i, :] = np.reshape(dataset[3][:, index:index + 100], (1, 500))
        output[i, 3] = 1
        
    for i in range(each_size*4, each_size*5):
        index = random.randint(9000, 9900)
        batch[i, :] = np.reshape(dataset[4][:, index:index + 100], (1, 500))
        output[i, 4] = 1
        
    for i in range(each_size*5, each_size*6):
        index = random.randint(9000, 9900)
        batch[i, :] = np.reshape(dataset[5][:, index:index + 100], (1, 500))
        output[i, 5] = 1
    
    return batch, output

W_conv1 = weight_variable([5, 5, 1, 16])
b_conv1 = bias_variable([16])
x_image = tf.reshape(x, [-1, 5, 100, 1])
h_conv1 = tf.nn.relu(conv2d_pad(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_1_5(h_conv1)

W_conv2 = weight_variable([5, 5, 16, 32])
b_conv2 = bias_variable([32])
h_conv2 = tf.nn.relu(conv2d_pad(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_1_4(h_conv2)

W_conv3 = weight_variable([2, 2, 32, 64])
b_conv3 = bias_variable([64])
h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
h_pool3 = max_pool_2_2(h_conv3)

W_conv4 = weight_variable([2, 2, 64, 128])
b_conv4 = bias_variable([128])
h_conv4 = tf.nn.relu(conv2d(h_pool3, W_conv4) + b_conv4)

W_fc1 = weight_variable([128, 32])
b_fc1 = bias_variable([32])
h_flat = tf.reshape(h_conv4, [-1, 128])
h_fc1 = tf.nn.relu(tf.matmul(h_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([32, 6])
b_fc2 = bias_variable([6])
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
"""
saver = tf.train.Saver()
saver.save(sess, "./CNNmodel.ckpt")
CNNdict = {"accMatrix":accMatrix, "testMatrix":testMatrix, "dataset":dataset, "row_dataset":row_dataset, "matrix":matrix}
pickle.dump(CNNdict, open('CNNdict.pickle', mode='wb'))
"""
