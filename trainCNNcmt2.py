# -*- coding: utf-8 -*-

import tensorflow as tf
from vibration_generator_abnormal_5level import generator
import numpy as np
import random
import pickle
import wave
from scipy import fromstring, int16
import math
import os
import shutil

"""
# logの保存先ディレクトリのpath
logs_path = './log/'
# もし該当のディレクトリが存在したら削除する
if os.path.exists(logs_path):
    shutil.rmtree(logs_path)
    
# 保存先ディレクトリを作成する
os.mkdir(logs_path)
"""
batch_size = 120
training_epochs = 1000000
num_of_kinds = 3
accMatrix = np.zeros((training_epochs))
testMatrix = np.zeros((training_epochs))

dataset = []

dummy = np.zeros((2, 2752512))
wr = wave.open('./data/test01_ch1.WAV', 'rb')
dummy[0, :] = fromstring(wr.readframes(wr.getnframes()), dtype = int16)
wr = wave.open('./data/test01_ch2.WAV', 'rb')
dummy[1, :] = fromstring(wr.readframes(wr.getnframes()), dtype = int16)
dataset.append(dummy)

dummy = np.zeros((2, 2752512))
wr = wave.open('./data/test02_ch1.WAV', 'rb')
dummy[0, :] = fromstring(wr.readframes(wr.getnframes()), dtype = int16)
wr = wave.open('./data/test02_ch2.WAV', 'rb')
dummy[1, :] = fromstring(wr.readframes(wr.getnframes()), dtype = int16)
dataset.append(dummy)

dummy = np.zeros((2, 2752512))
wr = wave.open('./data/test03_ch1.WAV', 'rb')
dummy[0, :] = fromstring(wr.readframes(wr.getnframes()), dtype = int16)
wr = wave.open('./data/test03_ch2.WAV', 'rb')
dummy[1, :] = fromstring(wr.readframes(wr.getnframes()), dtype = int16)
dataset.append(dummy)

# eq1 に異常があるかないか
sess = tf.Session()
x = tf.placeholder("float", shape=[None, 20000])
y_ = tf.placeholder("float", shape=[None, 3])

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

def conv2d_stride2(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 2, 2, 1], padding='VALID')

def max_pool_2_2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')
def max_pool_1_4(x):
    return tf.nn.max_pool(x, ksize=[1, 1, 4, 1],
                          strides=[1, 1, 4, 1], padding='SAME')

def max_pool_1_5(x):
    return tf.nn.max_pool(x, ksize=[1, 1, 5, 1],
                          strides=[1, 1, 5, 1], padding='SAME')
    
# RMSの計算
def calRms(data):
    square = np.power(data,2)
    rms = math.sqrt(sum(square)/len(data))
    return rms

# バッチ作成
def make_batch(batch_size):
    batch = np.zeros((batch_size, 20000))
    batch = np.array(batch, dtype=np.float32)
    output = np.zeros((batch_size, 3))
    output = np.array(output, dtype=np.int32)
    
    for i in range(40):
        index = random.randint(0, 2000000)
        dummyMatrix = dataset[0][:, index:index + 10000].T
        dummyMatrix[:, 0] /= calRms(dummyMatrix[:, 0])
        dummyMatrix[:, 1] /= calRms(dummyMatrix[:, 1])
        batch[i, :] = np.reshape(dummyMatrix, (1, 20000))
        output[i, 0] = 1
        
    for i in range(40, 80):
        index = random.randint(0, 2000000)
        dummyMatrix = dataset[1][:, index:index + 10000].T
        dummyMatrix[:, 0] /= calRms(dummyMatrix[:, 0])
        dummyMatrix[:, 1] /= calRms(dummyMatrix[:, 1])
        batch[i, :] = np.reshape(dummyMatrix, (1, 20000))
        output[i, 1] = 1
    for i in range(80, 120):
        index = random.randint(0, 2000000)
        dummyMatrix = dataset[2][:, index:index + 10000].T
        dummyMatrix[:, 0] /= calRms(dummyMatrix[:, 0])
        dummyMatrix[:, 1] /= calRms(dummyMatrix[:, 1])
        batch[i, :] = np.reshape(dummyMatrix, (1, 20000))
        output[i, 2] = 1
    
    return batch, output

# testバッチ作成
def make_batch_test(batch_size):
    batch = np.zeros((batch_size, 20000))
    batch = np.array(batch, dtype=np.float32)
    output = np.zeros((batch_size, 3))
    output = np.array(output, dtype=np.int32)
    
    for i in range(40):
        index = random.randint(2010000, 2700000)
        dummyMatrix = dataset[0][:, index:index + 10000].T
        dummyMatrix[:, 0] /= calRms(dummyMatrix[:, 0])
        dummyMatrix[:, 1] /= calRms(dummyMatrix[:, 1])
        batch[i, :] = np.reshape(dummyMatrix, (1, 20000))
        output[i, 0] = 1
        
    for i in range(40, 80):
        index = random.randint(2010000, 2700000)
        dummyMatrix = dataset[1][:, index:index + 10000].T
        dummyMatrix[:, 0] /= calRms(dummyMatrix[:, 0])
        dummyMatrix[:, 1] /= calRms(dummyMatrix[:, 1])
        batch[i, :] = np.reshape(dummyMatrix, (1, 20000))
        output[i, 1] = 1
    for i in range(80, 120):
        index = random.randint(2010000, 2700000)
        dummyMatrix = dataset[2][:, index:index + 10000].T
        dummyMatrix[:, 0] /= calRms(dummyMatrix[:, 0])
        dummyMatrix[:, 1] /= calRms(dummyMatrix[:, 1])
        batch[i, :] = np.reshape(dummyMatrix, (1, 20000))
        output[i, 2] = 1
    
    return batch, output


W_conv1 = weight_variable([9, 9, 2, 64])
b_conv1 = bias_variable([64])
x_image = tf.reshape(x, [-1, 100, 100, 2])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2_2(h_conv1)

W_conv2 = weight_variable([5, 5, 64, 128])
b_conv2 = bias_variable([128])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2_2(h_conv2)

W_conv3 = weight_variable([4, 4, 128, 256])
b_conv3 = bias_variable([256])
h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
h_pool3 = max_pool_2_2(h_conv3)

W_conv4 = weight_variable([4, 4, 256, 512])
b_conv4 = bias_variable([512])
h_conv4 = tf.nn.relu(conv2d(h_pool3, W_conv4) + b_conv4)
h_pool4 = max_pool_2_2(h_conv4)

W_conv5 = weight_variable([3, 3, 512, 1024])
b_conv5 = bias_variable([1024])
h_conv5 = tf.nn.relu(conv2d(h_pool4, W_conv5) + b_conv5)

W_fc1 = weight_variable([1024, 128])
b_fc1 = bias_variable([128])
h_flat = tf.reshape(h_conv5, [-1, 1024])
h_fc1 = tf.nn.relu(tf.matmul(h_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([128, 32])
b_fc2 = bias_variable([32])
h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)

W_fc3 = weight_variable([32, 3])
b_fc3 = bias_variable([3])
y_conv = tf.nn.softmax(tf.matmul(h_fc2_drop, W_fc3) + b_fc3)

cross_entropy = -tf.reduce_sum(y_ * tf.log(tf.clip_by_value(y_conv,1e-10,1.0)))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

tf.summary.scalar("train_loss", cross_entropy)
tf.summary.scalar("train_accuracy", accuracy)
# 全ての可視化対象を統合して一つの操作にまとめる
# merged_summary_op = tf.summary.merge_all()

sess.run(tf.initialize_all_variables())
# summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())
for i in range(training_epochs):
    batch, output = make_batch(batch_size)
    train_accuracy = accuracy.eval(session=sess, feed_dict={x: batch, y_: output, keep_prob: 1.0})
    # summary = merged_summary_op.eval(session=sess, feed_dict={x: batch, y_: output, keep_prob: 1.0})
    accMatrix[i] = train_accuracy
    # summary_writer.add_summary(summary, i)
    
    train_step.run(session=sess, feed_dict={x: batch, y_: output, keep_prob: 0.5})

    batch, output = make_batch_test(batch_size)
    test_accuracy = accuracy.eval(session=sess, feed_dict={x: batch, y_: output, keep_prob: 1.0})
    testMatrix[i] = test_accuracy

    if i%10 ==0:
        loss = cross_entropy.eval(session=sess, feed_dict={x: batch, y_: output, keep_prob: 1.0})
        print(i)
        print(train_accuracy)
        print(test_accuracy)
        print(loss)
# summary_writer.close()
batch, output = make_batch_test(batch_size)
y_conv = y_conv.eval(session=sess, feed_dict={x: batch, y_: output, keep_prob: 1.0})
CNNdict = {"accMatrix":accMatrix, "testMatrix":testMatrix, "y_conv":y_conv}
pickle.dump(CNNdict, open('CNNcmtdict.pickle', mode='wb'))
