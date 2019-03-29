# -*- coding: utf-8 -*-

import tensorflow as tf
import math
import numpy as np
import random
import pickle
import wave
from scipy import fromstring, int16
from scipy.fftpack import fft

batch_size = 100
training_epochs = 130000
num_of_kinds = 200
lossMatrix = np.zeros((training_epochs))
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
x = tf.placeholder("float", shape=[None, 200])
y_ = tf.placeholder("float", shape=[None, 200])

# 荷重作成
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

# バイアス作成
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

# RMSの計算
def calRms(data):
    square = np.power(data,2)
    rms = math.sqrt(sum(square)/len(data))
    return rms

# fftしてRMSを用いて短冊化する
def processing_data(sample):
    sample = sample/(calRms(sample))
    sample = fft(sample)
    sample = abs(sample)
    new = np.zeros((100))
    new = np.array(new, dtype=np.float32)
    for i in range(100):
        new[i] = calRms(sample[10*i:10*i+15])
    return new

# バッチ作成
def make_batch(batch_size):
    batch = np.zeros((batch_size, 200))
    batch = np.array(batch, dtype=np.float32)
    output = np.zeros((batch_size, 200))
    output = np.array(output, dtype=np.float32)
    
    for i in range(batch_size):
        index = random.randint(0, 2000000)
        batch[i, 0:100] = processing_data(dataset[0][0, index:index + 10000])
        batch[i, 100:200] = processing_data(dataset[0][1, index:index + 10000])
        output[i, 0:100] = processing_data(dataset[0][0, index:index + 10000])
        output[i, 100:200] = processing_data(dataset[0][1, index:index + 10000])
    
    return batch, output

# testバッチ作成
def make_batch_test(batch_size):
    batch = np.zeros((batch_size, 200))
    batch = np.array(batch, dtype=np.float32)
    output = np.zeros((batch_size, 200))
    output = np.array(output, dtype=np.float32)
    
    for i in range(batch_size):
        index = random.randint(2010000, 2650000)
        batch[i, 0:100] = processing_data(dataset[0][0, index:index + 10000])
        batch[i, 100:200] = processing_data(dataset[0][1, index:index + 10000])
        output[i, 0:100] = processing_data(dataset[0][0, index:index + 10000])
        output[i, 100:200] = processing_data(dataset[0][1, index:index + 10000])
    
    return batch, output

W_fc1 = weight_variable([200, 128])
b_fc1 = bias_variable([128])
h_flat = tf.reshape(x, [-1, 200])
h_fc1 = tf.nn.sigmoid(tf.matmul(h_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([128, 32])
b_fc2 = bias_variable([32])
h_fc2 = tf.nn.sigmoid(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)

W_fc3 = weight_variable([32, 128])
b_fc3 = bias_variable([128])
h_fc3 = tf.nn.sigmoid(tf.matmul(h_fc2_drop, W_fc3) + b_fc3)

h_fc3_drop = tf.nn.dropout(h_fc3, keep_prob)

W_fc4 = weight_variable([128, 200])
b_fc4 = bias_variable([200])
y_out = tf.matmul(h_fc3_drop, W_fc4) + b_fc4

each_square = tf.square(y_ - y_out)
loss = tf.reduce_mean(each_square)
train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

ret_loss = tf.matmul(tf.matmul(tf.matmul(tf.matmul(each_square, tf.transpose(W_fc4)), tf.transpose(W_fc3)), tf.transpose(W_fc2)), tf.transpose(W_fc1))

saver = tf.train.Saver()
# sess.run(tf.initialize_all_variables())
saver.restore(sess, "./AEmodel2.ckpt")
for i in range(training_epochs):
    batch, output = make_batch(batch_size)
    train_loss = loss.eval(session=sess, feed_dict={x: batch, y_: output, keep_prob: 1.0})
    train_each_square = each_square.eval(session=sess, feed_dict={x: batch, y_: output, keep_prob: 1.0})
    train_ret_loss = ret_loss.eval(session=sess, feed_dict={x: batch, y_: output, keep_prob: 1.0})
    lossMatrix[i] = train_loss
    
    train_step.run(session=sess, feed_dict={x: batch, y_: output, keep_prob: 0.5})

    batch, output = make_batch_test(batch_size)
    test_loss = loss.eval(session=sess, feed_dict={x: batch, y_: output, keep_prob: 1.0})
    testMatrix[i] = test_loss

    if i%100 == 0:
        print(i)
        print(train_loss)
        print(test_loss)
        batch, output = make_batch_test(batch_size)
        CNNdict = {"lossMatrix":lossMatrix, "testMatrix":testMatrix}
        pickle.dump(CNNdict, open('AEcmtdict2.pickle', mode='wb'))
saver.save(sess, "./AEmodel3.ckpt")
