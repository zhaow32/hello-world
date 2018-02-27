#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 25 15:16:47 2018

@author: wenjiezhao
"""


from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True);
import tensorflow as tf
sess = tf.InteractiveSession()

##placeholder
X = tf.placeholder(tf.float32, shape=[None, 784]) #28 * 28 taille image 1 = 1pixel car noir et blanc "X" valeur
y_ = tf.placeholder(tf.float32, shape = [None, 10])

##varialbes
W = tf.Variable(tf.zeros([784, 10])) # 28*28 = 784 , 10 -> 0 à 9  "W" = weight = poid
b = tf.Variable(tf.zeros([10])) #chiffre de 0 à 9 a reconnaitre "b" = constante 
sess.run(tf.initialize_all_variables())

#prediction and loss function
y= tf.nn.softmax(tf.matmul(tf.reshape(X,[-1,784]), W) + b) #fonction "matmul": produit matriciel "-1": reussite obligatoire
##y=tf.matmul(X,W)+b
#Place holder

# 3. Predicted Class and Loss Function
##cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))
cross_entropy=-tf.reduce_sum(y_*tf.log(y))
# Train the Model

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
for i in range(1000):
    batch = mnist.train.next_batch(50)
    train_step.run(feed_dict={X: batch[0], y_: batch[1]})

# Evaluate the Model
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(accuracy.eval(feed_dict={X: mnist.test.images, y_: mnist.test.labels}))



##build a Multilayer Convolutional Network
def weight_variable(shape):
    initial=tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)
def bias_variable(shape):
    initial=tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')
def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],
                          strides=[1,2,2,1],padding='SAME')
    
##第一层卷积
    
##由一个conv加一个maxpooling完成。在每个5*5的patch中输出32个特征，所以权重张量
##形状是【5,5,1,32】，前两个是patch的大小，第三个是输入通道的数目，最后是输出的通道数目
W_conv1=weight_variable([5,5,1,32])
b_conv1=bias_variable([32])

##为了用这一层，我们把x变成一个4d向量，其第2、第3维对应图片的宽、高，最后一维代表图片的颜色通道数
##(因为是灰度图所以这里的通道数为1，如果是rgb彩色图，则为3)。
x_image=tf.reshape(X,[-1,28,28,1])

# then we covolve x_image with the weight tensor, add the bias, apply the RuLu
##function, and finally maxpooling.

h_conv1=tf.nn.relu(conv2d(x_image, W_conv1)+ b_conv1)
h_pool1=max_pool_2x2(h_conv1)



##第二层卷积
W_conv2=weight_variable([5,5,32,64])
b_conv2=bias_variable([64])

h_conv2=tf.nn.relu(conv2d(h_pool1,W_conv2)+ b_conv2)
h_pool2=max_pool_2x2(h_conv2)

##密集链接层

##现在图片尺寸减小到7*7，我们加入一个有1024个神经元的full-connected layer用于处理整个图片
##我们把池化输出的张量reshape成一些向量，乘上权重矩阵，加上bias，然后对其使用ReLu
W_fc1=weight_variable([7 * 7 * 64,1024])
b_fc1=bias_variable([1024])

h_pool2_flat=tf.reshape(h_pool2,[-1,7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1) + b_fc1)


##dropout
## 为了减少过拟合，我们在输出层之前加入dropout，我们用一个placeholder来代表一个神经元的输出
##在dropout中保持不变的概率。这样我们可以在训练过程中启用dropout。在测试中关闭dropout。
##tensorflow 的tf.nn.dropout操作除了可以屏蔽神经元的输出外，还会自动处理神经元输出值的scale
##所以用dropout的时候可以不用考虑scale

keep_prob=tf.placeholder("float")
h_fc1_drop=tf.nn.dropout(h_fc1, keep_prob)

##输出层

W_fc2=weight_variable([1024,10])
b_fc2=bias_variable([10])

##y_conv=tf.matmul(h_fc1_drop,W_fc2)+b_fc2

y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop,W_fc2)+ b_fc2)

##评估
cross_entropy= -tf.reduce_sum(y_ *tf.log(y_conv))
##cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.initialize_all_variables())

for i in range(20000):
    batch = mnist.train.next_batch(50) ##change batch size will change the accuracy
    if i%100 == 0:
        train_accuracy = accuracy.eval(session=sess,feed_dict={X: batch[0], y_: batch[1], keep_prob: 0.5})
        print("step %d, training accuracy %g"%(i, train_accuracy))
    train_step.run(feed_dict={X: batch[0], y_: batch[1], keep_prob: 0.5})

print("test accuracy %g"%accuracy.eval(feed_dict={X: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
