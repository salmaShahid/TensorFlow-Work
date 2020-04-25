import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)
print(type(mnist))
print((type(mnist.train.images)))
print(mnist)
#Chicking Images
print(mnist.train.images)

print(mnist.train.num_examples)
print(mnist.test.num_examples)
print(mnist.validation.num_examples)

import matplotlib.pyplot as plt
single_image=mnist.train.images[2].reshape(28,28)
plt.imshow(single_image,cmap="gist_gray")


#PLACEHOLDER
x=tf.placeholder(tf.float32,[None,784])

#VARIABLES
W=tf.Variable(tf.zeros([784,10]))
b=tf.Variable(tf.zeros([10]))

#GRAPH OPERATIONS
y=tf.matmul(x,W)+b

#lOSS FUNCTION
y_true=tf.placeholder(tf.float32,[None,10])
cross_entropy=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true,logits=y))

#OPTIMIZER
optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.5)
train=optimizer.minimize(cross_entropy)

# Session
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for step in range(1000):
        x_batch, y_batch = mnist.train.next_batch(100)
        sess.run(train, feed_dict={x: x_batch, y_true: y_batch})

    # EVALUATE THE MODEL
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_true, 1))

    acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    print(sess.run(acc, feed_dict={x: mnist.test.images, y_true: mnist.test.labels}))