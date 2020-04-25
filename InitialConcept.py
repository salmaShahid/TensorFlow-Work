import tensorflow as tf


a = tf.Variable(1, name = "a")
b = tf.Variable(2, name="b")
f = a+b

init = tf.global_variables_initializer()
with tf.Session() as s:
    init.run()
    print( f.eval() )


#placeholder
x = tf.placeholder(tf.int32)
y = tf.placeholder(tf.int32)
adds = tf.add(x,y)
with tf.Session() as p:
    print("addition ",p.run(adds,feed_dict={x: 20, y: 30}))


from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data",one_hot=True)
print(type(mnist))
print((type(mnist.train.images)))