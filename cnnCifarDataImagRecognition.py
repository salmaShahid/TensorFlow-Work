#The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class.
#  There are 50000 training images and 10000 test images.
import matplotlib.pyplot as plt
import numpy as np

#load data
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        cifar_dict = pickle.load(fo, encoding='bytes')
    return cifar_dict

dirs = ['batches.meta','data_batch_1','data_batch_2','data_batch_3','data_batch_4','data_batch_5','test_batch']
all_data = [0,1,2,3,4,5,6]
CIFAR_DIR = 'C:/Users/LeNoVo T430/PycharmProjects/MlLib/Tensorflow/cifar-10-batches-py/'
for i,direc in zip(all_data,dirs):
    all_data[i] = unpickle(CIFAR_DIR+direc)
batch_meta = all_data[0]
data_batch1 = all_data[1]
data_batch2 = all_data[2]
data_batch3 = all_data[3]
data_batch4 = all_data[4]
data_batch5 = all_data[5]
test_batch = all_data[6]

print(batch_meta)
print(data_batch1.keys())
# Put the code here that transforms the X array!
X=data_batch1[b'data'].reshape(10000,3,32,32).transpose(0,2,3,1)
plt.imshow(X[101])
plt.show()

#helping function
def one_hot_encode(vec, vals=10):
    '''
    For use to one-hot encode the 10- possible labels
    '''
    n = len(vec)
    out = np.zeros((n, vals))
    out[range(n), vec] = 1
    return out


class CifarHelper():

    def __init__(self):
        self.i = 0

        self.all_train_batches = [data_batch1, data_batch2, data_batch3, data_batch4, data_batch5]
        self.test_batch = [test_batch]

        self.training_images = None
        self.training_labels = None

        self.test_images = None
        self.test_labels = None

    def set_up_images(self):
        print("Setting Up Training Images and Labels")

        self.training_images = np.vstack([d[b"data"] for d in self.all_train_batches])
        train_len = len(self.training_images)

        self.training_images = self.training_images.reshape(train_len, 3, 32, 32).transpose(0, 2, 3, 1) / 255
        self.training_labels = one_hot_encode(np.hstack([d[b"labels"] for d in self.all_train_batches]), 10)

        print("Setting Up Test Images and Labels")

        self.test_images = np.vstack([d[b"data"] for d in self.test_batch])
        test_len = len(self.test_images)

        self.test_images = self.test_images.reshape(test_len, 3, 32, 32).transpose(0, 2, 3, 1) / 255
        self.test_labels = one_hot_encode(np.hstack([d[b"labels"] for d in self.test_batch]), 10)

    def next_batch(self, batch_size):
        x = self.training_images[self.i:self.i + batch_size].reshape(100, 32, 32, 3)
        y = self.training_labels[self.i:self.i + batch_size]
        self.i = (self.i + batch_size) % len(self.training_images)
        return x, y


#use above code like this
#  Before tf.Session run these two lines
ch = CifarHelper()
ch.set_up_images()

# During session to grab the next batch use this line
# (Just like  mnist.train.next_batch)
# batch = ch.next_batch(100)

#create model
import tensorflow as tf
#Create 2 placeholders, x and y_true. Their shapes should be:
# x shape = [None,32,32,3]
# y_true shape = [None,10]

x=tf.placeholder(tf.float32,shape=[None,32,32,3])
y_true=tf.placeholder(tf.float32,shape=[None,10])

#Create one more placeholder called hold_prob. No need for shape here.
# This placeholder will just hold a single probability for the dropout.
hold_prob=tf.placeholder(tf.float32)
"""
Helper Functions
Grab the helper functions from MNIST with CNN we'll need:

init_weights
init_bias
conv2d
max_pool_2by2
convolutional_layer
normal_full_layer
"""

#Creating the Weight Function
def init_weights(shape):
    init_random_dist=tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(init_random_dist)

#Creating the Bias Function
def init_bias(shape):
    init_bias_rand=tf.constant(0.1,shape=shape)
    return tf.Variable(init_bias_rand)


#Creating the conv2d
def conv2d(x,w):
    return tf.nn.conv2d(x,w,strides=[1,1,1,1],padding="SAME")

#Creating max_pool_2by2
def max_pool_2by2(x):
    return tf.nn.max_pool(x,
                         ksize=[1,2,2,1],
                         strides=[1,2,2,1],
                         padding="SAME")

#Creating a Convolutional Layer
def convolutional_layer(input_x,shape):
    w=init_weights(shape)
    b=init_bias([shape[3]])
    return tf.nn.relu(conv2d(input_x,w)+b) #relu activation func

#Creating a Normal Layer
def normal_layer(input_layer,size):
    input_size=int(input_layer.get_shape()[1])
    w=init_weights([input_size,size])
    b=init_bias([size])
    return tf.matmul(input_layer,w)+b

"""
Create a convolutional layer and a pooling layer as we did for MNIST. Its up to you what the 2d size of the convolution should be, but the last two digits need to be 3 and 32 because of the 3 color channels and 32 pixels. So for example you could use:
    convo_1 = convolutional_layer(x,shape=[4,4,3,32])
"""

convo_1 = convolutional_layer(x,shape=[4,4,3,32])
convo_1_pooling=max_pool_2by2(convo_1)
#Create the next convolutional and pooling layers. The last two dimensions of the convo_2 layer should be 32,64
convo_2 = convolutional_layer(convo_1_pooling,shape=[4,4,32,64])
convo_2_pooling=max_pool_2by2(convo_2)

# Now create a flattened layer by reshaping the pooling layer into [-1,8 * 8 * 64] or [-1,4096]
print(8*8*64)

#convo flatting
convo_flat=tf.reshape(convo_2_pooling,[-1,8*8*64])
#Create a new full layer using the normal_full_layer function and passing in your flattend
# convolutional 2 layer with size=1024.
#  (we would also choose to reduce this to something like 512)

#Creating a normal full layer
full_layer_one=tf.nn.relu(normal_layer(convo_flat,512))

#Now create the dropout layer with tf.nn.dropout, remember to pass in your hold_prob placeholder.
dropout_layer=tf.nn.dropout(full_layer_one,hold_prob)

# Finally set the output to y_pred by passing in the dropout layer into the
# normal_full_layer function. The size should be 10 because of the 10 possible labels
y_pred=normal_layer(dropout_layer,10)

# Loss Function
# Create a cross_entropy loss function
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true,logits=y_pred))

# Create the optimizer using an Adam Optimizer.
optimizer=tf.train.AdamOptimizer(learning_rate=0.001)
train=optimizer.minimize(cross_entropy)

# Create a variable to intialize all the global tf variables.
init=tf.global_variables_initializer()

# Perform the training and test print outs in a Tf session and run your model!

with tf.Session() as sess:
    sess.run(init)

    for i in range(5000):
        batch = ch.next_batch(100)
        sess.run(train, feed_dict={x: batch[0], y_true: batch[1], hold_prob: 0.5})

        # PRINT OUT A MESSAGE EVERY 100 STEPS
        if i % 100 == 0:
            print('Currently on step {}'.format(i))
            print('Accuracy is:')
            # Test the Train Model
            matches = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_true, 1))

            acc = tf.reduce_mean(tf.cast(matches, tf.float32))

            print(sess.run(acc, feed_dict={x: ch.test_images, y_true: ch.test_labels, hold_prob: 1.0}))
            print('\n')