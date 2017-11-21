import tensorflow as tf

from config import Config
import numpy as np
import os
import time

from multiprocessing import Pool

from helperFunctions import getUCF101
from helperFunctions import loadFrame

import h5py
import cv2

# Define FLAGS

flags = tf.app.flags
tf.flags.DEFINE_float("MOVING_AVERAGE_DECAY", 0.9, "Decay")
tf.flags.DEFINE_float("batch_norm_DECAY", 0.9, "BN Decay")
tf.flags.DEFINE_float("batch_norm_EPSILON",0.001 , "Epsilon")
tf.flags.DEFINE_float("FC_WEIGHT_DECAY",0.00004 , "FC_WEIGHT_DECAY")
tf.flags.DEFINE_float("FC_WEIGHT_STDDEV",0.01 , "F C _WEIGHT_STDDEV")
FLAGS = tf.flags.FLAGS
RESNET_VARIABLES = 'resnet_variables'
UPDATE_OPS_COLLECTION = 'resnet_update_ops'  # must be grouped with training op



mean_subtract = np.asarray([103.062623801, 115.902882574, 123.151630838],np.float32)

tf.app.flags.DEFINE_integer('input_size', 224, "input image size")


# parts of model functions
def stack(x, c):
    for n in range(c['num_blocks']):
        s = c['stack_stride'] if n == 0 else 1
        c['block_stride'] = s
        with tf.variable_scope('block%d' % (n + 1)):
            x = block(x, c)
    return x


def block(x, c):
    filters_in = x.get_shape()[-1]
    m = 4 if c['bottleneck'] else 1
    filters_out = m * c['block_filters_internal']

    shortcut = x  # branch 1

    c['conv_filters_out'] = c['block_filters_internal']

    if c['bottleneck']:
        with tf.variable_scope('a'):
            c['ksize'] = 1
            c['stride'] = c['block_stride']
            x = conv(x, c)
            x = batch_norm(x, c)
            x = tf.nn.relu(x)

        with tf.variable_scope('b'):
            x = conv(x, c)
            x = batch_norm(x, c)
            x = tf.nn.relu(x)

        with tf.variable_scope('c'):
            c['conv_filters_out'] = filters_out
            c['ksize'] = 1
            assert c['stride'] == 1
            x = conv(x, c)
            x = batch_norm(x, c)
    else:
        with tf.variable_scope('A'):
            c['stride'] = c['block_stride']
            assert c['ksize'] == 3
            x = conv(x, c)
            x = batch_norm(x, c)
            x = tf.nn.relu(x)

        with tf.variable_scope('B'):
            c['conv_filters_out'] = filters_out
            assert c['ksize'] == 3
            assert c['stride'] == 1
            x = conv(x, c)
            x = batch_norm(x, c)

    with tf.variable_scope('shortcut'):
        if filters_out != filters_in or c['block_stride'] != 1:
            c['ksize'] = 1
            c['stride'] = c['block_stride']
            c['conv_filters_out'] = filters_out
            shortcut = conv(shortcut, c)
            shortcut = batch_norm(shortcut, c)

    return tf.nn.relu(x + shortcut)

def weight_variable(shape):
    initializer = tf.truncated_normal_initializer(stddev=0.1)
    weights = _get_variable('weights',shape=shape,dtype='float',initializer=initializer,weight_decay=0.0004)
    return weights


def batch_norm(x, c):
    return tf.contrib.layers.batch_norm(x,is_training=c['is_training'],center=True,scale=True,decay=FLAGS.batch_norm_DECAY,updates_collections=UPDATE_OPS_COLLECTION)
    
def bias_variable(shape):
    biases = _get_variable('biases',shape=shape,initializer=tf.zeros_initializer())
    return biases

def fc(x, c):
    num_units_in = x.get_shape()[1]
    num_units_out = c['fc_units_out']
    shape=[num_units_in, num_units_out]
    weights = weight_variable(shape)
    biases = bias_variable([num_units_out])
    x = tf.nn.xw_plus_b(x, weights, biases)
    return x


def _get_variable(name,shape,initializer,weight_decay=0.0,dtype='float',trainable=True):
    if weight_decay > 0:
        regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
    else:
        regularizer = None
    collections = [tf.GraphKeys.GLOBAL_VARIABLES, RESNET_VARIABLES]
    return tf.get_variable(name,shape=shape,initializer=initializer,dtype=dtype,regularizer=regularizer,collections=collections,trainable=trainable)


def conv(x, c):

    filters_in = x.get_shape()[-1]
    s = [c['ksize'], c['ksize'], filters_in, c['conv_filters_out']]
    
    weights=weight_variable(s)
    return tf.nn.conv2d(x, weights, [1, c['stride'], c['stride'], 1], padding='SAME')


def max_pool(x, ksize=3, stride=2):
    return tf.nn.max_pool(x,ksize=[1, ksize, ksize, 1],strides=[1, stride, stride, 1],padding='SAME')

# model definition
def inference(x, is_training,
              num_classes=1000,
              num_blocks=[3, 4, 6, 3],  # defaults to 50-layer network
              use_bias=False, # defaults to using batch norm
              bottleneck=True,
              keep_prob=0.5):
    c = Config()
    c['bottleneck'] = bottleneck
    c['is_training'] = is_training

    c['ksize'] = 3
    c['stride'] = 1
    c['use_bias'] = use_bias
    c['fc_units_out'] = num_classes
    c['num_blocks'] = num_blocks
    c['stack_stride'] = 2

    with tf.variable_scope('scale1'):
        c['conv_filters_out'] = 64
        c['ksize'] = 7
        c['stride'] = 2
        x = conv(x, c)
        x = batch_norm(x, c)
        x = tf.nn.relu(x)

    with tf.variable_scope('scale2'):
        x = max_pool(x, ksize=3, stride=2)
        c['num_blocks'] = num_blocks[0]
        c['stack_stride'] = 1
        c['block_filters_internal'] = 64
        x = stack(x, c)

    with tf.variable_scope('scale3'):
        c['num_blocks'] = num_blocks[1]
        c['block_filters_internal'] = 128
        assert c['stack_stride'] == 2
        x = stack(x, c)

    with tf.variable_scope('scale4'):
        c['num_blocks'] = num_blocks[2]
        c['block_filters_internal'] = 256
        x = stack(x, c)

    with tf.variable_scope('scale5'):
        c['num_blocks'] = num_blocks[3]
        c['block_filters_internal'] = 512
        x = stack(x, c)

    # post-net
    x = tf.reduce_mean(x, reduction_indices=[1, 2], name="avg_pool")

    if num_classes != None:
        with tf.variable_scope('fc'):
            output = fc(tf.nn.dropout(x,keep_prob=keep_prob), c)

    return output, tf.nn.softmax(output), x

IMAGE_SIZE = 224
NUM_CLASSES = 101
batch_size = 32


#Define placeholders
X = tf.placeholder(tf.float32, [None, IMAGE_SIZE, IMAGE_SIZE, 3])
y = tf.placeholder(name='label',dtype=tf.float32,shape=[None,NUM_CLASSES])
is_training = tf.placeholder(tf.bool ,shape=())
keep_prob = tf.placeholder(tf.float32 ,shape=())

logits, pred, features = inference(X,
                         num_classes=101,
                         is_training=is_training,
                         bottleneck=True,
                         num_blocks=[3, 4, 6, 3],
                         use_bias=False,
                         keep_prob=keep_prob)


#Cross entropy loss
cross_entropy_mean  = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = y))
#Regularization Loss 
regularization_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
#Adding up
loss = tf.add_n([cross_entropy_mean] + regularization_loss)

#Mask that checks if the prediction is correct
correct_prediction = tf.equal(tf.argmax(logits,1),tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

tvar = tf.trainable_variables()
all_vars = tf.global_variables()



opt = tf.train.MomentumOptimizer(0.001, 0.9)
grads = opt.compute_gradients(loss)

apply_gradient_op = opt.apply_gradients(grads)

batchnorm_updates = tf.get_collection(UPDATE_OPS_COLLECTION)
batchnorm_updates_op = tf.group(*batchnorm_updates)
train_op = tf.group(apply_gradient_op, batchnorm_updates_op)

#start session
sess = tf.InteractiveSession()

init = tf.global_variables_initializer()
sess.run(init)

saver = tf.train.Saver(all_vars)
saver.restore(sess,tf.train.latest_checkpoint('basemodel_UCF101/'))

#Directory for UCF 101 dataset
data_directory = '/projects/eot/balq/UCF101/'
class_list, train, test = getUCF101(base_directory = data_directory)

pool_threads = Pool(10,maxtasksperchild=200)

#Run for 20 epochs
for epoch in range(0,20):

    ##### TEST
    test_accuracy = 0.0
    accuracy_count = 0
    random_indices = np.random.permutation(len(test[0]))
    t1 = time.time()
    for i in range(0,500-batch_size,batch_size):
        augment = False
        video_list = [(test[0][k],augment) 
                        for k in random_indices[i:(batch_size+i)]]
        data = pool_threads.map(loadFrame,video_list)

        next_batch = 0
        for video in data:
            if video.size==0: # there was an exception, skip this batch
                next_batch = 1
        if(next_batch==1):
            continue

        data = np.asarray(data,dtype=np.float32)

        labels_batch = np.asarray(
                        test[1][random_indices[i:(batch_size+i)]]
                        )
        y_batch = np.zeros((batch_size,NUM_CLASSES),dtype=np.float32)
        y_batch[np.arange(batch_size),labels_batch] = 1

        acc_p = sess.run(accuracy,
                feed_dict={X: data, y: y_batch, is_training:False, keep_prob:1.0})

        test_accuracy += acc_p
        accuracy_count += 1
    test_accuracy = test_accuracy/accuracy_count
    print('t:%f TEST:%f' % (float(time.time()-t1), test_accuracy))

    ###### TRAIN
    random_indices = np.random.permutation(len(train[0]))
    for i in range(0, len(train[0])-batch_size,batch_size):

        t1 = time.time()

        augment = True
        video_list = [(train[0][k],augment)
                       for k in random_indices[i:(batch_size+i)]]
        data = pool_threads.map(loadFrame,video_list)

        next_batch = 0
        for video in data:
            if video.size==0: # there was an exception, skip this
                next_batch = 1
        if(next_batch==1):
            continue

        data = np.asarray(data,dtype=np.float32)

        t_data_load = time.time()-t1

        labels_batch = np.asarray(train[1][random_indices[i:(batch_size+i)]])
        y_batch = np.zeros((batch_size,NUM_CLASSES),dtype=np.float32)
        y_batch[np.arange(batch_size),labels_batch] = 1

        _, loss_p, acc_p = sess.run([train_op,loss,accuracy],
                feed_dict={X: data, y: y_batch, is_training:True, keep_prob:0.5})

        t_train = time.time() - t1 - t_data_load

        print('epoch: %d i: %d t_load: %f t_train: %f loss: %f acc: %f'
                     % (epoch,i,t_data_load,t_train,loss_p,acc_p))
        
if not os.path.exists('single_frame_model/'):
    os.makedirs('single_frame_model/')
all_vars = tf.global_variables()
saver = tf.train.Saver(all_vars)
saver.save(sess,'single_frame_model/model')

pool_threads.close()
pool_threads.terminate()

prediction_directory = 'UCF-101-predictions/'
if not os.path.exists(prediction_directory):
    os.makedirs(prediction_directory)
for label in class_list:
    if not os.path.exists(prediction_directory+label+'/'):
        os.makedirs(prediction_directory+label+'/')
        
acc_top1 = 0.0
acc_top5 = 0.0
acc_top10 = 0.0
confusion_matrix = np.zeros((NUM_CLASSES,NUM_CLASSES),dtype=np.float32)
random_indices = np.random.permutation(len(test[0]))
for i in range(len(test[0])):

    t1 = time.time()

    index = random_indices[i]

    filename = test[0][index]
    filename = filename.replace('.avi','.hdf5')
    filename = filename.replace('UCF-101','UCF-101-hdf5')

    h = h5py.File(filename,'r')
    nFrames = len(h['video'])

    data = np.zeros((nFrames,IMAGE_SIZE,IMAGE_SIZE,3),dtype=np.float32)
    prediction = np.zeros((nFrames,NUM_CLASSES),dtype=np.float32)

    for j in range(nFrames):
        frame = h['video'][j] - mean_subtract
        frame = cv2.resize(frame,(IMAGE_SIZE,IMAGE_SIZE))
        frame = frame.astype(np.float32)
        data[j,:,:,:] = frame
    h.close()

    loop_i = list(range(0,nFrames,200))
    loop_i.append(nFrames)

    for j in range(len(loop_i)-1):
        data_batch = data[loop_i[j]:loop_i[j+1]]

        curr_pred = sess.run(pred,
            feed_dict={X: data_batch, is_training:False, keep_prob:1.0})
        prediction[loop_i[j]:loop_i[j+1]] = curr_pred

    filename = filename.replace(data_directory+'UCF-101-hdf5/',prediction_directory)
    if(not os.path.isfile(filename)):
        with h5py.File(filename,'w') as h:
            h.create_dataset('predictions',data=prediction)

    label = test[1][index]
    prediction = np.sum(np.log(prediction),axis=0)
    argsort_pred = np.argsort(-prediction)[0:10]

    confusion_matrix[label,argsort_pred[0]] += 1
    if(label==argsort_pred[0]):
        acc_top1 += 1.0
    if(np.any(argsort_pred[0:5]==label)):
        acc_top5 += 1.0
    if(np.any(argsort_pred[:]==label)):
        acc_top10 += 1.0

    print('i:%d nFrames:%d t:%f (%f,%f,%f)' 
          % (i,nFrames,time.time()-t1,acc_top1/(i+1),acc_top5/(i+1), acc_top10/(i+1)))

number_of_examples = np.sum(confusion_matrix,axis=1)
for i in range(NUM_CLASSES):
    confusion_matrix[i,:] = confusion_matrix[i,:]/np.sum(confusion_matrix[i,:])

results = np.diag(confusion_matrix)
indices = np.argsort(results)

sorted_list = np.asarray(class_list)
sorted_list = sorted_list[indices]
sorted_results = results[indices]

for i in range(NUM_CLASSES):
    print(sorted_list[i],sorted_results[i],number_of_examples[indices[i]])
