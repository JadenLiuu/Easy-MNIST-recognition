import numpy as np
import os
import tensorflow as tf
import sys

#### number of classes (5 here, due to 0~4), setting of using dropout
num_cls = 5
using_dropout = True # to do bonus2 or not

    
"""
This function compute accuracy of `x_data` by checking the `y_label`
"""    
def acc(x_data, y_label):
    global prediction
    
    y_hat = sess.run(prediction, {X0: x_data})
    corr = tf.cast(tf.equal(tf.argmax(y_hat, 1), y_label), tf.float32)
    acc = tf.reduce_mean(corr)
    
    result = sess.run(acc, {X0: x_data, Y0: y_label})
    return result

"""
AP is a sub-function used by function mAP, 
which is decided to compute the average precision of each input class `cls_k`
"""
def AP(output, target, cls_k):
    # sort predictions
    sort_ind = np.argsort(output) # sort the prediction to justify the retrieval performance   
    sort_ind = sort_ind[::-1]    
        
    pos = 0.0
    total_count = 0.0
    precision_at_ind = 0.0
    for ind in sort_ind:
        label = target[ind]
        total_count += 1
        if label == cls_k:
            pos += 1
            precision_at_ind += pos/total_count
    
    if pos == 0.0:
        return precision_at_ind
    
    return precision_at_ind / pos
    
"""
By averaging the mean of AP of each classes,
this function returns the mAP of `x_data`
"""
def mAP(x_data, y_label):
    ap = np.zeros(5)
    
    y_hat = sess.run(prediction, {X0: x_data}) # shape: 2558, 5 ; ndarray    
        
    for k in range(num_cls):
        output = y_hat[:, k]                
        ap[k] = AP(output, y_label, k)                
            
    mAP = tf.reduce_mean(tf.convert_to_tensor(ap))
    result = sess.run(mAP, {X0: x_data, Y0: y_label})
    return result

"""
This function computes recall by computing TP/(TP+FN)
"""    
def recall(x_data, y_label):
    tmp = np.zeros(5)
    y_hat = sess.run(prediction, {X0: x_data}) # shape: 2558, 5 ; ndarray    
    predict_label = np.argmax(y_hat, axis=1)    
        
    for k in range(num_cls):        
        label_k = (y_label==k) # bool
        TP_FN = sum(label_k.astype(int)) # number of label k in `y_label`        
        predict_k = np.logical_and((predict_label==k),label_k)        
        TP = sum(predict_k.astype(int)) # number of True positive
        
        tmp[k] = TP/TP_FN
        #print(tmp[k])
    
    recall = tf.reduce_mean(tf.convert_to_tensor(tmp))
    result = sess.run(recall, {X0: x_data, Y0: y_label})
    return result
    

# define variable function
def weight_var(name, shape):
    #normal = tf.Variable(tf.truncated_normal(shape, stddev=0.1, dtype= tf.float64))
    var = tf.get_variable(name, shape=shape, dtype=tf.float64, initializer=tf.contrib.layers.variance_scaling_initializer())
    return var

def bias_var(shape):
    return tf.Variable(tf.constant(0.1, dtype=tf.float64, shape=shape))

def shuffle_aligned_list(data):
    """Shuffle arrays in a list by shuffling each array identically."""
    num = data[0].shape[0]
    p = np.random.permutation(num)
    return [d[p] for d in data]


def batch_generator(data, batch_size, shuffle=True):
    """Generate batches of data.

    Given a list of array-like objects, generate batches of a given
    size by yielding a list of array-like objects corresponding to the
    same slice of each input.
    """
    if shuffle:
        data = shuffle_aligned_list(data)

    batch_count = 0
    while True:
        if batch_count * batch_size + batch_size >= len(data[0]):
            batch_count = 0

            if shuffle:
                data = shuffle_aligned_list(data)

        start = batch_count * batch_size
        end = start + batch_size
        batch_count += 1
        yield [d[start:end] for d in data]
        

###### Do not modify here ###### 

# to make this notebook's output stable across runs
def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)

reset_graph()

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/")

# training on MNIST but only on digits 0 to 4
X_train1 = mnist.train.images[mnist.train.labels < 5]
y_train1 = mnist.train.labels[mnist.train.labels < 5]
X_valid1 = mnist.validation.images[mnist.validation.labels < 5]
y_valid1 = mnist.validation.labels[mnist.validation.labels < 5]
X_test1 = mnist.test.images[mnist.test.labels < 5]
y_test1 = mnist.test.labels[mnist.test.labels < 5]

###### Do not modify here ###### 



#------------------------------------------------------------------------------------Network START
# define placeholder
X0 = tf.placeholder(tf.float64, [None, 784])
Y0 = tf.placeholder(tf.int64, [None])

# define layer 1
w1 = weight_var("w1",[784,128])
L1_output = tf.nn.elu(tf.matmul(X0,w1)+bias_var([128]))

# define layer 2
w2 = weight_var("w2",[128,128])
L2_output = tf.nn.elu(tf.matmul(L1_output,w2)+bias_var([128]))
if using_dropout:
    L2_output = tf.nn.dropout(L2_output, 0.5)

# define layer 3
w3 = weight_var("w3",[128,128])
L3_output = tf.nn.elu(tf.matmul(L2_output,w3)+bias_var([128]))
if using_dropout:
    L3_output = tf.nn.dropout(L3_output, 0.5)

# define layer 4
w4 = weight_var("w4",[128,128])
L4_output = tf.nn.elu(tf.matmul(L3_output,w4)+bias_var([128]))
if using_dropout:
    L4_output = tf.nn.dropout(L4_output, 0.5)

# define layer 5
w5 = weight_var("w5",[128,128])
L5_output = tf.nn.elu(tf.matmul(L4_output,w5)+bias_var([128]))
#if using_dropout:
#    L5_output = tf.nn.dropout(L5_output, 0.5)

# define softmax layer
w_predict = weight_var("w_predict",[128,num_cls])
prediction = tf.matmul(L5_output,w_predict)+bias_var([num_cls])
#------------------------------------------------------------------------------------Network END


# error & optimizer
error = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=prediction, labels=Y0)
loss = tf.reduce_mean(error)
train_step = tf.train.AdamOptimizer(0.001).minimize(loss)

# session
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
sess.run(tf.global_variables_initializer())

# saver
saver = tf.train.Saver()
if not using_dropout:
    if not os.path.exists('net'):
        os.makedirs('net')
    check_path = "net/Team36_HW2.ckpt"
else:
    if not os.path.exists('net_drop'):
        os.makedirs('net_drop')
    check_path = "net_drop/Team36_HW2.ckpt"

# batch setting     
batch_size = 500
generate_batch = batch_generator([X_train1, y_train1], batch_size, shuffle=True)
# training 
last_score = 0.0
for i in range(10000):
    x0, y0 = next(generate_batch)
    sess.run(train_step, feed_dict={X0: x0, Y0: y0})        
    if i%50 == 0:
        val_result = acc(X_valid1, y_valid1)
        print('Acc of val data: {} at iter {iter}'.format(val_result, iter=i))
        if val_result > last_score:                
            if os.path.exists(check_path): ## remove last saved model
                os.remove(check_path)                           
            saver.save(sess, check_path) ## # save the current model to check path                 
            last_score = val_result # update last score
        else:
            break
    

            
using_dropout = False  ## Not using dropout during testing
print('early stop at iteration %d' % i)
print('Acc of test data: {}'.format(acc(X_test1, y_test1)))
print('mAP of test data: {}'.format(mAP(X_test1, y_test1)))
print('Recall of test data: {}'.format(recall(X_test1, y_test1)))  

#=============================================Training process====================================================
#1. Define architecture of this network : [fc-784 to 128]+[fc-128 to 128]*4+[fc-128 to 5]
#    - by calling sub-function weight_var and bias_var to produce Variables.
#    - the initializer of weight variables are `tf.contrib.layers.variance_scaling_initializer()`
#2. compute error by using `sparse_softmax_cross_entropy_with_logits`
#3. Set optimizer with Adam, and set the initial learning rate to 0.001
#4. run session to train this network
#    - for every 50 epochs, we test the network on validation set
#    - In comparison of last 50 epochs, if the score of validation set becomes worse, we do a early stop,
#        otherwise, we remove the net we stored before, and save the current model to net/net_drop directory.
#5. [Bonus2] by setting `using_dropout` at line 8, this network could perform dropout at 2nd&3rd&4th fc layers
#    - default setting of `using_dropout` is True

#---------------------- base result ----------------------

#early stop at iteration 300
#Acc of test data: 0.9916326403617859
#mAP of test data: 0.9961790667255332
#Recall of test data: 0.991575376527425


#---------------------- base result + bonus 2-------------

#early stop at iteration 200
#Acc of test data: 0.9708114266395569
#mAP of test data: 0.9683457847848265
#Recall of test data: 0.9706112350284363

