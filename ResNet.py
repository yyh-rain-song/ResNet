import tensorflow as tf
import numpy as np

from data_set import Data_set
import load_data


# input = output = [None, 32, 32, 64]
def residual_unit(train, w_1, w_2):
    conv1 = tf.nn.conv2d(input=train, filter=w_1, strides=[1, 1, 1, 1], padding='SAME')
    conv1 = tf.nn.relu(conv1)
    conv2 = tf.nn.conv2d(input=conv1, filter=w_2, strides=[1, 1, 1, 1], padding='SAME')
    conv2 = tf.nn.relu(conv2)
    return tf.add(conv1, conv2)


# input : [None, 4*2, 32, 32]
# output: [None, 2, 32, 32]
def residual_net_model(input_x, conv1_w, conv2_w, res_num, res_w):
    # reshape into [None, 32, 32, 4*2] for convolution
    input_x = tf.reshape(input_x, [-1, 32, 32, 8])
    conv1 = tf.nn.conv2d(input=input_x, filter=conv1_w, strides=[1, 1, 1, 1], padding='SAME')
    # output: [None, 32, 32, 64]
    In = conv1
    for i in range(int(res_num/2)):
        out = residual_unit(In, res_w[2*i], res_w[2*i+1])
        In = out
    out = In
    conv2 = tf.nn.conv2d(input=out, filter=conv2_w, strides=[1, 1, 1, 1], padding='SAME')
    # output: [None, 2, 32, 32]
    return tf.reshape(conv2, [-1, 2, 32, 32])


# input: [None, 2, 32, 32]
# output: [None, 2, 32, 32]
def three_fusion(recent, near, distant, weight):
    shape = tf.shape(recent)

    weight = tf.reshape(weight, (3, 1, 2, 32, 32))
    weight = tf.tile(weight, [1, shape[0], 1, 1, 1])
    output = tf.multiply(recent, weight[0])
    output = tf.add(output, tf.multiply(near, weight[1]))
    output = tf.add(output, tf.multiply(distant, weight[2]))

    return output


# input: [None, 9]
# output: [None, 2, 32, 32]
def fully_connected(input):
    output = tf.layers.dense(input, units=10, activation=tf.nn.relu)
    output = tf.layers.dense(output, units=2*32*32, activation=tf.nn.relu)
    output = tf.reshape(output, (-1, 2, 32, 32))
    print(output.shape)
    return output


def net_model(input_x, extern_input, num_steps, batch_size, max_x, min_x):
    # define placeholder and variable to train
    X = tf.placeholder(tf.float32, [None, 3, 4*2, 32, 32])
    # X [, 3 dimensions, 4 days per dimension, 2, 32, 32]
    E = tf.placeholder(tf.float32, [None, extern_input.shape[1]])
    # extern input shape: [None, 9]
    Y = tf.placeholder(tf.float32, [None, 2, 32, 32])
    conv1_w = tf.Variable(tf.random_normal([3, 3, 3, 2*4, 64], stddev=0.01))
    conv2_w = tf.Variable(tf.random_normal([3, 3, 3, 64, 2], stddev=0.01))
    residual_numer = 4
    res_w = tf.Variable(tf.random_normal([3, residual_numer, 3, 3, 64, 64], stddev=0.01))
    fusion_w = tf.Variable(tf.random_normal([3, 2, 32, 32], stddev=0.01))

    # construct the net
    x_recent = X[:, 0, :, :, :]
    x_near = X[:, 1, :, :, :]
    x_distant = X[:, 2, :, :, :]
    recent_out = residual_net_model(x_recent, conv1_w=conv1_w[0], conv2_w=conv2_w[0],
                                    res_w=res_w[0], res_num=residual_numer)
    near_out = residual_net_model(x_near, conv1_w=conv1_w[1], conv2_w=conv2_w[1],
                                  res_w=res_w[1], res_num=residual_numer)
    distant_out = residual_net_model(x_distant, conv1_w=conv1_w[2], conv2_w=conv2_w[2],
                                     res_w=res_w[2], res_num=residual_numer)
    total = three_fusion(recent_out, near_out, distant_out, fusion_w)
    extern_out = fully_connected(E)
    # prediction: [None, 2, 32, 32]

    prediction = tf.tanh(tf.add(total, extern_out))

    # define loss and optimizer
    loss = tf.reduce_mean(tf.square(prediction - Y))
    eval_matrix = tf.sqrt(loss)*(max_x-min_x)
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
    init_op = tf.global_variables_initializer()
    print("========== Model built ==========")

    # dealing with input
    data_set = Data_set(0.95)
    data_set.build(input_data=input_x, extern_input=extern_input)

    # compute the graph
    stop_round = 0
    minimal_eval = 1000
    with tf.Session() as sess:
        sess.run(init_op)
        for step in range(num_steps):
            x, y, e = data_set.next_batch(batch_size=batch_size)
            _, L, eval = sess.run([optimizer, loss, eval_matrix], feed_dict={X: x, Y: y, E: e})
            print("step {0}, loss {1}, evaluate {2}".format(step, L, eval))
            if eval < minimal_eval:
                minimal_eval = eval
                stop_round = 0
            else:
                stop_round += 1
            # if stop_round == early_stopping:
            #     break
        print("========== Training complete ==========")
        tx, ty, te = data_set.test_data()
        print("=== begin test ====")
        L, eval = sess.run([loss, eval_matrix], feed_dict={X: tx, Y: ty, E: te})
        print("Testing loss {0}, evaluate {1}".format(L, eval))


data_input, extern, timestamp, min_data, max_data = load_data.load_all_data()
net_model(input_x=data_input, extern_input=extern, num_steps=50, batch_size=128, max_x=max_data, min_x=min_data)
