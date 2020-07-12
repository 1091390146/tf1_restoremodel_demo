
import tensorflow.compat.v1 as tf
from cnn_utils import *
SAVE_FILE = "./model/model"
SECOND_TRAING = False
np.random.seed(1)

X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()


X_train = X_train_orig/255.
X_test = X_test_orig/255.
Y_train = convert_to_one_hot(Y_train_orig, 6).T
Y_test = convert_to_one_hot(Y_test_orig, 6).T
print ("number of training examples = " + str(X_train.shape[0]))
print ("number of test examples = " + str(X_test.shape[0]))
print ("X_train shape: " + str(X_train.shape))
print ("Y_train shape: " + str(Y_train.shape))
print ("X_test shape: " + str(X_test.shape))
print ("Y_test shape: " + str(Y_test.shape))
conv_layers = {}

def create_placeholders(n_H0, n_W0, n_C0, n_y):
    X = tf.placeholder(tf.float32, shape=(None, n_H0, n_W0, n_C0), name="X")
    Y = tf.placeholder(tf.float32, shape=(None, n_y), name="Y")
    return X, Y

def initialize_parameters():
    tf.set_random_seed(1)

    W1 = tf.get_variable('W1', [4, 4, 3, 8], initializer=tf.contrib.layers.xavier_initializer(seed=0))
    W2 = tf.get_variable('W2', [2, 2, 8, 16], initializer=tf.contrib.layers.xavier_initializer(seed=0))

    parameters = {"W1": W1,
                  "W2": W2}
    return parameters

def forward_propagation(X, parameters):
    W1 = parameters['W1']

    W2 = parameters['W2']

    Z1 = tf.nn.conv2d(X, W1, strides= [1,1,1,1], padding= 'SAME')

    A1 = tf.nn.relu(Z1)

    P1 = tf.nn.max_pool(A1, ksize= [1,8,8,1], strides= [1,8,8,1], padding= 'SAME')

    Z2 = tf.nn.conv2d(P1, W2, strides= [1,1,1,1], padding= 'SAME')

    A2 = tf.nn.relu(Z2)

    P2 = tf.nn.max_pool(A2, ksize= [1,4,4,1], strides= [1,4,4,1], padding= 'SAME')

    P2 = tf.layers.flatten(P2)

    Z3 = tf.layers.dense(inputs=P2 , units=6, name='Z3')

    #将Z3添加到集合中
    tf.add_to_collection('output', Z3)

    return Z3


def compute_cost(Z3, Y):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits= Z3, labels=Y),name="cost")
    return cost

def model(X_train, Y_train, X_test, Y_test, learning_rate = 0.009, num_epochs = 100,
        minibatch_size = 64, print_cost = True):
    tf.set_random_seed(1)
    seed = 3
    (m, n_H0, n_W0, n_C0) = X_train.shape
    n_y = Y_train.shape[1]
    costs = []

    X, Y = create_placeholders(n_H0, n_W0, n_C0, n_y)
    parameters = initialize_parameters()
    Z3 = forward_propagation(X, parameters)

    cost = compute_cost(Z3, Y)
    optimizer = tf.train.AdamOptimizer(learning_rate= learning_rate).minimize(cost,name="optimizer")


    with tf.Session() as sess:
        saver = tf.train.Saver()

        #通过模型进行训练，注意前面已经构建了图，而restore函数是给前面的变量进行赋值。
        if SECOND_TRAING:
            init = tf.global_variables_initializer()
            sess.run(init)
            #注意可以在restore之前进行初始化操作，之后不可以
            saver.restore(sess, SAVE_FILE)
        else:
            init = tf.global_variables_initializer()
            sess.run(init)

        for epoch in range(num_epochs):
            minibatch_cost = 0.
            num_minibatches = int(m / minibatch_size)
            seed = seed + 1
            minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)

            for minibatch in minibatches:
                (minibatch_X, minibatch_Y) = minibatch
                _, temp_cost = sess.run([optimizer, cost], feed_dict={X:minibatch_X, Y:minibatch_Y})
                minibatch_cost += temp_cost / num_minibatches


            if print_cost == True and epoch % 5 == 0:
                print("Cost after epoch %i: %f" %(epoch, minibatch_cost)    )
                savepath = saver.save(sess,SAVE_FILE)
                print("Model saved in file: %s" %savepath)
            if print_cost == True and epoch %1 == 0:
                costs.append(minibatch_cost)

        # plot the cost
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

        # Calculate the correct predictions
        predict_op = tf.argmax(Z3, 1)
        correct_prediction = tf.equal(predict_op, tf.argmax(Y, 1))

        # Calculate accuracy on the test set
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print(accuracy)
        train_accuracy = accuracy.eval({X: X_train, Y: Y_train})
        test_accuracy = accuracy.eval({X: X_test, Y: Y_test})
        print("Train Accuracy:", train_accuracy)
        print("Test Accuracy:", test_accuracy)

        return train_accuracy, test_accuracy, parameters

#不对图进行重构，利用加载的图进行训练
def train_by_model(X_train, Y_train, X_test, Y_test, learning_rate = 0.009, num_epochs = 100,
        minibatch_size = 64, print_cost = True):

    tf.set_random_seed(1)
    seed = 3
    (m, n_H0, n_W0, n_C0) = X_train.shape
    n_y = Y_train.shape[1]
    costs = []

    with tf.Session() as sess:
        #这里直接将模型的图导入并构建为default
        saver = tf.train.import_meta_graph('./model/model.meta')
        saver.restore(sess, SAVE_FILE)
        graph = tf.get_default_graph()
        X = graph.get_tensor_by_name('X:0')
        Y = graph.get_tensor_by_name('Y:0')
        cost = graph.get_tensor_by_name('cost:0')
        optimizer = graph.get_operation_by_name('optimizer')

        Z3 = tf.get_collection('output')[0]

        for epoch in range(num_epochs):
            minibatch_cost = 0.
            num_minibatches = int(m / minibatch_size)
            seed = seed + 1
            minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)

            for minibatch in minibatches:
                (minibatch_X, minibatch_Y) = minibatch
                _, temp_cost = sess.run([optimizer, cost], feed_dict={X:minibatch_X, Y:minibatch_Y})
                minibatch_cost += temp_cost / num_minibatches


            if print_cost == True and epoch % 5 == 0:
                print("Cost after epoch %i: %f" %(epoch, minibatch_cost))
                savepath = saver.save(sess,SAVE_FILE)
                print("Model saved in file: %s" %savepath)
            if print_cost == True and epoch %1 == 0:
                costs.append(minibatch_cost)
        # plot the cost
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

        # Calculate the correct predictions
        predict_op = tf.argmax(Z3, 1)
        correct_prediction = tf.equal(predict_op, tf.argmax(Y, 1))

        # Calculate accuracy on the test set
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        train_accuracy = accuracy.eval({X: X_train, Y: Y_train})
        test_accuracy = accuracy.eval({X: X_test, Y: Y_test})
        print("Train Accuracy:", train_accuracy)
        print("Test Accuracy:", test_accuracy)

    return

def accuracy(X_train, Y_train, X_test, Y_test):
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph('./model/model.meta')
        saver.restore(sess, SAVE_FILE)
        graph = tf.get_default_graph()
        X = graph.get_tensor_by_name('X:0')
        Y = graph.get_tensor_by_name('Y:0')

        #get_collection返回的是一个列表，这里取第一个
        Z3 = tf.get_collection('output')[0]

        # Calculate the correct predictions
        predict_op = tf.argmax(Z3, 1)
        correct_prediction = tf.equal(predict_op, tf.argmax(Y, 1))

        # Calculate accuracy on the test set
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        #注意这里的eval函数还是要运行session
        train_accuracy = accuracy.eval({X: X_train, Y: Y_train})
        test_accuracy = accuracy.eval({X: X_test, Y: Y_test})
        print("Train Accuracy:", train_accuracy)
        print("Test Accuracy:", test_accuracy)

    return

#_, _, parameters = model(X_train, Y_train, X_test, Y_test)
#train_by_model(X_train, Y_train, X_test, Y_test)
accuracy(X_train, Y_train, X_test, Y_test)