import tensorflow as tf
import numpy as np

xy = np.loadtxt('data-04-zoo.csv', delimiter=',', dtype=np.float32)
x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]

nb_classes = 7

X = tf.placeholder(tf.float32, [None, 16])
Y = tf.placeholder(tf.int32, [None, 1])  # 0~6
# tf.one_hot을 하면 rank + 1이 된다. 즉 1차원 증가된 결과를 전달한다. 그래서 reshape을 해야 한다.
Y_one_hot = tf.one_hot(Y, nb_classes)
# -1은 everything 즉 전체를 이야기한다.
Y_one_hot = tf.reshape(Y_one_hot, [-1, nb_classes])

W = tf.Variable(tf.random_normal([16, nb_classes]), name='weight')
b = tf.Variable(tf.random_normal([nb_classes]), name='bias')

logits = tf.matmul(X, W) + b
hypothesis = tf.nn.softmax(logits)

cost_i = tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y_one_hot)
cost = tf.reduce_mean(cost_i)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

prediction = tf.argmax(hypothesis, 1) # hypothesis로 나온 값중 가장 큰 값의 index를 넘긴다. 즉 0~6사이의 값중 하나로 만든다.
correct_prediction = tf.equal(prediction, tf.argmax(Y_one_hot, 1))
accurary = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) # reduce_mean은 평균을 구한다.

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(2000):
        sess.run(optimizer, feed_dict={X: x_data, Y: y_data})
        if step % 100 == 0:
            loss, acc = sess.run([cost, accurary], feed_dict={
                                 X: x_data, Y: y_data})
            print("Step: {:5}\t, Loss: {:.3f}\tAcc: {:.2f}".format(
                step, loss, acc))

    pred = sess.run(prediction, feed_dict={X: x_data})
    for p, y in zip(pred, y_data.flatten()):
        print("[{}] Prediction: {}, True Y: {}".format(p == int(y), p, int(y)))
