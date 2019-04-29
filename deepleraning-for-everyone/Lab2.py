import tensorflow as tf

# 1. Build Graph
# 2. Run Graph
# 3. Update Grand and Return Value

# x_train = [1, 2, 3]
# y_train = [1, 2, 3]

# W = tf.Variable(tf.random_normal([1]), name='weight')  # 1 is shape
# # trainable Variable
# b = tf.Variable(tf.random_normal([1]), name='bias')

# hypothesis = x_train * W + b

# # cost
# cost = tf.reduce_mean(tf.square(hypothesis - y_train))

# # minimize
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
# train = optimizer.minimize(cost)

# sess = tf.Session()
# sess.run(tf.global_variables_initializer())

# for step in range(2001):
#     sess.run(train)
#     if step % 20 == 0:
#         print(step, sess.run(cost), sess.run(W), sess.run(b))

# using placeholder
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

W = tf.Variable(tf.random_normal([1]), name='weight')  # 1 is shape
# trainable Variable
b = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = X * W + b

# cost
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# minimize
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(2001):
    cost_val, W_val, b_val, _ = sess.run([cost, W, b, train], feed_dict={
                                         X: [1, 2, 3, 4, 5], Y: [2.1, 3.1, 4.1, 5.1, 6.1]})
    if step % 20 == 0:
        print(step, cost_val, W_val, b_val)

print(sess.run(hypothesis, feed_dict={X: [5]}))
print(sess.run(hypothesis, feed_dict={X: [2.5]}))
print(sess.run(hypothesis, feed_dict={X: [1.5, 3.5]}))
