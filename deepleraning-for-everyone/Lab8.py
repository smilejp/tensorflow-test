import tensorflow as tf
import numpy as np

t = np.array([0, 1, 2, 3, 4, 5, 6])

# pp.pprint(t)
print(t.ndim)
print(t.shape)
print(t[0], t[1], t[-1])
print(t[2:5], t[4:-1])
print(t[:2], t[3:])
print(t[:-1])

t2 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
print(t2)
print(t2.ndim)
print(t2.shape)

t3 = tf.constant([1, 2, 3, 4])
# tf.shape(t3).eval()

t4 = tf.constant([[[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], [
                 [13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]]]])
# rank = 4
# shape = (1,2,3,4)

# Axis
#

matrix1 = tf.constant([[1, 2], [3, 4]])
matrix2 = tf.constant([[1], [2]])
print('matrix1 shape:', matrix1.shape)
print('matrix2 shape:', matrix2.shape)
sess = tf.Session()
print(sess.run(tf.matmul(matrix1, matrix2)))

matrix3 = tf.constant([[3, 3]])
print('matrix3 shape:', matrix3.shape)
