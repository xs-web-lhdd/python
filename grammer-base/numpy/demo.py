# import numpy
import numpy as np

student = np.dtype([('name', 'S20'), ('age', 'i8'), ('marks', 'f4')])

# a = np.array([('ab', 21, 50)], dtype=student)
# print(a)


# data = [
#     [1, 2, 3, 4],
#     [5, 6, 7, 8]
# ]
# v = np.ndarray.ndim(data)


# a = np.arange(24)
# print(a.ndim)
# b = a.reshape(2, 4, 3)
# print(b.ndim)


# a = np.array([[1, 2, 3], [1, 2, 3]])
# print(a)
# print(a.shape)


# a = np.array([[1, 2, 3], [4, 5, 6]])
# b = a.reshape(3, 2)
# a.shape = (3, 2)
# print('a', a, 'b', b)


# x = [1, 2, 3]
# a = np.asarray(x)
# print(a)

# s = b'Hello World'
# a = np.frombuffer(s, dtype='S1')
#
# print(a)
# print('len: ', len(a))


# x = np.arange(32).reshape((8, 4))
# print(x)
#
# print('--------------二维数组下标对应的行------------')
# y = x[[4, 3, 3, 2]]
# print(y)


# x = np.arange(32)
# print(x)
#
# print('--------------一维数组下标对应的行------------')
# y = x[4, 3, 3, 2]
# print(y)

x = np.arange(32).reshape(8, 4)
# print(x[[(1, 0), (1, 3), (1, 1), (1, 2)]])


print(np.ix_([1,5,7,2],[0,3,1,2]))

