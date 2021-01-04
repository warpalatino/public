import numpy as np 

print('----> starts here')

#create array of ten zeros
my_array = np.zeros(10)
print(my_array)
print('---------')

#create array of ten ones
my_array = np.ones(10)
print(my_array)
print('---------')

#create array of ten fives
my_array = np.ones(10) * 5
print(my_array)
print('---------')

#create array of integers from 10 to 50
my_array = np.arange(10, 51)
print(my_array)
print('---------')

#create array of even integers from 10 to 50
my_array = np.arange(10, 51, 2) 
print(my_array)
print('---------')

#create 3x3 matrix with values 0 to 8
my_matrix = np.arange(9).reshape (3,3) 
print(my_matrix)
print('---------')

#create 3x3 identity matrix 
my_matrix = np.eye(3)
print(my_matrix)
print('---------')

#generate random numbers between 0 and 1
my_matrix = np.random.rand(1)
print(my_matrix)
print('---------')

#generate array of random numbers sampled from a normal distribution
my_array = np.random.randn(25)
print(my_array)
print('---------')

#generate array of random numbers sampled from a normal distribution
my_array = np.arange(1,101)/100
my_matrix = my_array.reshape(10,10)
print(my_matrix)
print('---------')

#create array of linearly spaced points between 0 and 1
my_array = np.linspace(0, 1, 20) 
print(my_array)
print('---------')



# *****************
#scalars = numbers
print('scalar')
s = 5
# print(type(s))
print('----------')

# vectors
print('vector')
v = np.array([5,-2,4])
# print(type(v))
print(v.shape)
print('----------')

# matrix = array containing vectors
print('matrix')
m = np.array([[5,12,6],[-3,0,14]])
# print(type(m))
print(m.shape)
print('----------')



# creating a tensor
# ------
print('source matrix m1 for tensor')
m1 = np.array([[5,10,6],[-3,1,11]])
print(m1)
print('source matrix m2 for tensor')
m2 = np.array([[9,8,7],[1,3,-5]])
print(m2)
print('tensor')
tensor = np.array([m1,m2])
print(tensor)
print(tensor.shape)
print('----------')
# manual tensor
tensor2 = np.array([[[ 5, 12,  6], [-3,  0, 14]], [[ 9,  8,  7], [ 1,  3, -5]]])


# math ops
# ------
# addition for vectors
print('addition for vectors')
v1 = np.array([1,2,3,4,5])
v2 = np.array([7,8,9,10,11])
v3 = v1 + v2
print(v3)
print(v3.shape)
print('----------')
# -- addition, matrix keeps the shape
print('addition for matrices')
m3 = m1 + m2
print(m3)
print(m3.shape)
print('----------')
# -- subtraction, matrix keeps the shape
print('subtraction for matrices')
m4 = m1 - m2
print(m4)
print(m3.shape)
print('----------')
# -- dot product, multiplication
print('dot product - vectors')
x = np.array([2,8,-4])
y = np.array([1,-7,3])
m5 = np.dot(x,y)
print(m5)
#output is always a single number
print('dot product - matrix')
z = np.array([[5,10,6],[-3,1,11]])
w = np.array([[5,10],[-3,1],[-3,1]])
m5 = np.dot(z,w)
# matrix (m * n) multiplied by matrix (n * k) = matrix (m*k)
print(m5)
print(z.shape)
print(w.shape)
print(m5.shape)
print('----------')




# transposing 
# https://lihan.me/2018/01/numpy-reshape-and-transpose/
# ------
print('transposing')
print(m3)
m6 = m3.T
print(m6)
print('----------')
print('reshaping')
print(m3)
m7 = m3.reshape(3,2)
print(m7)
print('----------')
# *****************




#---
#use this starting matrix for indexing and selection
new_matrix = np.arange(1,26).reshape(5,5)
print(new_matrix)
print('---------')
#---

#grab a slice removing first two rows and first column
test_matrix = new_matrix[2:,1:]
print(test_matrix)
print('---------')

#grab the value 20 from the new_matrix
test_matrix = new_matrix[3,4]
print(test_matrix)
print('---------')

#grab the the second column in matrix
test_matrix = new_matrix[:3,1:2]
print(test_matrix)
print('---------')

#grab last two rows in matrix
test_matrix = new_matrix[3:, :]
print(test_matrix)
print('---------')

#sum all values in matrix
test_matrix = new_matrix.sum()
print(test_matrix)
print('---------')

#sum all columns in matrix
test_matrix = new_matrix.sum(axis=0)
print(test_matrix)
print('---------')

#st dev for all values in matrix
test_matrix = new_matrix.std()
print(test_matrix)
print('---------')
#or...
test_matrix = np.std(new_matrix)
print(test_matrix)
print('---------')

#anchor the generation of random numbers
seed = np.random.seed(77)
test_array_seed = np.random.rand(1)

print(test_array_seed)
print('---------')