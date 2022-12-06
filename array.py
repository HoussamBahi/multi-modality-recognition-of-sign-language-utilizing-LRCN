import numpy as nmp
X = nmp.array( [ [ 1, 6, 7,3],
                 [ 5, 9, 2,3],
                 [ 3, 8, 4,3],
                  ])
Y = nmp.array( [ [ 1, 6, 7,3],
                 [ 5, 9, 2,3],
                 [ 3, 8, 4,3],
                  ])
print(X.shape)
#z=nmp.concatenate(X[],Y[])
z=X+Y
print("z shape",z.shape)
print("z",z)
# print("x -1",X[:-2])
# y=X[:1],X[:2]
# print("x -1",y)

#
#
# Y= nmp.array([[ 1, 6, 7,3]])
# print("y",Y, Y.shape)
# xx=[]
# for i in range(5):
#     xx.append(Y)
# print(xx)
# xx = nmp.array(xx, dtype=nmp.float32)
# print(xx.shape)
# yy=[]
# for i in range(3):
#     yy.append(xx)
#
# yy = nmp.array(yy, dtype=nmp.float32)
# print(yy.shape)
