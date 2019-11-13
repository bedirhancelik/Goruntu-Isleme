import numpy as np
import matplotlib.pyplot as plt

im_1=plt.imread("panda.jpg")
plt.imshow(im_1)
plt.show()

my_histogram_R_G_B={} # R,G,B her biri için ayrı ayrı histogram
m,n,p=im_1.shape
for i in range(m):
    for j in range(n):
        s=(im_1[i,j,0]) # ,im_1[i,j,1],im_1[i,j,2]) # s=im_1[i,j,:], s cannot be Key
        if (0,s) in my_histogram_R_G_B.keys(): # because its type is np.ndar
            my_histogram_R_G_B[(0,s)]=my_histogram_R_G_B[(0,s)]+1
        else:
            my_histogram_R_G_B[(0,s)]=1
print(my_histogram_R_G_B) # 0'ın histogramı

m,n,p=im_1.shape
for i in range(m):
    for j in range(n):
        s=(im_1[i,j,1]) # ,im_1[i,j,1],im_1[i,j,2]) # s=im_1[i,j,:], s cannot be Key
        if (1,s) in my_histogram_R_G_B.keys(): # because its type is np.ndar
            my_histogram_R_G_B[(1,s)]=my_histogram_R_G_B[(1,s)]+1
        else:
            my_histogram_R_G_B[(1,s)]=1
            
print(my_histogram_R_G_B) #0 ve 1 in histgoramı


m,n,p=im_1.shape
for i in range(m):
    for j in range(n):
        s=(im_1[i,j,2]) # ,im_1[i,j,1],im_1[i,j,2]) # s=im_1[i,j,:], s cannot be Key
        if (2,s) in my_histogram_R_G_B.keys(): # because its type is np.ndar
            my_histogram_R_G_B[(2,s)]=my_histogram_R_G_B[(2,s)]+1
        else:
            my_histogram_R_G_B[(2,s)]=1
            
print(my_histogram_R_G_B) #o 1 ve 2 nın hıstogrmaı


my_histogram={} 

m,n,p=im_1.shape
for i in range(m):
    for j in range(n):
        s=(im_1[i,j,0],im_1[i,j,1],im_1[i,j,2]) # s=im_1[i,j,:], s cannot be Key in di
        if s in my_histogram.keys(): # because its type is np.ndarray
            my_histogram[s]=my_histogram[s]+1
        else:
            my_histogram[s]=1

print("|n üçlü histogram \n",my_histogram)








