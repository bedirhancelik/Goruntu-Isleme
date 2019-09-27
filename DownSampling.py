import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#label sütununu kaldırmak

#df=pd.read_csv("mnist_train.csv")
#unlabeledData=df.iloc[:,1:]
#unlabeledData.to_csv("unlabeledMnistTrain.csv")


#train_data=np.loadtxt("unlabeledMnistTrain.csv",delimiter=",")
#print(train_data.ndim)
#print(train_data.shape)

##listeyi tek tek dolaşıp onları arttırmak    
def my_function(myList=[9,3,5,6,2,3,6]):
    for i in range(len(myList)):
        #print(i,myList[i])
        myList[i] = myList[i] + 1
    print(myList)
#my_function(["bir",2,3,4,5,6,7,8]) error bunu çözen yapı ndarray        
my_function()


myList1 = np.array(list([9,3,5,6,2,3,6])) 
print(myList1+1) ##çözdü

def myFunction(myArray=np.array(list([9,3,5,6,2,3,6]))):
    return myArray - 10
print(myFunction()) # yukardaki fonksiyonlaştırıldı


im1 = plt.imread("istanbul.jpg")
print(im1.ndim,im1.shape)

#resmin kırmızı pixellerini azalttık.

def myFunction2(im100,s=5):
    im1=im100
    m,n,p=im1.shape
    im2= np.zeros((m,n,3),dtype = int)
    for m in range(im1.shape[0]):
        for n in range(im1.shape[1]):
            im2[m,n,0] = im1[m,n,0] + s
            im2[m,n,1] = im1[m,n,0]
            im2[m,n,2] = im1[m,n,0]

    return im2
im3=myFunction(im1)
#im1[:,:,0] = im1[:,:,0] - 25
plt.imshow(im1)
plt.show()


#Down Sampling m ve n boyutlarını yarıya indirmek yani resmi küçültmek resmin 4 te 1 ini almaktır

def myFunction3(im500):
    k,l,p=im500.shape
    m=int(k/2)
    n=int(l/2)
    im600 = np.zeros((m,n), dtype = int)
    for m in range(m):
        for n in range(n):
            s0 = (im500[m*2,n*2,0] /3  + im500[m*2,n*2,1] / 3 + im500[m*2,n*2,2] / 3)
            #verdiği hata pixel degerlerinin türünden,(int) tasmasından dolayı
            
         #   s1 = (im500[m,n+1,0] + im500[m,n+1,1] + im500[m,n+1,2]) / 3
            
         #   s2 = (im500[m+1,n,0] + im500[m+1,n,1] + im500[m+1,n,2]) / 3
            
         #   s3 = (im500[m+1,n+1,0] + im500[m+1,n+1,1] + im500[m+1,n+1,2]) / 3
            
         #   s=(s0+s1+s2+s3) / 4
            
            im600[m,n] = int(s0)

    return im600

plt.imshow(myFunction3(im1),cmap="gray")
plt.show()

#rgb 3 boyutlu m n lik resmi m/2 n/2 geri döndüren fonksiyon yazınız.

def myFunction4(im500):
    m,n,p=im500.shape
    k=int(m/2)
    l=int(n/2)
    im600 = np.zeros((k,l,3), dtype = int)
    for m in range(k):
        for n in range(l):
         #   s0 = (im500[m*2,n*2,0] /3  + im500[m*2,n*2,1] / 3 + im500[m*2,n*2,2] / 3)
            #verdiği hata pixel degerlerinin türünden,(int) tasmasından dolayı
            
         #   s1 = (im500[m,n+1,0] + im500[m,n+1,1] + im500[m,n+1,2]) / 3
            
         #   s2 = (im500[m+1,n,0] + im500[m+1,n,1] + im500[m+1,n,2]) / 3
            
         #   s3 = (im500[m+1,n+1,0] + im500[m+1,n+1,1] + im500[m+1,n+1,2]) / 3
            
         #   s=(s0+s1+s2+s3) / 4
            
            im600[m,n,0] = im500[m * 2,n * 2,0]
            im600[m,n,1] = im500[m * 2,n * 2,1]
            im600[m,n,2] = im500[m * 2,n * 2,2]

    return im600


plt.imshow(myFunction4(im1))
plt.show()
