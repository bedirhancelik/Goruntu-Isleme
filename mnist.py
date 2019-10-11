#%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt

#image_size = 28 # width and length
#no_of_different_labels = 10 #  i.e. 0, 1, 2, 3, ..., 9
#image_pixels = image_size * image_size

#data okyuma
train_data = np.loadtxt("mnist_train.csv", 
                        delimiter=",")
test_data = np.loadtxt("mnist_test.csv", 
                       delimiter=",") 
print(train_data[10,0])



#resmi reshapeleme
im3 = train_data[10,:] #10.satırdaki resmi aldık.

im4 = im3[1:] #label i attık

im5 = im4.reshape(28,28) #2 boyutlu resme donusturduk

plt.imshow(im5,cmap="gray")
plt.show()

#train data da kac tane 3 oldugunu bulan bir fonksiyon yazınız?

m,n=train_data.shape #60000,785


def howMany(k):
    s=0
    for i in range(m):
        if(train_data[i,0]==k):
            s=s+1
    return s

print(howMany(3))

#butun sayılardan kac tane var?
for i in range(10):
    c=howMany(i)
    print(i,"  ",c)
    
    
    
    
#0 sayısının standart sapması ve varyansı nedir gray level ortalama??
    
m,n = train_data.shape

s,k,t,l=0,0,0,350
 #digit_class = train_data[i,0], # class bilgisi
 #top_left = train_data[i,1], # sol üst oiksel renk degeri
 #bottom_right = train_data[i,784] # sag alttaki piksel degeri
for i in range(m):
    if(train_data[i,0]==k):
        s = s + 1
        t = t + train_data[i,l+1]
mean_1 = t / s
print(mean_1) #mean

s,t=0,0
for i in range(m):
    if(train_data[i,0]==k):
        s = s + 1
        diff_1 = train_data[i,l+1]-mean_1
        t = t + diff_1 * diff_1
std_1 = np.sqrt(t / (s - 1))

print(mean_1,std_1) # mean ve standart sapma 

# s k t ve l ne?????

import math
def my_pdf1(x, mu=0.0, sigma=1.0):
    x = float(x - mu) / sigma
    return math.exp(-x*x/2.0) / math.sqrt(2.0*math.pi) / sigma
print(my_pdf1(10,1,3))


#def get_my_mean_and_std(k=0,l=350):
#........
#reism olusturup o resmin hangi sayı olduguu bulduran fonksiyon bu yazılan fonksiyonu butun classlara butun pixellere uygulucaz
#