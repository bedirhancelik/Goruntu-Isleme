# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np

myList=[2,4,3,40,5,6,3,3,2,1]

#ortalama
toplam=0
sayac=0
for i in myList:
    sayac=sayac+1
    toplam=toplam+i
    
mean=toplam/sayac
#print(mean)

#varyans
toplam=0
sayac=0
for i in myList:
    sayac=sayac+1
    toplam = toplam + (i - mean)*(i - mean)
    
varyans=toplam/(sayac-1)
#print(varyans)

#fonksiyon hali ort ve varyans
def myFunction(myList=[2,4,3,40,5,6,3,3,2,1]):
    
    toplam=0
    sayac=0
    for i in myList:
        sayac=sayac+1
        toplam=toplam+i
    
    mean=toplam/sayac
    
    toplam=0
    sayac=0
    for i in myList:
        sayac=sayac+1
        toplam = toplam + (i - mean)*(i - mean)
    
    varyans=toplam/(sayac-1)
    
    return mean,varyans

print(myFunction())



myHistogram={} #dictionary hash yapısı
for i in myList:
    if i in myHistogram.keys():
        myHistogram[i] = myHistogram[i] + 1
    
    else:
        myHistogram[i] = 1
        
print(myHistogram)


#histogram grafiği
    
def myFunction2(im=plt.imread("panda.jpg")):
    print(im.shape,im.ndim)
    
    myHist={}
    m,n,p = im.shape
    for i in range (m):
        for j in range (n):
            if(im[i,j,0] in myHist.keys()): 
                myHist[im[i,j,0]]= myHist[im[i,j,0]] + 1
            else:
                myHist[im[i,j,0]] = 1
                    
    return myHist

myHistogram=myFunction2()

x=[] #color pigment
y=[] #pixel?????*
for key in myHistogram.keys():
    x.append(key)
    y.append(myHistogram[key])
print(x,"\n",y)

plt.bar(x,y)
plt.show
