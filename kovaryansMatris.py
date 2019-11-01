# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 13:54:16 2019

@author: Bedirhan Çelik
"""
import numpy as np

#işaret dili tanıma ? sınava hazırlık olarak

boy=[180,170,170,175,181,175,177,185,179,160]
kilo=[95,70,60,79,60,63,83,80,75,50]

X = np.stack((boy,kilo),axis=0) #iki dizinin üst üste eklenmiş hali, stacklemek..
sigma=np.cov(X)  #ortalamadan olan farklarının çarpımı her bir değerin. (boy-ortboy) * (kilo - ortkilo)

#print("boy ve kilonun kovaryansı\n",sigma)

#artık elimizdeki verinin kovaryansını almayı biliyoruz. bu bilgiyi pdf değeri gibi bayes teorieminde veri sınıflandırmak için kullanıcaz.

#☺ yukarıdakilerin fonksiyonlaştırılması

def generateData():    
    boy=[180,170,170,175,181,175,177,185,179,160]
    kilo=[95,70,60,79,60,63,83,80,75,50]
    X = np.stack((boy,kilo),axis=0)
    return X

def getCovMatrix(X):
    sigma = np.cov(X)
    return sigma

data = generateData()
getCovMatrix(data)
print(data,"\n\n\n",getCovMatrix(data))

#multivariate normal distribution fonksiyonu yazıcaz.

def multivariate_normal(x, d, mean, covariance):
    """pdf of the multivariate normal distribution."""
    x_m = x - mean
    return (1. / (np.sqrt((2 * np.pi)**d * np.linalg.det(covariance))) * 
            np.exp(-(np.linalg.solve(covariance, x_m).T.dot(x_m)) / 2))


#print(np.mean(data[0,:]),np.mean(data[1,:]))
sample = [165,75]
d=2 
#örnek kişi bu kişinin data setinde olma olasıgılı multivariateden cıkan deger
mean = np.array([np.mean(data[0,:]),np.mean(data[1,:])])
covariance = getCovMatrix(data)

print(multivariate_normal(sample,d,mean,covariance))



for i in range(10):
    v=167+i
    x_1=[v,72]
    s=multivariate_normal(x_1,d,mean,covariance)
    print(v," ",s)
#X_test = np.stack((boy,boy),axis=0) #(boy-ortboy)**2
#print("boy dizisinin kovaryansı\n",np.cov(X_test))