import numpy as np
import matplotlib.pyplot as plt
import os
print(os.getcwd())

jpg_files=[f for f in os.listdir() if f.endswith(".jpg")]
print(jpg_files)

im_1 = plt.imread("panda.jpg")
print(type(im_1))
print(im_1.ndim)
print(im_1.shape)

print(im_1[:,100,:])

m,n,p=im_1.shape

# yeni resim için boyutları belirledik
new_image = np.zeros((m,n),dtype=float)

#resmi siyah beyaza dönüştürmek için rgb degerlerini tek bir değere dönüştürdük
for i in range(m):
    for j in range(n):
        s = (im_1[i,j,0] + im_1[i,j,1] + im_1[i,j,2]) / 3
        new_image[i,j]=s        

rotated_image = np.zeros((n,m),dtype=float)
#j ve i değişiyor
for i in range(m):
    for j in range(n):
        s = (im_1[i,j,0] + im_1[i,j,1] + im_1[i,j,2]) / 3
        rotated_image[j,i]=s        

plt.imshow(im_1)
plt.show()
plt.imshow(new_image,cmap="gray")
plt.show()
plt.imshow(rotated_image,cmap="gray")
plt.show()
#plt.imsave("test_2.png",new_image,cmap="gray")




#im_2=im_1[:,:,0]
#plt.imshow(im_2)
#plt.show()
#plt.imsave("Merhaba Görüntü İşleme",im_2)



