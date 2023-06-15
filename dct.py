import numpy as np 
import pandas as pd 
import graphlearning as gl
import cv2
from matplotlib import pyplot as plt
import sys
import os
from scipy.fftpack import dct
from scipy.fftpack import idct

path = sys.argv[1]

if os.path.exists(path):
    print ("Folder exist")

else:
    print("Folder doesn't exist")

images_file = [path + x for x in os.listdir(path) if "png" in x]
for img in images_file:
    image = img

def Load_image(image):
    global img
    img = plt.imread(image)
    plt.figure(figsize=(15,15))
    plt.imshow(img,cmap='gray')
    plt.show()
    print('Data type: '+str(img.dtype))
    print('Pixel intensity set: (%d,%d)'%(img.min(),img.max()))
    print(img.shape)
Load_image(image)

def dct2(f):
    return dct(dct(f, axis=0, norm='ortho' ),axis=1, norm='ortho')

def idct2(f):
    return idct(idct(f, axis=0 , norm='ortho'), axis=1 , norm='ortho')


def patterns():
    m = 0
    dct_basis = np.zeros((64,64))
    for i in range(8):
        for j in range(8):
            n = np.zeros((8,8))
            n[i,j]=1
            dct_basis[m,:] = idct2(n).flatten()
            m+=1

    dct_basis = dct_basis/np.max(np.absolute(dct_basis))
    gl.utils.image_grid(dct_basis,n_rows=8,n_cols=8)
patterns()

def liner_pattern():
    block = 8
    global img_dct
    img_dct = np.zeros_like(img)
    for i in range(0,img.shape[0],block):
        for j in range(0,img.shape[1],block):
            img_dct[i:(i+block),j:(j+block)] = dct2(img[i:(i+block),j:(j+block)])

    plt.figure(figsize=(10,10))
    plt.imshow(img_dct,cmap='gray',vmin=0,vmax=np.max(img_dct)*0.01)
    plt.show()
liner_pattern()

def reconstruct():
    global img_threshold, img_comp
    threshold = 0.1
    img_threshold = img_dct * (np.absolute(img_dct) > threshold*np.max(np.absolute(img_dct)))
    block = 8
    img_comp = np.zeros_like(img)
    for i in range(0,img.shape[0],block):
        for j in range(0,img.shape[1],block):
            img_comp[i:(i+block),j:(j+block)] = idct2(img_threshold[i:(i+block),j:(j+block)])
    
    plt.figure(figsize=(10,10))
    plt.imshow(img_comp,cmap='gray')
    plt.imshow(np.hstack((img,img_comp, img-img_comp+0.5)), cmap='gray', vmin=0, vmax=1)
    plt.show()

reconstruct()

def compress_ratio():
    fraction = np.sum(img_threshold != 0.0)/img.size
    print('Compression ratio: %.1f:1'%(1/fraction))
    print("remaining %.2f%% of DCT coefficients"%(100*fraction))

compress_ratio()

def PSNR():
    MSE = np.sum((img-img_comp)**2)/img.size
    PSNR = 10*np.log10(np.max(img)**2/MSE)
    print('PSNR: %.2f dB'%PSNR)

PSNR()