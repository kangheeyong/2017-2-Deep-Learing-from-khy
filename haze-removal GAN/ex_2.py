import numpy as np
import pickle
import gzip
import os
import cv2


with gzip.open('hazed_images.pickle.gzip','rb') as f :
    hazed_images = pickle.load(f)


with gzip.open('origin_images.pickle.gzip','rb') as f :
    origin_images = pickle.load(f)

print(origin_images.shape)
print(hazed_images.shape[0])







