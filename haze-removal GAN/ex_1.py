import os
import cv2
import numpy as np
import pickle
import gzip


hazed_images_name = sorted(os.listdir('smaller hazed'))

for i in range(len(hazed_images_name)) :

    img = cv2.imread('smaller hazed/'+hazed_images_name[i],cv2.IMREAD_COLOR)

    if i == 0 :
        a = [img]
    else :
        a = np.append(a, np.array([img]),axis=0)
    


a = 2.0*a/255.0 - 1.0

with gzip.open('hazed_images.pickle.gzip','wb') as f :
    pickle.dump(a,f)




origin_images_name = sorted(os.listdir('smaller origin'))
for i in range(len(origin_images_name)) :

    img = cv2.imread('smaller origin/'+origin_images_name[i],cv2.IMREAD_COLOR)

    if i == 0 :
        a = [img]
    else :
        a = np.append(a, np.array([img]),axis=0)
    
a = 2.0*a/255.0 - 1.0
with gzip.open('origin_images.pickle.gzip','wb') as f :
    pickle.dump(a,f)




