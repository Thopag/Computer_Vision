import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv 

def plot_contours(img , s= 5):
   
   if not np.isin(img, [0, 1]).all():
       print(" image is not binary (should do something about it later)")
   
   SE= cv.getStructuringElement(cv.MORPH_RECT,(s,s))
   dilated_img = cv.dilate(img,SE)
   
   img_contour = dilated_img - img
   
   plt.imshow(img_contour,cmap="gray")
   return img_contour