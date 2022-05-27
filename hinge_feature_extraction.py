# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 11:36:02 2020

@author: swati
"""
import imutils
from PIL import Image, ImageEnhance
import cv2
import matplotlib.pyplot as plt
import numpy as np


N_ANGLE_BINS = 40
BIN_SIZE = 360 // N_ANGLE_BINS
LEG_LENGTH = 25

class Hinge():
    def __init__(self):
        self.sharpness_factor = 10
        self.bordersize = 3
        self.show_images = False
        self.is_binary = False
        
    def preprocess_binary_image(self, img_file, sharpness_factor = 10, bordersize = 3):
        im = Image.open(img_file)
        
        enhancer = ImageEnhance.Sharpness(im)
        im_s_1 = enhancer.enhance(sharpness_factor)
        # plt.imshow(im_s_1, cmap='gray')
        image = 255 - np.array(im_s_1)
        bw_image = cv2.copyMakeBorder(
            image,
            top=bordersize,
            bottom=bordersize,
            left=bordersize,
            right=bordersize,
            borderType=cv2.BORDER_CONSTANT,
            value=[255]
        )
        # plt.imshow(bw_image, cmap='gray')
        return bw_image, image
        
    def preprocess_image(self, img_file, sharpness_factor = 10, bordersize = 3):
        im = Image.open(img_file)
        
        enhancer = ImageEnhance.Sharpness(im)
        im_s_1 = enhancer.enhance(sharpness_factor)
        # plt.imshow(im_s_1, cmap='gray')
        
        (width, height) = (im.width * 2, im.height * 2)
        im_s_1 = im_s_1.resize((width, height))
        if self.show_images: plt.imshow(im_s_1, cmap='gray')
        image = np.array(im_s_1)
        image = cv2.copyMakeBorder(
            image,
            top=bordersize,
            bottom=bordersize,
            left=bordersize,
            right=bordersize,
            borderType=cv2.BORDER_CONSTANT,
            value=[255,255,255]
        )
        orig_image = image.copy()
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.GaussianBlur(image,(3,3),0)
        if self.show_images: plt.imshow(image, cmap='gray')
        (thresh, bw_image) = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        if self.show_images: plt.imshow(bw_image, cmap='gray')
        return bw_image, orig_image
    
    def get_contour_pixels(self, bw_image):
        contours, _= cv2.findContours(
            bw_image, cv2.RETR_TREE, 
            cv2.CHAIN_APPROX_NONE
            ) 
        # contours = imutils.grab_contours(contours)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[1:]
        
        img2 = bw_image.copy()[:,:,np.newaxis]
        img2 = np.concatenate([img2, img2, img2], axis = 2)
        
        if self.show_images:
            for cnt in contours : 
                cv2.drawContours(img2, [cnt], 0, (255, 0, 0), 1)  
                
            plt.imshow(img2, cmap='gray')
        return contours
    
    def get_hinge_features(self, img_file):
        if self.is_binary:
            bw_image, _ = self.preprocess_binary_image(img_file, self.sharpness_factor, self.bordersize)
        else:
            bw_image, _ = self.preprocess_image(img_file, self.sharpness_factor, self.bordersize)
        contours = self.get_contour_pixels(bw_image)
        
        hist = np.zeros((N_ANGLE_BINS, N_ANGLE_BINS))
            
        # print([len(cnt) for cnt in contours])
        for cnt in contours:
            n_pixels = len(cnt)
            if n_pixels <= LEG_LENGTH:
                continue
            
            points = np.array([point[0] for point in cnt])
            xs, ys = points[:, 0], points[:, 1]
            point_1s = np.array([cnt[(i + LEG_LENGTH) % n_pixels][0] for i in range(n_pixels)])
            point_2s = np.array([cnt[(i - LEG_LENGTH) % n_pixels][0] for i in range(n_pixels)])
            x1s, y1s = point_1s[:, 0], point_1s[:, 1]
            x2s, y2s = point_2s[:, 0], point_2s[:, 1]
            
            phi_1s = np.degrees(np.arctan2(y1s - ys, x1s - xs) + np.pi)
            phi_2s = np.degrees(np.arctan2(y2s - ys, x2s - xs) + np.pi)
            
            indices = np.where(phi_2s > phi_1s)[0]
            
            for i in indices:
                phi1 = int(phi_1s[i] // BIN_SIZE) % N_ANGLE_BINS
                phi2 = int(phi_2s[i] // BIN_SIZE) % N_ANGLE_BINS
                hist[phi1, phi2] += 1
                
        normalised_hist = hist / np.sum(hist)
        feature_vector = normalised_hist[np.triu_indices_from(normalised_hist, k = 1)]
        
        return feature_vector

if __name__ == '__name__':
    import argparse
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--img_file", type=str, default=r"IMAGE FILE LOCATION")
    parser.add_argument("--sharpness_factor", type=int, default=10)
    parser.add_argument("--bordersize", type=int, default=3)
    parser.add_argument("--show_images", type=bool, default=False)
    parser.add_argument("--is_binary", type=bool, default=True)
    opt = parser.parse_args()
    
    hinge = Hinge(opt)
    feature_vector = hinge.get_hinge_features((opt.img_file))
    
    print ("shape of feature vector: {}".format(feature_vector.shape))




