import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import seaborn as sns
from skimage.feature import hog, local_binary_pattern
from sklearn.svm import LinearSVC


lables = ["Males", "Females"]


def read_data(filename):
    data = []
    currentDirectory = os.getcwd()
    # for each class we read it's data
    for lable in lables:
        directory = filename+"\\" + lable+"\\"+lable+"\\"
        path = os.path.join(currentDirectory, directory)
        class_lable = lables.index(lable)
        print(str(class_lable) + "start ..")
        for imagename in os.listdir(path):
            try:
                img = cv2.imread(os.path.join(path, imagename), cv2.IMREAD_GRAYSCALE)
                # we might need to reshape image here
                data.append([img, class_lable])
            except Exception as e:
                print(e)

    return np.array(data)

###########################################
# def hist(lbp):
#     n_bins = int(lbp.max() + 1)
#     return plt.hist(lbp.ravel(), density=True, bins=n_bins, range=(0, n_bins),
#                     facecolor='0.5')
def hist( lbp):
    n_bins = int(lbp.max() + 1)
    hist, _ = np.histogram(lbp, density=True, bins=n_bins, range=(0,n_bins))
    return hist
  
##########################################

def resized_images(data):
    out=[]
    for idx, img in enumerate(data):

        width = 64
        height = 128
        dim = (width, height)

        # data[idx] = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # resize image
        resized = cv2.resize(data[idx], dim, interpolation=cv2.INTER_AREA)
        out.append(resized)
    return out
