import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import seaborn as sns
from skimage.feature import hog, local_binary_pattern
from sklearn.svm import LinearSVC, SVC
from sklearn.model_selection import GridSearchCV

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
                img = cv2.imread(os.path.join(path, imagename))
                greyImg= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                # we might need to reshape image here
                data.append([greyImg, class_lable])
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

def resized_images(data, width, height):
    out=[]
    for idx, img in enumerate(data):

        dim = (width, height)

        # data[idx] = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # resize image
        resized = cv2.resize(data[idx], dim, interpolation=cv2.INTER_AREA)
        out.append(resized)
    return out


##########################################
def LBPfeatures(images, radius, pointNum):
    hist_LBP = []
    for img in images:
        # print(img)
        if type(pointNum) == int and type(radius) == int:
            lbp = local_binary_pattern(img, int(pointNum), int(radius), method="uniform")
            (hist, _) = np.histogram(lbp.ravel(), bins=range(0, pointNum + 3), range=(0, pointNum + 2))
            hist_LBP.append(hist)
    return hist_LBP

    #########################################

def GridSearch_tuning(X_train, y_train):
    param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001, 0.00001, 10]}
    # Make grid search classifier
    clf_grid = GridSearchCV(SVC(kernel="rbf"), param_grid, verbose=1)
    # Train the classifier
    clf_grid.fit(X_train, y_train)
    # extract the parameters
    best_params = clf_grid.best_params_
    # return the parameters
    return best_params