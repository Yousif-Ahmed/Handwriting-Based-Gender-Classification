import cv2
import os
import numpy as np
from skimage.feature import hog, local_binary_pattern

from hinge_feature_extraction import * 
from cold_feature_extraction  import * 

labels = ["Males", "Females"]

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

def read_data(filename, windows=True):
    X = []
    Y = []
    currentDirectory = os.getcwd()
    # for each class we read it's data
    for label in labels:
        if windows:
            directory = filename+"\\" + label+"\\"+label+"\\"
        else:
            directory = f"%s/%s/%s/"%(filename, label, label)
        path = os.path.join(currentDirectory, directory)
        class_label = labels.index(label)
        print(str(class_label) + "start ..")
        for image_name in os.listdir(path):
            try:
                img = cv2.imread(os.path.join(path, image_name), cv2.IMREAD_GRAYSCALE)
                # resize all images to the same size
                img = cv2.resize(img, (2000, 1800), interpolation=cv2.INTER_AREA)
                X.append(img)
                Y.append(class_label)

            except Exception as e:
                print(e)

    return np.asarray(X), np.asarray(Y)

###########################################
def extract_features(imgs, options):
    HOG_feature = []
    HINGE_feature = []
    COLD_feature = []
    # LBP_feature = []
    
    # HINGE obj
    hinge = Hinge(options)
    # COLD obj
    cold = Cold(options)

    numPoints = options['LBP_numPoints']
    # radius = options['LBP_radius']
    # method = options['LBP_method']
    HOG_width = options['HOG_width']
    HOG_height = options['HOG_height']
    
    for img in imgs:
        # HOG Feature Extraction
        hog_img = cv2.resize(img, (HOG_width, HOG_height), interpolation=cv2.INTER_AREA)
        hog_feat, hog_image = hog(hog_img, orientations=9, pixels_per_cell=(16, 16),
                                       cells_per_block=(3, 3), visualize=True, multichannel=False)
        HOG_feature.append(hog_feat)
        
        # HINGE feature
        image = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        hinge_f = hinge.get_hinge_features(image)
        HINGE_feature.append(hinge_f)
        
        # COLD feature
        cold_f = cold.get_cold_features(image)
        COLD_feature.append(cold_f)
        
        # LBP
        # lbp = local_binary_pattern(img, numPoints, radius, method="uniform")
        # LBP_feature.append(lbp.flatten())
        
        
    
    return HOG_feature, HINGE_feature, COLD_feature



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
    dim = (width, height)

    for idx, img in enumerate(data):

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