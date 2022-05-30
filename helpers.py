import time
import cv2
import os
import numpy as np
from skimage.feature import hog, local_binary_pattern

from hinge_feature_extraction import * 
from cold_feature_extraction  import * 
from sklearn.svm import  SVC
from sklearn.model_selection import GridSearchCV
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler

labels = ["Males", "Females"]
# feature extraction parameters
opt = {
        'sharpness_factor': 10,
        'bordersize': 3,
        'show_images': False,
        'is_binary': False,
        'LBP_numPoints': 8,
        'LBP_radius':1,
        'LBP_method': 'uniform',
        'HOG_width': 64,
        'HOG_height': 128,
    }
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import pickle
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
    HOG_width = options['HOG_width']
    HOG_height = options['HOG_height']
    
    for img in (imgs):
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

#############################################

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
############################################
def save_model (model , model_name):
    filename = model_name+'.sav'
    pickle.dump(model, open(filename, 'wb'))
    
def load_model (model_name):
    filename = model_name+'.sav'
    loaded_model = pickle.load(open(filename, 'rb'))
    return loaded_model
###########################################
def model_pipeline_testing(filename ,model_name ):
    '''
        This function responsible for testing our models
        we have to svm models one for cold feature and one for hinge feature 
    '''
    
    result_file = open("results.txt" ,"w")
    time_file = open("time.txt" , "w")
    
    # loading trained models 
    model  = load_model(model_name)
    
    hinge_f_vector=[]
    cold_f_vector=[]
    Hog_f_vector =[]
    
    currentDirectory = os.getcwd()
    directory = filename+"\\"
    path =os.path.join(currentDirectory, directory)
    for imagename in tqdm(os.listdir(path)):
            try:
                img = cv2.imread(os.path.join(path, imagename), cv2.IMREAD_GRAYSCALE)
                # resize all images to the same size
                img = cv2.resize(img, (2000, 1800), interpolation=cv2.INTER_AREA)
                
                start = time.time()
                # getting all features
                hod_f , hinge_f , cold_f =extract_features([img], opt)
                #storing it in the feature vectors 
                end = time.time()
                
                # adding image features 
                Hog_f_vector.append(hod_f)
                hinge_f_vector.append(hinge_f)
                cold_f_vector.append(cold_f)
                
                time_file.write(str(end-start) +"\n")
                hinge_f_vector.append(hinge_f)
                cold_f_vector.append(cold_f)
                Hog_f_vector.append(hod_f)
            except Exception as e :
                    print (e)
                    
    scaler = MinMaxScaler()
    HOG_feature_scalad = scaler.fit_transform(Hog_f_vector)
    COLD_feature_scaled  = scaler.fit_transform(cold_f_vector)
    HINGE_feature_scaled = scaler.fit_transform(hinge_f_vector)

    all_features = np.concatenate((HOG_feature_scalad, HINGE_feature_scaled), axis=1)
    all_features = np.concatenate((all_features, COLD_feature_scaled), axis=1)

    
    # using trained model for hinge and cold 
    y_pred =model.predict(all_features)
    
    for i in y_pred:
        result_file.write(str(i) +"\n")
    result_file.close()
    time_file.close()