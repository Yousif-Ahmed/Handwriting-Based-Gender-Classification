import cv2
import os
from cv2 import HOUGH_MULTI_SCALE
import numpy as np
from skimage.feature import hog, local_binary_pattern
import csv
from hinge_feature_extraction import * 
from cold_feature_extraction  import * 
from sklearn.svm import  SVC
from sklearn.model_selection import GridSearchCV
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix, fbeta_score, accuracy_score
import seaborn as sns
from time import time
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pickle


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



def evaluate(results):
    """
    Visualization code to display results of various learners.
    
    inputs:
      - results: a dict containing different learners result ['train_time', 'acc_train', 'f_train', 'pred_time', 'acc_test', 'f_test']
    """
  
    # Create figure
    fig, ax = plt.subplots(2, 3, figsize = (15,9))

    # Constants
    bar_width = 0.3
    colors = ['#A00000','#00A0A0','#00A000']
    
    # Super loop to plot four panels of data
    for k, learner in enumerate(results.keys()):
        for j, metric in enumerate(['train_time', 'acc_train', 'f_train', 'pred_time', 'acc_test', 'f_test']):
            for i in np.arange(3):
                
                # Creative plot code
                ax[j//3, j%3].bar(k*bar_width, results[learner][metric], width = bar_width, color = colors[k])
                
    # Add unique y-labels
    ax[0, 0].set_ylabel("Time (in seconds)")
    ax[0, 1].set_ylabel("Accuracy Score")
    ax[0, 2].set_ylabel("F-score")
    ax[1, 0].set_ylabel("Time (in seconds)")
    ax[1, 1].set_ylabel("Accuracy Score")
    ax[1, 2].set_ylabel("F-score")
    
    # Add x-labels
    ax[1, 0].set_xlabel("Training set sizes")
    ax[1, 1].set_xlabel("Training set sizes")
    ax[1, 2].set_xlabel("Training set sizes")
    
    # Add titles
    ax[0, 0].set_title("Model Training")
    ax[0, 1].set_title("Accuracy Score on Training Subset")
    ax[0, 2].set_title("F-score on Training Subset")
    ax[1, 0].set_title("Model Predicting")
    ax[1, 1].set_title("Accuracy Score on Testing Set")
    ax[1, 2].set_title("F-score on Testing Set")
    
    
    # Set y-limits for score panels
    ax[0, 1].set_ylim((0, 1))
    ax[0, 2].set_ylim((0, 1))
    ax[1, 1].set_ylim((0, 1))
    ax[1, 2].set_ylim((0, 1))

    # Create patches for the legend
    patches = []
    for i, learner in enumerate(results.keys()):
        patches.append(mpatches.Patch(color = colors[i], label = learner))
    plt.legend(handles = patches, bbox_to_anchor = (-.80, 2.53), \
               loc = 'upper center', borderaxespad = 0., ncol = 3, fontsize = 'x-large')
    
    # Aesthetics
    plt.suptitle("Performance Metrics for Three Supervised Learning Models", fontsize = 16, y = 1.10)
    fig.tight_layout(pad=3.0)
    fig.show()    
####################################################

def train_predict(learner, X_train, y_train, X_test, y_test): 
    '''
    inputs:
       - learner: the learning algorithm to be trained and predicted on
       - X_train: features training set
       - y_train: income training set
       - X_test: features testing set
       - y_test: income testing set
    '''
    
    results = {}
    
    # Fit the learner to the training data using slicing with 'sample_size'
    start = time() # Get start time
    learner = learner.fit(X_train, y_train)
    end = time() # Get end time
    
    # Calculate the training time
    results['train_time'] = end-start
        
    # Get the predictions on the test set(X_test),
    #       then get predictions on the first 300 training samples(X_train)
    start = time() # Get start time
    predictions_test = learner.predict(X_test)
    end = time() # Get end time

    predictions_train = learner.predict(X_train)

    # Calculate the total prediction time
    results['pred_time'] = end-start
            
    # Compute accuracy on the first 300 training samples
    results['acc_train'] = accuracy_score(y_train, predictions_train)
        
    # Compute accuracy on test set 
    results['acc_test'] = accuracy_score(y_test, predictions_test)
    
    # Compute F-score on the the first 300 training samples
    results['f_train'] = fbeta_score(y_train, predictions_train, beta = 0.5)
        
    # Compute F-score on the test set
    results['f_test'] = fbeta_score(y_test, predictions_test, beta = 0.5)
       
    # Success
    print("{} trained.".format(learner.__class__.__name__))
        
    # Return the results
    return results
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
    
    Hinge_f_vector=[]
    Cold_f_vector=[]
    Hog_f_vector =[]
    
    currentDirectory = os.getcwd()
    directory = filename+"\\"
    path =os.path.join(currentDirectory, directory)
    for imagename in tqdm(os.listdir(path)):
            try:
                img = cv2.imread(os.path.join(path, imagename), cv2.IMREAD_GRAYSCALE)
                # resize all images to the same size
                img = cv2.resize(img, (2000, 1800), interpolation=cv2.INTER_AREA)
                
                start = time()
                # getting all features
                hog_f , hinge_f , cold_f =extract_features([img], opt)
                #storing it in the feature vectors 
                end = time()
                
                # adding image features 
                Hog_f_vector.append(hog_f[0])
                Hinge_f_vector.append(hinge_f[0])
                Cold_f_vector.append(cold_f[0])
                
                time_file.write(str(round((end-start),2)) +"\n")
            except Exception as e :
                    print (e)
                    
    cold_scaler , hinge_scaler , hog_scaler =load_scaler()
    print (np.shape(Hog_f_vector))
    print (np.shape(Cold_f_vector))
    print(np.shape(Hinge_f_vector))

    HOG_feature_scalad = hog_scaler.transform(Hog_f_vector)
    COLD_feature_scaled  = cold_scaler.transform(Cold_f_vector)
    HINGE_feature_scaled = hinge_scaler.transform(Hinge_f_vector)

    all_features = np.concatenate((HOG_feature_scalad, HINGE_feature_scaled), axis=1)
    all_features = np.concatenate((all_features, COLD_feature_scaled), axis=1)

    
    # using trained model for hinge and cold 
    y_pred =model.predict(all_features)
    
    for i in y_pred:
        result_file.write(str(1-i) +"\n")
    result_file.close()
    time_file.close()
# #############################################
def load_scaler():
    cold_scaler = pickle.load(open('COLD_scaler.sav', 'rb'))
    hinge_scaler = pickle.load(open('HINGE_scaler.sav', 'rb'))
    hog_scaler = pickle.load(open('HOG_scaler.sav', 'rb'))
    return cold_scaler , hinge_scaler , hog_scaler


###############################################
def read_csv_data(fileName, isLabel=True):
    rows = []
    with open(fileName, 'r') as file:
        csvreader = csv.reader(file, delimiter=',')
        for row in csvreader:
            if(isLabel==True):
                rows.append(row[0])
            else:
                rows.append(row)
        
    if(isLabel==False):
        data = np.array(rows).astype(np.float32)
    else:
        data = np.array(rows).astype(np.float32).astype(np.int32)
    return data
#######################Getting The Accuracy from a pre Trained Model#########################
def loadModelAndGetAccuraccyLocalTesting(testingFeaturesScaledFileName,testingLabelsFileName, ClfFileName):
    clfModel = pickle.load(open(ClfFileName, 'rb'))
    all_test_features_scaled = read_csv_data(testingFeaturesScaledFileName, isLabel=False)
    y_test=read_csv_data(testingLabelsFileName, isLabel=True)
    test_pred = clfModel.predict(all_test_features_scaled)
    print(f"Testing accuracy is %0.2f"%(np.sum(test_pred == y_test)/len(y_test) * 100))

    return y_test, test_pred


#############################################


def model_evaluation (y_test , y_predict , Lable ="Testing"):
    # testing our model using confusion matrix 
    cf = confusion_matrix(y_test, y_predict)
    cf_sum = cf.sum(axis = 1)[:, np.newaxis]
    cf = np.round(cf / cf_sum * 100, 2)
    float_formatter = "{:.2f}".format
    np.set_printoptions(formatter={'float_kind':float_formatter})    
    cr = 0.0
    for i in range(0, cf.shape[0]):
            cr += cf[i][i]

    cr /= cf.shape[0]
    print('classification rate '+Lable +' = '+ str(np.round(cr, 2)))

    print(classification_report(y_test, y_predict))
    print('Confusion Matrix ' + Lable + " Data")
    
    print(repr(cf))
    sns.heatmap(cf, annot=True,  fmt='', cmap='Blues')
