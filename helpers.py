import cv2
import os
import numpy as np
from skimage.feature import hog, local_binary_pattern

from hinge_feature_extraction import * 
from cold_feature_extraction  import * 
from sklearn.svm import  SVC
from sklearn.model_selection import GridSearchCV

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