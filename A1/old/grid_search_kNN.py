import os
import feature_extraction as fe
import numpy as np
from sklearn.metrics import classification_report,accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn import svm, tree, linear_model, neighbors 
import time
from datetime import datetime
import warnings

global full_output, train_N, test_N

# -----------------------------CONFIG-------------------------------

warnings.filterwarnings('ignore')
useSavedFeatures = True
saveFeatures = False
train_N = 200
test_N = 20

# ---------------------FUNCTION DEFINITIONS-------------------------

base_dir = os.path.dirname(__file__)
full_output = ''

def printWW(str):
    global full_output
    print(str)
    full_output += str


def saveOutputToFile():
    global full_output
    full_output = f'Datetime: {datetime.now()}\n\n train_N: {train_N}\t\ttest_N: {test_N}\n\n' + full_output + '\n\n\n\n'
    with open(os.path.join(base_dir, 'model_selection_output.txt'), 'a') as f:
        f.write(full_output)

    pass


def get_data():
    global train_N, test_N

    path = os.path.join(base_dir, 'saved_features.npy')

    if useSavedFeatures and os.path.exists(path):
        tr_X, tr_y, te_X, te_y = np.load(path, allow_pickle=True)
    else:
        tr_X, tr_y = fe.extract_features_labels(training=True, training_N=train_N, test_N=test_N)
        te_X, te_y = fe.extract_features_labels(training=False, training_N=train_N, test_N=test_N)
        
        if saveFeatures:
            np.save(path, [tr_X, tr_y, te_X, te_y])

    # tr_X, te_X, tr_Y, te_Y = train_test_split(X, y, test_size=0.10, random_state=42)

    train_N = tr_X.shape[0]
    test_N = te_X.shape[0]

    return train_N, test_N, tr_X, tr_y, te_X, te_y

def img_SVM(training_images, training_labels, test_images, test_labels):

    # defining parameter range 
    param_grid = {'n_neighbors': [1, 3, 5, 7, 9, 11, 13],
              'p': [1, 2, 3, 4, 5],
              }
    
    grid = GridSearchCV(neighbors.KNeighborsClassifier(), param_grid, refit = True, verbose = 3, n_jobs=-1) 

    printWW(f'\n\nparam grid used: \n{param_grid}\n\n')

    # n_features = training_images.shape[1]
    # gamma = 1 / (n_features * training_images.var())
    # print(f'gamma [scale]: {gamma}')

    # fitting the model for grid search 
    grid.fit(training_images, training_labels) 
    
    printWW(f'best params: {grid.best_params_}\n\n')
    printWW(f'best estimator: {grid.best_estimator_}\n\n') 

    pred = grid.predict(test_images)

    # printWW(f'Prediction: {pred}\n\n')
    printWW(f'Accuracy: {accuracy_score(test_labels, pred)}\n\n')
    printWW(f'Classification report: {classification_report(test_labels, pred)}\n\n')


start_time = time.time()

train_N, test_N, tr_X, tr_y, te_X, te_y= get_data()
pred=img_SVM(tr_X.reshape((train_N, 68*2)), tr_y, te_X.reshape((test_N, 68*2)), te_y)

printWW("\n\n--- %s seconds ---\n\n" % (time.time() - start_time))

saveOutputToFile()