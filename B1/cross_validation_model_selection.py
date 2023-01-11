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
useSavedFeatures = False
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

    # tr_X = tr_X.reshape((train_N, 68*2))
    # te_X = te_X.reshape((test_N, 68*2))

    return train_N, test_N, tr_X, tr_y, te_X, te_y

def img_logreg(training_images, training_labels):
    # defining parameter range 
    param_grid = {'C': [10e-3, 10e-2, 10e-1, 1, 10, 100, 1000, 10000, 10e5, 10e6],
              'penalty': ['l1', 'l2']
              }
    grid = GridSearchCV(linear_model.LogisticRegression(max_iter=100000), param_grid, refit = True, verbose = 3, n_jobs=-1) 
    printWW(f'\n\nparam grid used: \n{param_grid}\n\n')
    grid.fit(training_images, training_labels)

    return grid

def img_SVM(training_images, training_labels):
    param_grid = {'C': [10e-3, 10e-2, 10e-1, 1, 10, 100, 1000, 10000, 10e5, 10e6],
                'gamma': [10e-10, 10e-9, 10e-8, 10e-7, 10e-6, 10e-5, 10e-4, 10e-3],
                'kernel': ['rbf']
                }  
    grid = GridSearchCV(svm.SVC(), param_grid, refit = True, verbose = 3, n_jobs=-1)  
    printWW(f'\n\nparam grid used: \n{param_grid}\n\n')
    grid.fit(training_images, training_labels)

    return grid

def img_kNN(training_images, training_labels):
    param_grid = {'n_neighbors': [1, 3, 5, 7, 9, 11, 13],
              'p': [1, 2, 3, 4, 5],
              }
    
    grid = GridSearchCV(neighbors.KNeighborsClassifier(), param_grid, refit = True, verbose = 3, n_jobs=-1) 
    printWW(f'\n\nparam grid used: \n{param_grid}\n\n')
    grid.fit(training_images, training_labels)

    return grid

def processResults(grid, test_images, test_labels): 
    printWW(f'best params: {grid.best_params_}\n\n')
    printWW(f'best estimator: {grid.best_estimator_}\n\n') 

    pred = grid.predict(test_images)

    printWW(f'Accuracy: {accuracy_score(test_labels, pred)}\n\n')
    printWW(f'Classification report: {classification_report(test_labels, pred)}\n\n')


start_time = time.time()

train_N, test_N, tr_X, tr_y, te_X, te_y= get_data()
gridLogReg = img_logreg(tr_X, tr_y)
processResults(gridLogReg, te_X, te_y)

gridSVM = img_SVM(tr_X, tr_y)
processResults(gridSVM, te_X, te_y)

gridKNN = img_kNN(tr_X, tr_y)
processResults(gridKNN, te_X, te_y)

printWW("\n\n--- %s seconds ---\n\n" % (time.time() - start_time))

saveOutputToFile()