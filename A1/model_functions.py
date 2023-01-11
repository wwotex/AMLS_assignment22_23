from sklearn.model_selection import GridSearchCV
from sklearn import svm, tree, linear_model, neighbors
from sklearn.metrics import classification_report,accuracy_score
import numpy as np
import globals

globals.initialize()

def printWW(str):
    print(str)
    globals.full_output += str

def img_logreg(training_images, training_labels):
    # defining parameter range 
    param_grid = {'C': [10e-3, 10e-2, 10e-1, 1, 10, 100, 1000, 10000, 10e5, 10e6],
              'penalty': ['l1', 'l2'],
              'solver':['liblinear']
              }
    grid = GridSearchCV(linear_model.LogisticRegression(max_iter=100000), param_grid, refit = True, verbose = 3, n_jobs=-1) 
    printWW(f'\n\nparam grid used: \n{param_grid}\n\n')
    grid.fit(training_images, training_labels)

    return grid

def img_SVM(training_images, training_labels):
    C_range = np.logspace(-2, 10, 7)
    gamma_range = np.logspace(-9, 3, 7)

    param_grid = {'C': C_range,
                'gamma': gamma_range,
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

