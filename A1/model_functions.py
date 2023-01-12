import pickle
from sklearn.model_selection import GridSearchCV
from sklearn import svm, tree, linear_model, neighbors, ensemble
from sklearn.metrics import classification_report,accuracy_score
import numpy as np
import globals
import graphing as gp
import os


globals.initialize()

def printWW(str):
    print(str)
    globals.full_output += str

def img_random_forest(training_images, training_labels):
    n_estimators = np.arange(10, 1000, 10)

    param_grid = {'n_estimators': n_estimators}
    grid = GridSearchCV(ensemble.RandomForestClassifier(), param_grid, refit = True, verbose = 3, n_jobs=-1, return_train_score=True)
    printWW(f'\n\nparam grid used: \n{param_grid}\n\n')
    grid.fit(training_images, training_labels)

    return grid

def img_SVM(training_images, training_labels):
    # C_range = np.logspace(-4, 6, 6)
    # gamma_range = np.logspace(-8, 4, 7)
    C_range = np.logspace(2, 5, 10)
    gamma_range = np.logspace(-8, -2, 8)

    param_grid = {'C': C_range,
                'gamma': gamma_range,
                'kernel': ['rbf']
                }  
    grid = GridSearchCV(svm.SVC(), param_grid, refit = True, verbose = 3, n_jobs=-1, return_train_score=True)  
    printWW(f'\n\nparam grid used: \n{param_grid}\n\n')
    grid.fit(training_images, training_labels)


    return grid

def img_kNN(training_images, training_labels):
    n_neighbors = np.arange(1, 100, 1)
    param_grid = {'n_neighbors': n_neighbors}
    
    grid = GridSearchCV(neighbors.KNeighborsClassifier(), param_grid, refit = True, verbose = 3, n_jobs=-1, return_train_score=True) 
    printWW(f'\n\nparam grid used: \n{param_grid}\n\n')
    grid.fit(training_images, training_labels)

    return grid

def processResults(grid, test_images, test_labels, model): 
    printWW(f'best params: {grid.best_params_}\n\n')
    printWW(f'best estimator: {grid.best_estimator_}\n\n') 

    pred = grid.predict(test_images)
    
    with open(os.path.join(globals.saved_data_dir, f'saved_results_{model}.pkl'), 'wb') as outp:
        pickle.dump([pred, test_labels], outp, pickle.HIGHEST_PROTOCOL)

    printWW(f'Accuracy: {accuracy_score(test_labels, pred)}\n\n')
    printWW(f'Classification report: {classification_report(test_labels, pred)}\n\n')

