import pickle
from sklearn.model_selection import GridSearchCV
from sklearn import svm, tree, linear_model, neighbors, ensemble
from sklearn.metrics import classification_report,accuracy_score
import numpy as np
import globals
import graphing as gp
import os
<<<<<<< HEAD
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
=======
>>>>>>> parent of 6258080 (added B1 solutions)


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
    C_range = np.logspace(-4, 6, 6)
    gamma_range = np.logspace(-8, 4, 7)

    param_grid = {'C': C_range,
                'gamma': gamma_range,
                'kernel': ['rbf']
                }  
    grid = GridSearchCV(svm.SVC(), param_grid, refit = True, verbose = 3, n_jobs=-1, return_train_score=True)  
    printWW(f'\n\nparam grid used: \n{param_grid}\n\n')
    grid.fit(training_images, training_labels)


    return grid

def img_kNN(training_images, training_labels):
    n_neighbors = np.arange(1, 10, 1)
    param_grid = {'n_neighbors': n_neighbors}
    
    grid = GridSearchCV(neighbors.KNeighborsClassifier(), param_grid, refit = True, verbose = 3, n_jobs=-1, return_train_score=True) 
    printWW(f'\n\nparam grid used: \n{param_grid}\n\n')
    grid.fit(training_images, training_labels)

    return grid

<<<<<<< HEAD
def create_model(num_filters=32, filter_size=3, pool_size=2):
    model = Sequential()
   # Add the first convolutional layer
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(150, 150, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Add a second convolutional layer
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Add a third convolutional layer
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Flatten the output of the convolutional layers
    model.add(Flatten())

    # Add a fully connected layer
    model.add(Dense(512, activation='relu'))

    # Add dropout to prevent overfitting
    model.add(Dropout(0.5))

    # Add the output layer
    model.add(Dense(5, activation='softmax'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def img_neural(training_images, training_labels):
    model = KerasClassifier(build_fn=create_model, epochs=10, batch_size=32, verbose=2)

    filepath = os.path.join(globals.saved_data_dir, "neural\\model_{epoch:02d}.h5")
    checkpoint = ModelCheckpoint(filepath, save_weights_only=False, save_best_only=False)
    
    model.fit(training_images, to_categorical(training_labels), callbacks=[checkpoint])

    return model

=======
>>>>>>> parent of 6258080 (added B1 solutions)
def processResults(grid, test_images, test_labels, model): 
    printWW(f'best params: {grid.best_params_}\n\n')
    printWW(f'best estimator: {grid.best_estimator_}\n\n') 

    pred = grid.predict(test_images)
    
    with open(os.path.join(globals.saved_data_dir, f'saved_results_{model}.pkl'), 'wb') as outp:
        pickle.dump([pred, test_labels], outp, pickle.HIGHEST_PROTOCOL)

    printWW(f'Accuracy: {accuracy_score(test_labels, pred)}\n\n')
    printWW(f'Classification report: {classification_report(test_labels, pred)}\n\n')

def processNeuralResults(model, test_images, test_labels):
    if model == None:
        filepath = os.path.join(globals.saved_data_dir, "neural\\model_08.h5")
        model = load_model(filepath)

    # printWW(f'\nbatch size: 32\n')
    # printWW(f'epochs: 8\n') 

    model_pred = model.predict(test_images)
    pred = np.argmax(model_pred, axis=1)
    print(f'predictions: \n{pred[:20]}\n')
    print(f'{test_labels[:20]}\n\n')

    # with open(os.path.join(globals.saved_data_dir, f'saved_results_neural.pkl'), 'wb') as outp:
    #     pickle.dump([pred, test_labels], outp, pickle.HIGHEST_PROTOCOL)

    # printWW(f'Accuracy: {accuracy_score(test_labels, pred)}\n\n')
    # printWW(f'Classification report: {classification_report(test_labels, pred)}\n\n')