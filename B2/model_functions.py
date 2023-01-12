import pickle
from sklearn.model_selection import GridSearchCV
from sklearn import svm, tree, linear_model, neighbors, ensemble
from sklearn.metrics import classification_report,accuracy_score
import numpy as np
import globals
import graphing as gp
import os
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from sklearn.model_selection import train_test_split



globals.initialize()

def printWW(str):
    print(str)
    globals.full_output += str

def create_model(num_filters=32, filter_size=3, pool_size=2):
    model = Sequential()
   # Add the first convolutional layer
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(150, 150, 3)))
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
    model = KerasClassifier(build_fn=create_model)

    # filepath = os.path.join(globals.saved_data_dir, "neural\\model_{epoch:02d}.h5")
    # checkpoint = ModelCheckpoint(filepath, save_weights_only=False, save_best_only=False)
    
    X_train, X_test, y_train, y_test = train_test_split(training_images, training_labels, test_size=0.20, random_state=42)

    # model.fit(training_images, to_categorical(training_labels), callbacks=[checkpoint])
    history = model.fit(X_train, to_categorical(y_train), batch_size=32, epochs=1, verbose=2, validation_data=(X_test, to_categorical(y_test)))
    
    history.history

    from matplotlib import pyplot as plt

    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(os.path.join(globals.image_dir, f'neural_learning.jpg'))
    plt.show()

    return model, history

def processNeuralResults(model, test_images, test_labels):
    if model == None:
        filepath = os.path.join(globals.saved_data_dir, "neural\\model_08.h5")
        model = load_model(filepath)

    printWW(f'\nbatch size: 32\n')
    printWW(f'epochs: 8\n') 

    model_pred = model.predict(test_images)
    pred = np.argmax(model_pred, axis=0)
    print(f'predictions: \n{pred}\n')
    print(f'{test_labels[:20]}\n\n')

    with open(os.path.join(globals.saved_data_dir, f'saved_results_neural.pkl'), 'wb') as outp:
        pickle.dump([pred, test_labels], outp, pickle.HIGHEST_PROTOCOL)

    printWW(f'Accuracy: {accuracy_score(test_labels, pred)}\n\n')
    printWW(f'Classification report: {classification_report(test_labels, pred)}\n\n')