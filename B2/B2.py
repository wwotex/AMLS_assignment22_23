import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import feature_extraction_updated as fe
import time
from datetime import datetime
import warnings
import globals
import pickle
from sklearn.metrics import classification_report,accuracy_score
import output as out
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import to_categorical

warnings.filterwarnings('ignore')
globals.initialize()
global train_N, test_N

# -----------------------------CONFIG-------------------------------

# you can train your model straight from the images or use pre-saved extracted features. 
# you can also save the features again to a numpy file
useSavedFeatures = False
saveFeatures = False
savedFeaturesFilename = 'saved_images_gray.pkl'
# number of datapoints to use
train_N = 10000
test_N = 2500

# ---------------------FUNCTION DEFINITIONS-------------------------

# fetch data into X and y
def get_neural_data():
    global train_N, test_N

    path = os.path.join(globals.saved_data_dir, savedFeaturesFilename)

    if useSavedFeatures and os.path.exists(path):
        with open(path, 'rb') as inp:
            tr_X, tr_y, te_X, te_y = pickle.load(inp)
    else:
        tr_X, tr_y = fe.extract_images_labels(training=True, training_N=train_N, test_N=test_N)
        te_X, te_y = fe.extract_images_labels(training=False, training_N=train_N, test_N=test_N)
        
        with open(path, 'wb') as outp:
            pickle.dump([tr_X, tr_y, te_X, te_y], outp, pickle.HIGHEST_PROTOCOL)

    # tr_X, te_X, tr_Y, te_Y  = train_test_split(X, y, test_size=0.10, random_state=42)

    train_N = tr_X.shape[0]
    test_N = te_X.shape[0]

    out.printWW(f'\t\tdata size: {train_N}  {test_N}')

    # tr_X = tr_X.reshape((train_N, 68*2))
    # te_X = te_X.reshape((test_N, 68*2))

    return train_N, test_N, tr_X, tr_y, te_X, te_y

# auxilary function for defining the CNN model
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

# measure execution time
start_time = time.time()

train_N, test_N, tr_X, tr_y, te_X, te_y= get_neural_data()

out.printWW(f'Datetime: {datetime.now()}\n\n train_N: {train_N}\t\ttest_N: {test_N}\n\n')

# fit data into CNN model and make predictions
model = KerasClassifier(build_fn=create_model, epochs=8, batch_size=32, verbose=2)
model.fit(tr_X, to_categorical(tr_y))
pred = model.predict(te_X)
# pred = np.argmax(pred, axis=1)
print(f'predictions: \n{pred[:20]}\n')
print(f'{te_y[:20]}\n\n')

# print result analysis and execution time
out.printWW(f'Accuracy: {accuracy_score(te_y, pred)}\n\n')
out.printWW(f'Classification report: {classification_report(te_y, pred)}\n\n')


out.printWW("\n\n--- %s seconds ---\n\n" % (time.time() - start_time))

out.saveOutputToFile()