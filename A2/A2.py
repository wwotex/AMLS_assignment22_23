import os
import feature_extraction_updated as fe
import model_functions as mf
import graphing as gp
import numpy as np
from matplotlib import pyplot as plt
import time
from datetime import datetime
import warnings
import globals
import pickle
from sklearn import svm, neighbors, ensemble
from sklearn.metrics import classification_report,accuracy_score

globals.initialize()

global train_N, test_N

# -----------------------------CONFIG-------------------------------

warnings.filterwarnings('ignore')
useSavedFeatures = True
saveFeatures = False
savedFeaturesFilename = 'all_features_saved_5000.npy'
train_N = 5000
test_N = 1000

# ---------------------FUNCTION DEFINITIONS-------------------------

def printWW(str):
    print(str)
    globals.full_output += str


def saveOutputToFile():
    globals.full_output = f'Datetime: {datetime.now()}\n\n train_N: {train_N}\t\ttest_N: {test_N}\n\n' + globals.full_output + '\n\n\n\n'
    with open(os.path.join(globals.saved_data_dir, 'model_selection_output.txt'), 'a') as f:
        f.write(globals.full_output)

    pass

def get_data():
    global train_N, test_N

    path = os.path.join(globals.saved_data_dir, savedFeaturesFilename)

    if useSavedFeatures and os.path.exists(path):
        tr_X, tr_y, te_X, te_y = np.load(path, allow_pickle=True)
    else:
        tr_X, tr_y = fe.extract_features_labels(training=True, training_N=train_N, test_N=test_N)
        te_X, te_y = fe.extract_features_labels(training=False, training_N=train_N, test_N=test_N)
        
        if saveFeatures:
            np.save(path, [tr_X, tr_y, te_X, te_y])

    # tr_X, te_X, tr_Y, te_Y  = train_test_split(X, y, test_size=0.10, random_state=42)

    train_N = tr_X.shape[0]
    test_N = te_X.shape[0]

    print(f'\t\tdata size: {train_N}  {test_N}')
    print(tr_X.shape)

    # tr_X = tr_X.reshape((train_N, 68*2))
    # te_X = te_X.reshape((test_N, 68*2))

    return train_N, test_N, tr_X, tr_y, te_X, te_y

start_time = time.time()

train_N, test_N, tr_X, tr_y, te_X, te_y= get_data()

model = svm.SVC(kernel='rbf', C=10000, gamma = 0.000001)
model.fit(tr_X, tr_y)
pred = model.predict(te_X)

with open(os.path.join(globals.saved_data_dir, f'saved_results_final.pkl'), 'wb') as outp:
    pickle.dump([pred, te_y], outp, pickle.HIGHEST_PROTOCOL)

printWW(f'Accuracy: {accuracy_score(te_y, pred)}\n\n')
printWW(f'Classification report: {classification_report(te_y, pred)}\n\n')


printWW("\n\n--- %s seconds ---\n\n" % (time.time() - start_time))

saveOutputToFile()