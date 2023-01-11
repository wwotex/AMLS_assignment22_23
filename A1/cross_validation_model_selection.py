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

globals.initialize()

global train_N, test_N

# -----------------------------CONFIG-------------------------------

warnings.filterwarnings('ignore')
useSavedFeatures = True
saveFeatures = False
savedFeaturesFilename = '200_data_points.npy'
train_N = 200
test_N = 20

# ---------------------FUNCTION DEFINITIONS-------------------------

base_dir = os.path.dirname(__file__)
saved_data_dir = os.path.join(base_dir, 'saved_data')
# full_output = ''

def printWW(str):
    print(str)
    globals.full_output += str


def saveOutputToFile():
    globals.full_output = f'Datetime: {datetime.now()}\n\n train_N: {train_N}\t\ttest_N: {test_N}\n\n' + globals.full_output + '\n\n\n\n'
    with open(os.path.join(saved_data_dir, 'model_selection_output.txt'), 'a') as f:
        f.write(globals.full_output)

    pass

def get_data():
    global train_N, test_N

    path = os.path.join(saved_data_dir, savedFeaturesFilename)

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
# gridLogReg = img_logreg(tr_X, tr_y)
# processResults(gridLogReg, te_X, te_y)

gridSVM = mf.img_SVM(tr_X, tr_y)
mf.processResults(gridSVM, te_X, te_y)
with open(os.path.join(saved_data_dir, 'saved_grid_SVM.pkl'), 'wb') as outp:
    pickle.dump(gridSVM, outp, pickle.HIGHEST_PROTOCOL)


# gridKNN = img_kNN(tr_X, tr_y)
# processResults(gridKNN, te_X, te_y)

printWW("\n\n--- %s seconds ---\n\n" % (time.time() - start_time))

saveOutputToFile()