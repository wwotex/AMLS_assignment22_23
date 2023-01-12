import os
import feature_extraction_updated as fe
import numpy as np
import time
from datetime import datetime
import warnings
import globals
import pickle
from sklearn import svm
from sklearn.metrics import classification_report,accuracy_score
import output as out

warnings.filterwarnings('ignore')
globals.initialize()
global train_N, test_N

# -----------------------------CONFIG-------------------------------

# you can train your model straight from the images or use pre-saved extracted features. 
# you can also save the features again to a numpy file
useSavedFeatures = False
saveFeatures = False
savedFeaturesFilename = 'all_features_saved_5000.npy'
# number of datapoints to use
train_N = 5000
test_N = 1000

# ---------------------FUNCTION DEFINITIONS-------------------------

# fetch data and extract features into X and y
def get_data():
    global train_N, test_N

    path = os.path.join(globals.saved_data_dir, savedFeaturesFilename)

    # check if user wants to use pre-saved features from a numpy file or to fetch everything straight from the images
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

    out.printWW(f'\n\n\t\tdata size: {train_N}  {test_N}')

    # tr_X = tr_X.reshape((train_N, 68*2))
    # te_X = te_X.reshape((test_N, 68*2))

    return train_N, test_N, tr_X, tr_y, te_X, te_y

# measure execution time
start_time = time.time()

train_N, test_N, tr_X, tr_y, te_X, te_y= get_data()

out.printWW(f'Datetime: {datetime.now()}\n\n train_N: {train_N}\t\ttest_N: {test_N}\n\n')

# fit data into SVM model and make predictions
model = svm.SVC(kernel='rbf', C=10000, gamma = 0.000001)
model.fit(tr_X, tr_y)
pred = model.predict(te_X)

# save results into a pickle file
with open(os.path.join(globals.saved_data_dir, f'saved_results_final.pkl'), 'wb') as outp:
    pickle.dump([pred, te_y], outp, pickle.HIGHEST_PROTOCOL)

# print result analysis and execution time
out.printWW(f'Accuracy: {accuracy_score(te_y, pred)}\n\n')
out.printWW(f'Classification report: {classification_report(te_y, pred)}\n\n')


out.printWW("\n\n--- %s seconds ---\n\n" % (time.time() - start_time))

out.saveOutputToFile()