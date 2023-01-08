from os.path import join
import feature_extraction as fe
import numpy as np
from sklearn.metrics import classification_report,accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import svm
import time


def get_data():
    
    tr_X, tr_y = fe.extract_features_labels(training=True)
    te_X, te_y = fe.extract_features_labels(training=False)
    
    # tr_X, te_X, tr_Y, te_Y = train_test_split(X, y, test_size=0.10, random_state=42)

    train_N = tr_X.shape[0]
    test_N = te_X.shape[0]

    return train_N, test_N, tr_X, tr_y, te_X, te_y

def img_SVM(training_images, training_labels, test_images, test_labels):
    classifier = svm.SVC(kernel='linear')

    classifier.fit(training_images, training_labels)

    pred = classifier.predict(test_images)

    print(pred)

    print("Accuracy:", accuracy_score(test_labels, pred))


start_time = time.time()

train_N, test_N, tr_X, tr_y, te_X, te_y= get_data()
pred=img_SVM(tr_X.reshape((train_N, 68*2)), tr_y, te_X.reshape((test_N, 68*2)), te_y)

print("\n\n--- %s seconds ---\n\n" % (time.time() - start_time))
