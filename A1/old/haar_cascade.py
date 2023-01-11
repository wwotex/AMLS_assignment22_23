import os
import numpy as np
from keras_preprocessing import image
import cv2
import dlib
from sklearn.metrics import classification_report,accuracy_score
import time
from datetime import datetime
import warnings

# # how much data to use
# training_N = 200
# test_N = 20

# define paths
base_dir = os.path.dirname(__file__)
assignment_dir = os.path.normpath(base_dir + os.sep + os.pardir)
training_set_dir = os.path.join(assignment_dir,'Datasets\\celeba')
test_set_dir = os.path.join(assignment_dir,'Datasets\\celeba_test')
training_images_dir = os.path.join(training_set_dir,'img')
test_images_dir = os.path.join(test_set_dir,'img')
labels_filename = 'labels.csv'

detector = dlib.get_frontal_face_detector()

def rect_to_bb(rect):
    # take a bounding predicted by dlib and convert it
    # to the format (x, y, w, h) as we would normally do
    # with OpenCV
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y

    # return a tuple of (x, y, w, h)
    return (x, y, w, h)


def run_dlib_shape(image):
    # in this function we load the image, detect the landmarks of the face, and then return the image and the landmarks
    # load the input image, resize it, and convert it to grayscale
    resized_image = image.astype('uint8')

    gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    gray = gray.astype('uint8')

    # detect faces in the grayscale image
    rects = detector(gray, 1)
    if not rects:
        return (36, 86, 90, 90)
    coord = rect_to_bb(rects[0])

    return coord


def get_images_labels(training, training_N, test_N):
    images_dir = training_images_dir if training else test_images_dir
    dataset_dir = training_set_dir if training else test_set_dir
    N = training_N if training else test_N
    # array of image paths
    image_paths = [os.path.join(images_dir, l) for l in os.listdir(images_dir)[:N]]
    
    # load labels.csv
    labels_file = open(os.path.join(dataset_dir, labels_filename), 'r')
    lines = labels_file.readlines()

    # make list of gender labels
    gender_labels = {line.split('\t')[0] : int(line.split('\t')[2]) for line in lines[1:]}
    all_features = []
    all_labels = []
    
    negative_paths = ''
    positive_paths = ''

    # get features from each image with dlib
    for img_path in image_paths:
        file_name= img_path.split('.')[-2].split('\\')[-1]
        
        if gender_labels[file_name] == -1:
            negative_paths += img_path + '\n'
        else:
            # load image
            img = image.img_to_array(image.load_img(img_path, target_size = None, interpolation = 'bicubic'))
            rect = run_dlib_shape(img)
            positive_paths += img_path + f' 1 {rect[0]} {rect[1]} {rect[2]} {rect[3]}\n'
        
        
        # all_features.append(img)
        all_labels.append(gender_labels[file_name])

    with open(os.path.join(base_dir, 'haar_neg.txt'), 'w') as f:
        f.write(negative_paths)

    with open(os.path.join(base_dir, 'haar_pos.txt'), 'w') as f:
        f.write(positive_paths)

    landmark_features = np.array(all_features)
    # convert (-1,1) into (0,1)
    all_labels = (np.array(all_labels) + 1)/2 
    return landmark_features, all_labels

get_images_labels(training=False, training_N=5000, test_N=1000)

# global full_output, train_N, test_N

# # -----------------------------CONFIG-------------------------------

# warnings.filterwarnings('ignore')
# train_N = 200
# test_N = 20

# # ---------------------FUNCTION DEFINITIONS-------------------------

# base_dir = os.path.dirname(__file__)
# full_output = ''

# def printWW(str):
#     global full_output
#     print(str)
#     full_output += str


# def saveOutputToFile():
#     global full_output
#     full_output = f'Datetime: {datetime.now()}\n\n train_N: {train_N}\t\ttest_N: {test_N}\n\n' + full_output + '\n\n\n\n'
#     with open(os.path.join(base_dir, 'model_selection_output.txt'), 'a') as f:
#         f.write(full_output)

#     pass


# def get_data():
#     global train_N, test_N

#     tr_X, tr_y = fe.get_images_labels(training=True, training_N=train_N, test_N=test_N)
#     te_X, te_y = fe.get_images_labels(training=False, training_N=train_N, test_N=test_N)


#     # tr_X, te_X, tr_Y, te_Y = train_test_split(X, y, test_size=0.10, random_state=42)

#     return train_N, test_N, tr_X, tr_y, te_X, te_y

# def img_SVM(training_images, training_labels, test_images, test_labels):

#     classifier = cv2.CascadeClassifier()

#     # fitting the model for grid search 
#     classifier. (training_images, training_labels) 

#     for tsam in test_images:
#         objects = classifier.detectMultiScale(tsam, 1.3, 5)
#         # loop through the detected objects
#         for (x, y, w, h) in objects:
#             pred = classifier.predict(tsam[y:y+h, x:x+w])

#     printWW(f'Prediction: {pred}\n\n')
#     printWW(f'Accuracy: {accuracy_score(test_labels, pred)}\n\n')
#     printWW(f'Classification report: {classification_report(test_labels, pred)}\n\n')


# start_time = time.time()

# train_N, test_N, tr_X, tr_y, te_X, te_y= get_data()
# pred=img_SVM(tr_X, tr_y, te_X, te_y)

# printWW("\n\n--- %s seconds ---\n\n" % (time.time() - start_time))

# saveOutputToFile()