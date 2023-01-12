import os
import numpy as np
from keras_preprocessing import image
import cv2
import dlib

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
path = os.path.join(base_dir,'shape_predictor_68_face_landmarks.dat')

predictor = dlib.shape_predictor(path)

def shape_to_np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((shape.num_parts, 2), dtype=dtype)

    # loop over all facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, shape.num_parts):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    # return the list of (x, y)-coordinates
    return coords

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
    num_faces = len(rects)

    if num_faces == 0:
        return None, resized_image

    face_areas = np.zeros((1, num_faces))
    face_shapes = np.zeros((136, num_faces), dtype=np.int64)

    # loop over the face detections
    for (i, rect) in enumerate(rects):
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        temp_shape = predictor(gray, rect)
        temp_shape = shape_to_np(temp_shape)

        # convert dlib's rectangle to a OpenCV-style bounding box
        # [i.e., (x, y, w, h)],
        #   (x, y, w, h) = face_utils.rect_to_bb(rect)
        (x, y, w, h) = rect_to_bb(rect)
        face_shapes[:, i] = np.reshape(temp_shape, [136])
        face_areas[0, i] = w * h
    # find largest face and keep
    dlibout = np.reshape(np.transpose(face_shapes[:, np.argmax(face_areas)]), [68, 2])

    return dlibout, resized_image

def extract_features_labels(training, training_N, test_N):
    images_dir = training_images_dir if training else test_images_dir
    dataset_dir = training_set_dir if training else test_set_dir
    N = training_N if training else test_N
    # array of image paths
    image_paths = [os.path.join(images_dir, l) for l in os.listdir(images_dir)[:N]]
    
    # load labels.csv
    labels_file = open(os.path.join(dataset_dir, labels_filename), 'r')
    lines = labels_file.readlines()

    # make list of gender labels
    gender_labels = {line.split('\t')[0] : int(line.split('\t')[3]) for line in lines[1:]}
    all_features = []
    all_labels = []
    
    # get features from each image with dlib
    for img_path in image_paths:
        file_name= img_path.split('.')[-2].split('\\')[-1]
        # load image
        img = image.img_to_array(
            image.load_img(img_path,
                            target_size = None,
                            interpolation = 'bicubic'))
        features, _ = run_dlib_shape(img)
        if features is not None:
            all_features.append(features)
            all_labels.append(gender_labels[file_name])

    landmark_features = np.array(all_features)
    # convert (-1,1) into (0,1)
    all_labels = (np.array(all_labels) + 1)/2 
    return landmark_features, all_labels
