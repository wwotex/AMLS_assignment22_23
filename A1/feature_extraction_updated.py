import os
import numpy as np
from keras_preprocessing import image
import cv2
import dlib
import output as out
from tqdm import tqdm

# define paths
base_dir = os.path.dirname(__file__)
assignment_dir = os.path.normpath(base_dir + os.sep + os.pardir)
training_set_dir = os.path.join(assignment_dir,'Datasets\\celeba')
test_set_dir = os.path.join(assignment_dir,'Datasets\\celeba_test')
training_images_dir = os.path.join(training_set_dir,'img')
test_images_dir = os.path.join(test_set_dir,'img')
labels_filename = 'labels.csv'
saved_data_dir = os.path.join(base_dir, 'saved_data')

# create detector and predictor objects for face and landmark detection
detector = dlib.get_frontal_face_detector()
path = os.path.join(saved_data_dir,'shape_predictor_68_face_landmarks.dat')

predictor = dlib.shape_predictor(path)

# convert facial landmarks into an array of tuples
def shape_to_np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((shape.num_parts, 2), dtype=dtype)

    # loop over all facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, shape.num_parts):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    # return the list of (x, y)-coordinates
    return coords

# convert the rectangles from dlib into bounding boxes from open-cv
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


# extract landmarks from one image
def run_dlib_shape(image):
    # in this function we load the image, detect the landmarks of the face, and then return the image and the landmarks
    # load the input image, resize it, and convert it to grayscale
    resized_image = image.astype('uint8')

    # change color scheme to grayscale
    gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    gray = gray.astype('uint8')

    # detect faces
    rects = detector(gray, 1)
    num_faces = len(rects)

    if num_faces == 0:
        return None

    face_areas = np.zeros((1, num_faces))
    face_shapes = np.zeros((136, num_faces), dtype=np.int64)

    # loop over the face detections
    for (i, rect) in enumerate(rects):
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        temp_shape = predictor(gray, rect)
        temp_shape = shape_to_np(temp_shape)


        # generate images for report
        # cv2_img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # max_intensity = np.max(cv2_img)
        # cv2_img = cv2_img * (255.0 / max_intensity)
        # cv2_img = np.uint8(cv2_img)
        # for shape in temp_shape:
        #     x, y = shape[0], shape[1]
        #     cv2_img = cv2.circle(cv2_img, (x,y), radius=2, color=(0, 0, 255), thickness=-1)
        # cv2_img = imutils.resize(cv2_img, width=500)
        # cv2.imwrite(os.path.join(base_dir,'landmarks.jpg'), cv2_img)


        # convert dlib's rectangle to a OpenCV-style bounding box
        # [i.e., (x, y, w, h)],
        (x, y, w, h) = rect_to_bb(rect)
        face_shapes[:, i] = np.reshape(temp_shape, [136])
        face_areas[0, i] = w * h

    # find largest face and keep
    dlibout = np.reshape(np.transpose(face_shapes[:, np.argmax(face_areas)]), [136])

    # return dlibout[:34]
    return dlibout

# extract features for all images
def extract_features_labels(training, training_N, test_N):
    # choose directories
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
    
    out.printWW(f'\nfetching {"training" if training else "test"} data\n')

    # get features from each image with dlib
    for img_path in tqdm(image_paths):
        file_name= img_path.split('.')[-2].split('\\')[-1]
        # load image

        img = image.img_to_array(
            image.load_img(img_path,
                            target_size = None,
                            interpolation = 'bicubic'))
        
        features = run_dlib_shape(img)

        if features is not None:
            all_features.append(features)
            all_labels.append(gender_labels[file_name])

    landmark_features = np.array(all_features)
    # convert (-1,1) into (0,1)
    all_labels = (np.array(all_labels) + 1)/2 
    return landmark_features, all_labels


# extract_features_labels(training=False, test_N=1, training_N=200)