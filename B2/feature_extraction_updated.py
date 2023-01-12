import os
import numpy as np
from keras_preprocessing import image

# # how much data to use
# training_N = 200
# test_N = 20

# define paths
base_dir = os.path.dirname(__file__)
assignment_dir = os.path.normpath(base_dir + os.sep + os.pardir)
training_set_dir = os.path.join(assignment_dir,'Datasets\\cartoon_set')
test_set_dir = os.path.join(assignment_dir,'Datasets\\cartoon_set_test')
training_images_dir = os.path.join(training_set_dir,'img')
test_images_dir = os.path.join(test_set_dir,'img')
labels_filename = 'labels.csv'
saved_data_dir = os.path.join(base_dir, 'saved_data')

def run_dlib_shape(image):
    resized_image = image.astype('uint8')
    return resized_image

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
    gender_labels = {line.split('\t')[0] : int(line.split('\t')[1]) for line in lines[1:]}
    all_features = []
    all_labels = []
    
    # get features from each image with dlib
    for img_path in image_paths:
        file_name= img_path.split('.')[-2].split('\\')[-1]
        print(f'processing: {file_name}')
        # load image
        img = image.img_to_array(
            image.load_img(img_path,
                            target_size = (150, 150),
                            interpolation = 'bicubic'))
        
        features = run_dlib_shape(img)

        if features is not None:
            all_features.append(features)
            all_labels.append(gender_labels[file_name])

    landmark_features = np.array(all_features)
    # convert (-1,1) into (0,1)
    all_labels = np.array(all_labels)
    return landmark_features, all_labels


if __name__ == "__main__":
    extract_features_labels(training=False, test_N=20, training_N=1)