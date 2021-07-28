import os
import cv2
import pickle
import random
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
import random

def get_train_valid_data(data_dir, resize):
    """
    get the training and validation data from the given path

    :param data_dir: path to train/valid dataset which includes the categories files
    :param resize: the reshape size (None means don't reshape)
    :return train/valid data
    """

    data = []
    categories = os.listdir(data_dir)

    for category in categories:
        path = os.path.join(data_dir, category)
        class_num = categories.index(category)
        for image_name in os.listdir(path):
            try:
                image = cv2.imread(os.path.join(path, image_name), cv2.IMREAD_COLOR)
                image = image[:,:,[2,1,0]]
                if resize==None:
                    data.append([image, class_num])
                else:
                    image = cv2.resize(image, (resize, resize))
                    data.append([image, class_num])
            except Exception as e:
                pass
    return data

def get_pickle_data(path):
    """
    get the data from saved pickle file
    :param a path to pickle file, which should include the pickle file name
    :return the data saved in pickle file
    """

    pickle_in = open(path, "rb")
    data = pickle.load(pickle_in)
    return data

def save_pickle_data(x, y, path, x_name, y_name):
    """
    save the data as pickle file
    :param x: x-data
    :param y: y-data
    :param path: path which should not include pickle name
    :param x_name: x-data pickle file name
    :param y_name: y-data pickle file name
    :return "Failed"-save failed, "Success"-save successfully
    """
    try: 
        pickle_out = open(os.path.join(path, x_name), "wb")
        pickle.dump(x, pickle_out)

        pickle_out = open(os.path.join(path, y_name), "wb")
        pickle.dump(y, pickle_out)
        pickle_out.close()
    except Exception as e:
        return "Failed"
    return "Success"

def data_shuffle(data):
    """
    Shuffle the data
    :param data: The data needs to be shuffled
    :return shuffled data
    """
    random.shuffle(data)
    return data

def data_split(data):
    """
    Split the data into x and y

    :param origin data
    :return x, y
    """
    x = []
    y = []
    for image, label in data:
        x.append(image)
        y.append(label)
    
    return x, y

def data_augmentation_random(input_path,
                    output_path,
                    augmentation_img_num,
                    rotation_range,
                    width_shift_range,
                    height_shift_range,
                    zoom_range,
                    horizontal_flip,
                    vertical_flip,
                    save_prefix="augmented"):

    data_gen = ImageDataGenerator(
        rotation_range=rotation_range, 
        width_shift_range=width_shift_range,
        height_shift_range=height_shift_range,
        zoom_range=zoom_range,
        horizontal_flip=horizontal_flip,
        vertical_flip=vertical_flip)

    img_list = os.listdir(input_path)

    for i in range(augmentation_img_num):
        rand_num = random.randint(0, (len(img_list)-1))
        aug_img = cv2.imread(os.path.join(input_path, img_list[rand_num]))
        aug_img = aug_img[:,:,[2,1,0]]
        for j in data_gen.flow(x=aug_img.reshape((1,)+aug_img.shape), batch_size=1, save_to_dir=output_path, save_prefix= img_list[rand_num] + "_" + save_prefix):
            break