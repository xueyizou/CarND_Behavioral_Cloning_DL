# -*- coding: utf-8 -*-
import numpy as np
from scipy.misc import imread,imsave, imresize
from sklearn.model_selection import train_test_split
import csv
import shutil
import os
import matplotlib.pyplot as plt

csvfile_path='./data/driving_log.csv'
original_data_path = './data/IMG/'
preprocessed_data_path = './data/M_IMG/'

steer_offset=0.25
h, w = 80, 80

def process_images():
    """crop and resize all images.
       generating more images by flipping them.
    """
    for file in os.listdir(original_data_path):
        path = original_data_path+file
        image= imread(path) #160X320X3
        image_cropped= imresize(image[60:140,:,:],(h,w))
        imsave(preprocessed_data_path+file, image_cropped)

        image_flipped = np.fliplr(image_cropped)
        imsave(preprocessed_data_path+"F_"+file, image_flipped)

#
def get_dataset():
    """using left/right camera data to recover the car back to track
       augmenting data by flipping some of the less frequent images.
    """
    reader = csv.reader(open(csvfile_path, newline=''), delimiter=',')

    img_names = []
    steers = []

    img_names_flip=[]
    steers_flip=[]

    reader.__next__()
    for row in reader:
        center_file_name = row[0].strip().replace('IMG/','')
        left_file_name = row[1].strip().replace('IMG/','')
        right_file_name = row[2].strip().replace('IMG/','')

        base_steer=float(row[3])
        img_names.append(center_file_name)
        steers.append(base_steer)

        #left image
        img_names.append(left_file_name)
        left_steer =min(base_steer+ steer_offset, 1.0)
        steers.append(left_steer)

        #right image
        img_names.append(right_file_name)
        right_steer=max(base_steer-steer_offset, -1.0)
        steers.append(right_steer)

        if abs(base_steer)>0.01:
            img_names_flip.append('F_'+center_file_name)
            steers_flip.append(-base_steer)

            img_names_flip.append('F_'+left_file_name)
            steers_flip.append(-left_steer)

            img_names_flip.append('F_'+right_file_name)
            steers_flip.append(-right_steer)

    img_names.extend(img_names_flip)
    steers.extend(steers_flip)

    return img_names, steers


def split_dataset(img_names, steers):
    """
    trian/validation dataset split. No test data here since we can use the simulator to test the learnt model.
    """
    img_names_train, img_names_val, steers_train, steers_val = train_test_split(img_names, steers, test_size=0.2, random_state=0)

    print("num. train: %d" % len(steers_train))
    np.save("img_names_train.npy", img_names_train)
    np.save("steers_train.npy", steers_train)

    print("num. val: %d" % len(steers_val))
    np.save("img_names_val.npy", img_names_val)
    np.save("steers_val.npy", steers_val)

if __name__=="__main__":
    ##do the following commented code only for once to save time.
    if os.path.exists(preprocessed_data_path):
        shutil.rmtree(preprocessed_data_path)
    os.mkdir(preprocessed_data_path)
    process_images()

    img_names, steers = get_dataset()
    print("num. data: %d" % len(steers))
    plt.hist(steers, 201)
    plt.title("distribution of steer angles of the dataset")
    plt.show()

    split_dataset(img_names, steers)

