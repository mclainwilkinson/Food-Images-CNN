import os
import numpy as np
import h5py
from PIL import Image
import matplotlib.pyplot as plt
from skimage.transform import resize
from sklearn.model_selection import train_test_split
import random

'''
This file will create an HDF5 database of images, labels, and categories given that the following
directories and files exist in the current working directory:
    - folder titled 'images' containing subfolders titled with a type of food
    - these subfolders (i.e. 'tacos' or 'hot_dog') contain .jpg files of pictures of food from folder title 
'''

# define 12 new categories in array
new_category_names = ['smooth_dessert', 'red_meat', 'egg', 'pasta', 'soup', 'salad', 'fried_food',
                      'sandwich', 'noodles', 'cake', 'sweet_breakfast', 'shell']

# create mapping from old categories --> new categories
category_name_mapping = {'bibimbap': 'egg',
                        'caesar_salad': 'salad',
                        'carrot_cake': 'cake',
                        'chocolate_cake': 'cake',
                        'chocolate_mousse': 'smooth_dessert',
                        'club_sandwich': 'sandwich',
                        'cup_cakes': 'cake',
                        'deviled_eggs': 'egg',
                        'eggs_benedict': 'egg',
                        'escargots': 'shell',
                        'filet_mignon': 'red_meat',
                        'fish_and_chips': 'fried_food',
                        'french_fries': 'fried_food',
                        'french_onion_soup': 'soup',
                        'french_toast': 'sweet_breakfast',
                        'fried_calamari': 'fried_food',
                        'frozen_yogurt': 'smooth_dessert',
                        'gnocchi': 'pasta',
                        'greek_salad': 'salad',
                        'grilled_cheese_sandwich': 'sandwich',
                        'hamburger': 'sandwich',
                        'hot_and_sour_soup': 'soup',
                        'hot_dog': 'sandwich',
                        'huevos_rancheros': 'egg',
                        'ice_cream': 'smooth_dessert',
                        'lasagna': 'pasta',
                        'lobster_bisque': 'soup',
                        'lobster_roll_sandwich': 'sandwich',
                        'macaroni_and_cheese': 'pasta',
                        'miso_soup': 'soup',
                        'mussels': 'shell',
                        'onion_rings': 'fried_food',
                        'oysters': 'shell',
                        'pad_thai': 'noodles',
                        'pancakes': 'sweet_breakfast',
                        'panna_cotta': 'smooth_dessert',
                        'pho': 'noodles',
                        'pork_chop': 'red_meat',
                        'poutine': 'fried_food',
                        'prime_rib': 'red_meat',
                        'pulled_pork_sandwich': 'sandwich',
                        'ramen': 'noodles',
                        'ravioli': 'pasta',
                        'red_velvet_cake': 'cake',
                        'spaghetti_bolognese': 'pasta',
                        'spaghetti_carbonara': 'pasta',
                        'steak': 'red_meat',
                        'strawberry_shortcake': 'cake',
                        'tiramisu': 'smooth_dessert',
                        'waffles': 'sweet_breakfast'}

# lists to add images and labels
images = []
labels = []

# resize images to be 128x128x3, add images and labels to respective lists
for folder in os.listdir('images'):
    if folder in category_name_mapping:
        print('converting images from:', folder, '-->', category_name_mapping[folder])
        for f in os.listdir('images/' + folder):
            image = np.array(Image.open('images/' + folder + '/' + f))
            try:
                resized_image = resize(image, (128, 128, 3))
                images.append(resized_image)
                labels.append(new_category_names.index(category_name_mapping[folder]))
            except:
                continue
    else:
        print(folder, 'not used to create target classes')
'''
# shuffle images and labels together
image_labels = list(zip(images, labels))
random.shuffle(image_labels)
images, labels = zip(*image_labels)
images = list(images)
labels = list(labels)
'''
# convert lists to numpy arrays
images = np.array(images)
labels = np.array(labels)
categories = np.array(new_category_names, dtype='S10')

'''
# split into train and test sets 80/20 split
random.seed(0)
num_images = len(images)
split_index = 4 * num_images // 5
train_images = images[:split_index]
train_labels = labels[:split_index]
test_images = images[split_index:]
test_labels = labels[split_index:]
print('number of training images:', len(train_images))
print('number of testing images:', len(test_images))
'''

# use sklearn train_test_split to create training and testing sets
train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.2, stratify=labels)

# create new dataset objects
train_filename = 'food_train.h5'
train_file = h5py.File(train_filename, 'w')
test_filename = 'food_test.h5'
test_file = h5py.File(test_filename, 'w')

# add datasets to dataset objects
train_categories = train_file.create_dataset('categories', data=categories)
train_labels = train_file.create_dataset('labels', data=train_labels)
train_images = train_file.create_dataset('images', data=train_images)

test_categories = test_file.create_dataset('categories', data=categories)
test_labels = test_file.create_dataset('labels', data=test_labels)
test_images = test_file.create_dataset('images', data=test_images)

# confirm that objects are in dataset
print('training dataset categories:', list(train_file.keys()))
print('label for training image 0:', train_file.get('labels')[0])
print('category for training image 0:', train_file.get('categories')[train_file.get('labels')[0]].decode())
image0 = train_file.get('images')[0]
print('size of training image 0:', image0.shape)
plt.imshow(image0)
plt.title('training image 0')
plt.show()
print('target classes for dataset:')
for cat in train_file.get('categories'):
    print(cat.decode())
u, c = np.unique(test_labels, return_counts=True)
for un, co in zip(u, c):
    print(un, co)
train_file.close()
test_file.close()
