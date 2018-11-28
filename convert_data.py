import h5py
import matplotlib.pyplot as plt
import numpy as np
import argparse

# get args
parser = argparse.ArgumentParser(description='convert dataset to new dataset')
parser.add_argument('infile', help='file to convert')
parser.add_argument('outfile', help='destination file')
args = parser.parse_args()

infile = args.infile
outfile = args.outfile

# create old database file object and read into lists
filename = infile
original = h5py.File(filename, 'r')

category_arrays = original.get('category')

category_names = [c.decode() for c in original.get('category_names')]

ims = original.get('images')

# define new categories in array
new_category_names = ['pie', 'smooth_dessert', 'red_meat', 'egg', 'pasta', 'soup', 'salad', 'fried_food', 
                      'sandwich', 'rice', 'dumpling', 'noodles', 'cake', 'sweet_breakfast', 'shell']

# create mapping from old categories --> new categories
category_name_mapping = {'apple_pie': 'pie',
                        'bibimbap': 'egg',
                        'caesar_salad': 'salad',
                        'carrot_cake': 'cake',
                        'cheesecake': 'pie',
                        'chocolate_cake': 'cake',
                        'chocolate_mousse': 'smooth_dessert',
                        'club_sandwich': 'sandwich',
                        'cup_cakes': 'cake',
                        'deviled_eggs': 'egg',
                        'dumplings': 'dumpling',
                        'eggs_benedict': 'egg',
                        'escargots': 'shell',
                        'filet_mignon': 'red_meat',
                        'fish_and_chips': 'fried_food',
                        'french_fries': 'fried_food',
                        'french_onion_soup': 'soup',
                        'french_toast': 'sweet_breakfast',
                        'fried_calamari': 'fried_food',
                        'fried_rice': 'rice',
                        'frozen_yogurt': 'smooth_dessert',
                        'gnocchi': 'pasta',
                        'greek_salad': 'salad',
                        'grilled_cheese_sandwich': 'sandwich',
                        'gyoza': 'dumpling',
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
                        'paella': 'rice',
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
                        'risotto': 'rice',
                        'spaghetti_bolognese': 'pasta',
                        'spaghetti_carbonara': 'pasta',
                        'steak': 'red_meat',
                        'strawberry_shortcake': 'cake',
                        'tiramisu': 'smooth_dessert',
                        'waffles': 'sweet_breakfast'}

# create lists for new images and category arrays
new_category_arrays = []
new_ims = []

# iterate through images, add image, new category array if image in new dataset
for i in range(len(ims)):
    category_name = category_names[np.where(category_arrays[i])[0][0]]
    if category_name in category_name_mapping:
        new_category_name = category_name_mapping[category_name]
        new_cat_array = np.full(15, False, dtype=bool)
        position = new_category_names.index(new_category_name)
        new_cat_array[position] = True
        new_category_arrays.append(new_cat_array)
        new_ims.append(ims[i])
    if i % int(len(ims)/10) == 0 and i != 0:
        print('{:.0%}'.format(i/len(ims)), 'complete')

# convert lists to numpy arrays
labels = np.array([np.where(label)[0][0] for label in new_category_arrays])
new_ims = np.array(new_ims)
new_category_names = np.array(new_category_names, dtype='S10')

# create new dataset object
newfilename = outfile
newfile = h5py.File(newfilename, 'w')

# add datasets to dataset object
new_category_names = newfile.create_dataset('categories', data=new_category_names)
new_category_arrays = newfile.create_dataset('labels', data=new_category_arrays)
new_images = newfile.create_dataset('images', data=new_ims)

# confirm that objects are in dataset
print('dataset categories:', list(newfile.keys()))
print('category for image 0:', newfile.get('categories')[0])
print('label for image 0:', newfile.get('labels')[0])
for cat in newfile.get('categories'):
    print(cat.decode())
newfile.close()
