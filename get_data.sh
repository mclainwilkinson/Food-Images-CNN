#!/bin/bash

# bash script to download and create training and testing sets from kaggle images

mkdir images
mkdir kaggle_data
cd kaggle_data/
kaggle datasets download -d kmader/food41
unzip food41.zip
cd ..
mv kaggle_data/images.zip images/images.zip
rm -rf kaggle_data/
cd images/
unzip images.zip
rm images.zip
cd ..
python3 create_dataset.py
rm -rf images/
echo "done."
