# Food-Images-CNN
classifying images of food using a convolution neural network built with PyTorch

### Requirements
* Python (3.5)
* PyTorch
* NVIDIA GPU card with CUDA capability

### How to Run
1. Run the following commands to download the Kaggle API, unzip, and h5py python packages
```
pip3 install kaggle
```
```
pip3 install h5py
```
```
sudo apt-get install unzip
```
2. Sign up for a Kaggle account at https://www.kaggle.com. 
3. Go to the 'Account' tab of your user profile (https://www.kaggle.com/<username>/account) and select 'Create API Token'. This will trigger the download of kaggle.json, a file containing your API credentials. 
4. Transfer kaggle.json file to your cloud instance: 
```
scp -i yourkey.pem kaggle.json ubuntu@x.x.x.x:~/.kaggle/kaggle.json
```
5. Change permission
```
chmod 600 ~/.kaggle/kaggle.json 
```
6. run get_data.sh to download images and create training and testing HDF5 databases
```
sh get_data.sh
```
7. You should now have 2 files food_train.h5 and food_test.h5 in your working directory. Train and test the network using the following command:
```
python3 food_CNN.py
```
