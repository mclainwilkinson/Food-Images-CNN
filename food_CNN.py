import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.datasets as dsets
import matplotlib.pyplot as plt
import h5py
import numpy as np
import time
import pandas as pd
pd.set_option('display.width', 320)
pd.set_option("display.max_columns", 15)
# -----------------------------------------------------------------------------------
# training parameters
batch_size = 40
num_epochs = 30

# train and test set files
h5train_file = "food_train.h5"
h5test_file = "food_test.h5"

# class for creating torch dataset from HDF5 database
class DatasetFromHdf5(torch.utils.data.Dataset):
    def __init__(self, file_path):
        super(DatasetFromHdf5, self).__init__()
        hf = h5py.File(file_path)
        self.data = hf.get('images')
        self.target = hf.get('labels')
        self.classes = hf.get('categories')

    def __getitem__(self, index):
        return torch.from_numpy(self.data[index, :, :, :].T).float(), self.target[index]

    def __len__(self):
        return self.data.shape[0]


# create datasets and data loaders
train_dataset = DatasetFromHdf5(h5train_file)

test_dataset = DatasetFromHdf5(h5test_file)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# print training/testing dataset sizes and target classes
target_classes = tuple(target_class.decode() for target_class in train_dataset.classes)
print('Number of training images:', train_dataset.__len__())
print('Number of testing images:', test_dataset.__len__())
print('target classes:', target_classes)
print('-------------------------------------------------------')
# -----------------------------------------------------------------------------------
# define the network model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(                # input size = 128 x 128
            nn.Conv2d(3, 64, kernel_size=10, padding=4),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2, padding=1))  # output size = 64 x 64
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=10, padding=4),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2, padding=1))  # output size = 32 x 32
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=10, padding=4),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2, padding=1))  # output size = 16 x 16
        self.layer4 = nn.Sequential(
            nn.Conv2d(256, 384, kernel_size=10, padding=4),
            nn.BatchNorm2d(384),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2, padding=1))  # output size = 8 x 8
        self.layer5 = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=4, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2, padding=1))  # output size = 5 x 5
        self.layer6 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2))             # output size = 2 x 2
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(2 * 2 * 128, 12)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.dropout(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

# -----------------------------------------------------------------------------------
# initialize the model
cnn = CNN()
cnn.cuda()
# -----------------------------------------------------------------------------------
# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(cnn.parameters(), lr=0.001)
# ----------------------------------------------------------------
# Train the Model
num_images = train_dataset.__len__()
start = time.time()
epoch_list = [e + 1 for e in range(num_epochs)]
loss_list = []
print('Training...')

for epoch in range(num_epochs):
    for i, (data, target) in enumerate(train_loader):
        images = Variable(data).cuda()
        labels = Variable(target).cuda()

        # Forward + Backward + Optimize
        optimizer.zero_grad()
        outputs = cnn(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f'
                  % (epoch + 1, num_epochs, i + 1, num_images // batch_size, loss.item()))
    loss_list.append(loss.item())

end = time.time()

# plot loss against epoch
plt.plot(epoch_list, loss_list)
plt.title('loss value by epoch')
plt.xlabel('epoch #')
plt.ylabel('loss value')
plt.show()

print("training time:", '%.2f' % (end - start), 'seconds')
print('-------------------------------------------------------')
# -----------------------------------------------------------------------------------
# Test the Model
cnn.eval()  # Change model to 'eval' mode (BN uses moving mean/var).

print('Testing...')

correct = 0
total = 0
correct_array = np.zeros(len(target_classes))
total_array = np.zeros(len(target_classes))
confusion = {x: [0 for i in range(len(target_classes))] for x in range(len(target_classes))}

for images, labels in test_loader:
    input_images = Variable(images).cuda()
    outputs = cnn(input_images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted.cpu() == labels).sum()
    for pred, actual in zip(predicted.cpu(), labels):
        confusion[int(actual)][int(pred)] += 1
        total_array[actual] += 1
        if pred == actual:
            correct_array[actual] += 1
    images = images
# -----------------------------------------------------------------------------------
# display overall results
print('-------------------------------------------------------')
print('Results')
print('Accuracy of the model on the test images: %d %%' % (100 * correct / total))
print('')
# display individual category results
print('Individual class accuracy:')
target_classes = tuple(target_class.decode() for target_class in train_dataset.classes)
for i, t in enumerate(target_classes):
    print(t + '\t', int(correct_array[i]), 'correct', '/', int(total_array[i]), 'total',
          '\t=>\t', '%.2f' % (100 * correct_array[i] / total_array[i]), '%', 'accuracy')

# show and print out class/accuracy for highest and lowest accuracy values
pct_array = correct_array / total_array
best = int(np.argmax(pct_array))
worst = int(np.argmin(pct_array))
print('')
print('class with highest accuracy:', target_classes[best],
      '%.2f' % (100 * correct_array[best] / total_array[best]), '%')
print('class with lowest accuracy:', target_classes[worst],
      '%.2f' % (100 * correct_array[worst] / total_array[worst]), '%')
print('')

# create and show confusion matrix
confusion = pd.DataFrame.from_dict(confusion)
confusion.columns = target_classes
confusion.index = target_classes
print('Confusion Matrix')
print(confusion)

# show 10 test images with target class & top 3 predicted outputs
for i, (output, label) in enumerate(zip(outputs.data.cpu(), labels)):
    if i < 10:
        output = np.array(output)
        ndx = output.argsort()[-3:][::-1]
        caption = 'Predicted Outputs: \n'
        for j, n in enumerate(ndx):
            caption = caption + (str(j + 1) + ': ' + target_classes[n] + '\n')
        image = np.array(images[i]).T
        plt.figure(i)
        plt.imshow(image)
        plt.title(target_classes[label])
        plt.xlabel(caption)
        plt.tight_layout()
plt.show()
# -----------------------------------------------------------------------------------
# Save the Trained Model
torch.save(cnn.state_dict(), 'cnn.pkl')
# -----------------------------------------------------------------------------------
