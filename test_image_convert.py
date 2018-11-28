import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from skimage.transform import resize

test_image = 'waffle.jpg'

im = Image.open(test_image)
npImg = np.array(im)
print('original shape:', npImg.shape)

plt.imshow(npImg)
plt.title('original image')
plt.show()

resized_image = resize(npImg, (64, 64, 3))
print('new shape:', resized_image.shape)

plt.imshow(resized_image)
plt.title('resized image')
plt.show()
