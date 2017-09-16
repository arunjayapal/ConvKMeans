"""PyTorch version of Conv K-Means.

Author: Yuhuang Hu
Email : duguyue100@gmail.com
"""
from __future__ import print_function
import os

import matplotlib.pyplot as plt
import numpy as np

from keras.preprocessing.image import ImageDataGenerator

import torch
import torchvision
import torchvision.transforms as transforms

from convk.kmeans import ConvKMeans
from convk.utils import load_cifar10

batch_size = 64
num_iteration = 30
# prepare data
#  transform = transforms.Compose(
#      [transforms.ToTensor(),
#       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
#  root_path = os.path.join(os.environ["HOME"], "share")
#  testset = torchvision.datasets.CIFAR10(root=root_path, train=False,
#                                         download=True, transform=transform)
#  testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
#                                           shuffle=False, num_workers=2)
#
#  test_loader = torch.utils.data.DataLoader(
#      torchvision.datasets.MNIST(
#          root_path, train=False, download=True,
#          transform=transforms.Compose([
#              transforms.ToTensor(),
#              transforms.Normalize((0.1307,), (0.3081,))
#          ])),
#      batch_size=batch_size, shuffle=False, num_workers=2)

X_train, _, X_test, _, _ = load_cifar10()
X_train = X_train.transpose((0, 3, 1, 2))
X_test = X_train.transpose((0, 3, 1, 2))

datagen = ImageDataGenerator(
        featurewise_center=True,
        samplewise_center=False,
        featurewise_std_normalization=True,
        samplewise_std_normalization=False,
        zca_whitening=True,
        rotation_range=0,
        width_shift_range=0.125,
        height_shift_range=0.125,
        horizontal_flip=True,
        vertical_flip=False,
        data_format="channels_first")
datagen.fit(X_train)
generator = datagen.flow(X_train, batch_size=batch_size)

print ("[MESSAGE] Data generator is prepared")


# visualize data
def imshow(img):
    img = (img / 2 + 0.5)*255
    npimg = np.asarray(img.numpy(), dtype=np.uint8)
    plt.imshow(np.transpose(npimg, (1, 2, 0)), cmap="hot")

# define input shape
input_shape = (batch_size, 3, 32, 32)

model = ConvKMeans(input_shape, 64, (8, 8), stride=4,
                   padding="same", groups=1, bias=False, lr=0.1)

trained_kernel = model.kernel.clone()
trained_kernel_max = trained_kernel.abs().max()
trained_kernel_max = trained_kernel_max.view(1, 1, 1, 1).expand_as(
    trained_kernel)
trained_kernel = trained_kernel/trained_kernel_max

print ("[MESSAGE] The original filter")
imshow(torchvision.utils.make_grid(model.kernel.data))
plt.show()

for iteration in range(num_iteration):
    prev_kernel = model.kernel.clone()
    for _ in xrange(30):
        X = torch.autograd.Variable(torch.Tensor(generator.next()),
                                    volatile=True)
        model(X)
    post_kernel = model.kernel.clone()
    loss = (post_kernel-prev_kernel).mean().abs().data.numpy()[0]

    print ("[MESSAGE] Iteration %d is done! Loss: %f" % (iteration+1, loss))

trained_kernel = model.kernel.clone()
trained_kernel_max = trained_kernel.abs().max()
trained_kernel_max = trained_kernel_max.view(1, 1, 1, 1).expand_as(
    trained_kernel)
print (trained_kernel_max.size())
trained_kernel = trained_kernel/trained_kernel_max

print ("[MESSAGE] The trained filters")
imshow(torchvision.utils.make_grid(model.kernel.data))
plt.show()
