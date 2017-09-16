"""PyTorch version of Conv K-Means.

Author: Yuhuang Hu
Email : duguyue100@gmail.com
"""
from __future__ import print_function
import os

import matplotlib.pyplot as plt
import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms

from convk.kmeans import ConvKMeans

batch_size = 512
num_iteration = 30
# prepare data
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
root_path = os.path.join(os.environ["HOME"], "share")
testset = torchvision.datasets.CIFAR10(root=root_path, train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)


# visualize data
def imshow(img):
    img = (img / 2 + 0.5)*255*1.9
    npimg = np.asarray(img.numpy(), dtype=np.uint8)
    plt.imshow(np.transpose(npimg, (1, 2, 0)), cmap="hot")

dataiter = iter(testloader)
images, labels = dataiter.next()

input_shape = images.shape
inputs = torch.autograd.Variable(images)

model = ConvKMeans(input_shape, 32, (16, 16), stride=1,
                   padding=0, dilation=1, groups=1, bias=False)

trained_kernel = model.kernel.clone()
trained_kernel_max = trained_kernel.abs().max()
trained_kernel_max = trained_kernel_max.view(1, 1, 1, 1).expand_as(
    trained_kernel)
print (trained_kernel_max.size())
trained_kernel = trained_kernel/trained_kernel_max

imshow(torchvision.utils.make_grid(model.kernel.data))
plt.show()

for iteration in range(num_iteration):
    model(inputs)

    print ("[MESSAGE] Iteration %d is done!" % (iteration))

trained_kernel = model.kernel.clone()
trained_kernel_max = trained_kernel.abs().max()
trained_kernel_max = trained_kernel_max.view(1, 1, 1, 1).expand_as(
    trained_kernel)
print (trained_kernel_max.size())
trained_kernel = trained_kernel/trained_kernel_max

imshow(torchvision.utils.make_grid(model.kernel.data))
plt.show()
