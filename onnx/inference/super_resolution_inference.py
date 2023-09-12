"""
(optional) Exporting a Model from PyTorch to ONNX and Running it using ONNX Runtime
========================================================================

In this tutorial, we describe how to convert a model defined
in PyTorch into the ONNX format and then run it with ONNX Runtime.

ONNX Runtime is a performance-focused engine for ONNX models,
which inferences efficiently across multiple platforms and hardware
(Windows, Linux, and Mac and on both CPUs and GPUs).
ONNX Runtime has proved to considerably increase performance over
multiple models as explained `here
<https://cloudblogs.microsoft.com/opensource/2019/05/22/onnx-runtime-machine-learning-inferencing-0-4-release>`__

For this tutorial, you will need to install `ONNX <https://github.com/onnx/onnx>`__
and `ONNX Runtime <https://github.com/microsoft/onnxruntime>`__.
You can get binary builds of ONNX and ONNX Runtime with
``pip install onnx onnxruntime``.
ONNX Runtime recommends using the latest stable runtime for PyTorch.

"""

# Some standard imports
import io
import numpy as np

from torch import nn
import torch.utils.model_zoo as model_zoo
import torch.onnx


# Super Resolution model definition in PyTorch
import torch.nn as nn
import torch.nn.init as init


class SuperResolutionNet(nn.Module):
    def __init__(self, upscale_factor, inplace=False):
        super(SuperResolutionNet, self).__init__()

        self.relu = nn.ReLU(inplace=inplace)
        self.conv1 = nn.Conv2d(1, 64, (5, 5), (1, 1), (2, 2))
        self.conv2 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.conv3 = nn.Conv2d(64, 32, (3, 3), (1, 1), (1, 1))
        self.conv4 = nn.Conv2d(32, upscale_factor ** 2, (3, 3), (1, 1), (1, 1))
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

        self._initialize_weights()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.pixel_shuffle(self.conv4(x))
        return x

    def _initialize_weights(self):
        init.orthogonal_(self.conv1.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv2.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv3.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv4.weight)

# Create the super-resolution model by using the above model definition.
torch_model = SuperResolutionNet(upscale_factor=3)

# Load pretrained model weights
model_url = 'https://s3.amazonaws.com/pytorch/test_data/export/superres_epoch100-44c6958e.pth'
batch_size = 1    # just a random number

# Initialize model with the pretrained weights
map_location = lambda storage, loc: storage
if torch.cuda.is_available():
    map_location = None
torch_model.load_state_dict(model_zoo.load_url(model_url, map_location=map_location))

# set the model to inference mode
torch_model.eval()


######################################################################
# Exporting a model in PyTorch works via tracing or scripting. This
# tutorial will use as an example a model exported by tracing.
# To export a model, we call the ``torch.onnx.export()`` function.
# This will execute the model, recording a trace of what operators
# are used to compute the outputs.
# Because ``export`` runs the model, we need to provide an input
# tensor ``x``. The values in this can be random as long as it is the
# right type and size.
# Note that the input size will be fixed in the exported ONNX graph for
# all the input's dimensions, unless specified as a dynamic axes.
# In this example we export the model with an input of batch_size 1,
# but then specify the first dimension as dynamic in the ``dynamic_axes``
# parameter in ``torch.onnx.export()``.
# The exported model will thus accept inputs of size [batch_size, 1, 224, 224]
# where batch_size can be variable.
#
# To learn more details about PyTorch's export interface, check out the
# `torch.onnx documentation <https://pytorch.org/docs/master/onnx.html>`__.
#

# Input to the model
x = torch.randn(batch_size, 1, 224, 224, requires_grad=True)
torch_out = torch_model(x)

# Export the model
torch.onnx.export(torch_model,               # model being run
                  x,                         # model input (or a tuple for multiple inputs)
                  "super_resolution.onnx",   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=10,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
                  dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                'output' : {0 : 'batch_size'}})

import onnx

onnx_model = onnx.load("super_resolution.onnx")
onnx.checker.check_model(onnx_model)


import onnxruntime

ort_session = onnxruntime.InferenceSession("super_resolution.onnx")

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

# compute ONNX Runtime output prediction
ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
ort_outs = ort_session.run(None, ort_inputs)

# compare ONNX Runtime and PyTorch results
np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)

print("Exported model has been tested with ONNXRuntime, and the result looks good!")


######################################################################
#
# We first resize the image to fit the size of the model's input (224x224).
# Then we split the image into its Y, Cb, and Cr components.
# These components represent a grayscale image (Y), and
# the blue-difference (Cb) and red-difference (Cr) chroma components.
# The Y component being more sensitive to the human eye, we are
# interested in this component which we will be transforming.
# After extracting the Y component, we convert it to a tensor which
# will be the input of our model.
#

from PIL import Image
import torchvision.transforms as transforms

img = Image.open("/Users/fl/github/IfonPIM/test/onnx_infer/input/cat.jpg")

resize = transforms.Resize([224, 224])
img = resize(img)

img_ycbcr = img.convert('YCbCr')
img_y, img_cb, img_cr = img_ycbcr.split()

to_tensor = transforms.ToTensor()
img_y = to_tensor(img_y)
img_y.unsqueeze_(0)


######################################################################
# Now, as a next step, let's take the tensor representing the
# grayscale resized cat image and run the super-resolution model in
# ONNX Runtime as explained previously.
#

ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(img_y)}
ort_outs = ort_session.run(None, ort_inputs)
img_out_y = ort_outs[0]


######################################################################
# At this point, the output of the model is a tensor.
# Now, we'll process the output of the model to construct back the
# final output image from the output tensor, and save the image.
# The post-processing steps have been adopted from PyTorch
# implementation of super-resolution model
# `here <https://github.com/pytorch/examples/blob/master/super_resolution/super_resolve.py>`__.
#

img_out_y = Image.fromarray(np.uint8((img_out_y[0] * 255.0).clip(0, 255)[0]), mode='L')

# get the output image follow post-processing step from PyTorch implementation
final_img = Image.merge(
    "YCbCr", [
        img_out_y,
        img_cb.resize(img_out_y.size, Image.BICUBIC),
        img_cr.resize(img_out_y.size, Image.BICUBIC),
    ]).convert("RGB")

# Save the image, we will compare this with the output image from mobile device
final_img.save("/Users/fl/github/IfonPIM/test/onnx_infer/input/cat_superres_with_ort.jpg")
