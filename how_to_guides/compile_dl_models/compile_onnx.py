from site import execusercustomize
import onnx
import numpy as np
import tvm
from tvm import te
import tvm.relay as relay
from tvm.contrib.download import download_testdata
from PIL import Image
from matplotlib import pyplot as plt


# Load pretrained ONNX model
def load_model():
    model_url = "".join([
        "https://gist.github.com/zhreshold/",
        "bcda4716699ac97ea44f791c24310193/raw/",
        "93672b029103648953c4e5ad3ac3aadf346a4cdc/",
        "super_resolution_0.2.onnx",
    ])
    model_path = download_testdata(model_url, "super_resolution.onnx")
    onnx_model = onnx.load(model_path)
    return onnx_model

def load_image():
    img_url = "https://github.com/dmlc/mxnet.js/blob/main/data/cat.png?raw=true"
    img_path = download_testdata(img_url, "cat.png", module="data")
    img = Image.open(img_path).resize((224, 224))
    return img
    
def compile_relay(x):
    target = "llvm"
    input_name = "1"
    # Passing in the shape dictionary to the relay.frontend.from_onnx method tells relay which ONNX parameters are inputs, 
    # and which are parameters, 
    # and provides a static definition of the input size.
    shape_dict = {input_name: x.shape}
    mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)
    device = tvm.cpu(0)
    with tvm.transform.PassContext(opt_level=1):
        executor = relay.build_module.create_executor(
            "graph", mod, device, target, params
        ).evaluate()
    dtype = "float32"
    tvm_output = executor(tvm.nd.array(x.astype(dtype))).numpy()
    return tvm_output
    
onnx_model = load_model()
img = load_image()

img_ycbcr = img.convert("YCbCr")  # convert to YCbCr
img_y, img_cb, img_cr = img_ycbcr.split()
x = np.array(img_y)[np.newaxis, np.newaxis, :, :]

tvm_output = compile_relay(x)

out_y = Image.fromarray(np.uint8((tvm_output[0, 0]).clip(0, 255)), mode="L")
out_cb = img_cb.resize(out_y.size, Image.BICUBIC)
out_cr = img_cr.resize(out_y.size, Image.BICUBIC)
result = Image.merge("YCbCr", [out_y, out_cb, out_cr]).convert("RGB")
canvas = np.full((672, 672 * 2, 3), 255)
canvas[0:224, 0:224, :] = np.asarray(img)
canvas[:, 672:, :] = np.asarray(result)
plt.imshow(canvas.astype(np.uint8))
plt.show()
