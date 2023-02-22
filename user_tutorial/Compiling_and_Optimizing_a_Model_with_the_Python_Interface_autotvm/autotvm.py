# Compile a pre-trained ResNet-50 v2 model for the TVM runtime.
# Run a real image through the compiled model, and interpret the output and model performance.
# Tune the model that model on a CPU using TVM.
# Re-compile an optimized model using the tuning data collected by TVM.
# Run the image through the optimized model, and compare the output and model performance.

from cProfile import label
import onnx
from tvm.contrib.download import download_testdata
from PIL import Image
import numpy as np
import tvm.relay as relay
import tvm
from tvm.contrib import graph_executor

onnx_model = onnx.load("resnet50-v2-7.onnx")

np.random.seed(0)

def get_img():
    img_url = "https://s3.amazonaws.com/model-server/inputs/kitten.jpg"
    img_path = download_testdata(img_url, "imagenet_cat.png", module="data")

    # Resize it to 224x224
    resized_image = Image.open(img_path).resize((224, 224))
    img_data = np.asarray(resized_image).astype("float32")

    # Our input image is in HWC layout while ONNX expects CHW input, so convert the array
    img_data = np.transpose(img_data, (2, 0, 1))

    # Normalize according to the ImageNet input specification
    imagenet_mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
    imagenet_stddev = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))
    norm_img_data = (img_data / 255 - imagenet_mean) / imagenet_stddev

    # Add the batch dimension, as we are expecting 4-dimensional input: NCHW.
    img_data = np.expand_dims(norm_img_data, axis=0)
    return img_data
    
    
def time_(module):
    import timeit

    timing_number = 10
    timing_repeat = 10
    unoptimized = (
        np.array(timeit.Timer(lambda: module.run()).repeat(repeat=timing_repeat, number=timing_number))
        * 1000
        / timing_number
    )
    unoptimized = {
        "mean": np.mean(unoptimized),
        "median": np.median(unoptimized),
        "std": np.std(unoptimized),
    }

    print(unoptimized)

def post_process(tvm_output):
    from scipy.special import softmax

    # Download a list of labels
    labels_url = "https://s3.amazonaws.com/onnx-model-zoo/synset.txt"
    labels_path = download_testdata(labels_url, "synset.txt", module="data")

    with open(labels_path, "r") as f:
        labels = [l.rstrip() for l in f]

    # Open the output and read the output tensor
    scores = softmax(tvm_output)
    scores = np.squeeze(scores)
    ranks = np.argsort(scores)[::-1]
    for rank in ranks[0:5]:
        print("class='%s' with probability=%f" % (labels[rank], scores[rank]))
    return labels
        
def unopt(onnx_model, img_data):
    target = "llvm"
    input_name = "data"
    shape_dict = {input_name: img_data.shape}

    mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)

    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(mod, target=target, params=params)

    dev = tvm.device(str(target), 0)
    module = graph_executor.GraphModule(lib["default"](dev))

    dtype = "float32"
    module.set_input(input_name, img_data)
    module.run()
    output_shape = (1, 1000)
    tvm_output = module.get_output(0, tvm.nd.empty(output_shape)).numpy()
    return tvm_output, module, mod, params

def softmax(z): 
    return np.exp(z)/((np.exp(z)).sum())

def opt(mod, target, params, labels):
    import tvm.auto_scheduler as auto_scheduler
    from tvm.autotvm.tuner import XGBTuner
    from tvm import autotvm
    # 搜索几个配置
    number = 10
    # 每个配置重复几次
    repeat = 1
    min_repeat_ms = 0  # since we're tuning on a CPU, can be set to 0
    timeout = 10  # in seconds

    # create a TVM runner
    # 搜索算法信息
    runner = autotvm.LocalRunner(
        number=number,
        repeat=repeat,
        timeout=timeout,
        min_repeat_ms=min_repeat_ms,
        enable_cpu_cache_flush=True,
    )
    # 成本模型信息
    tuning_option = {
        "tuner": "xgb",
        "trials": 20,
        "early_stopping": 100,
        "measure_option": autotvm.measure_option(
            builder=autotvm.LocalBuilder(build_func="default"), 
            runner=runner
            ),
        "tuning_records": "resnet-50-v2-autotuning.json",
    }
    
    # begin by extracting the tasks from the onnx model
    tasks = autotvm.task.extract_from_program(mod["main"], target=target, params=params)

    # Tune the extracted tasks sequentially.
    for i, task in enumerate(tasks):
        prefix = "[Task %2d/%2d] " % (i + 1, len(tasks))
        tuner_obj = XGBTuner(task, loss_type="rank")
        tuner_obj.tune(
            n_trial=min(tuning_option["trials"], len(task.config_space)),
            early_stopping=tuning_option["early_stopping"],
            measure_option=tuning_option["measure_option"],
            callbacks=[
                autotvm.callback.progress_bar(tuning_option["trials"], prefix=prefix),
                autotvm.callback.log_to_file(tuning_option["tuning_records"]),
            ],
        )
    with autotvm.apply_history_best(tuning_option["tuning_records"]):
        with tvm.transform.PassContext(opt_level=3, config={}):
            lib = relay.build(mod, target=target, params=params)

    dev = tvm.device(str(target), 0)
    module = graph_executor.GraphModule(lib["default"](dev))
    dtype = "float32"
    input_name = "data"
    shape_dict = {input_name: img_data.shape}
    module.set_input(input_name, img_data)
    module.run()
    output_shape = (1, 1000)
    tvm_output = module.get_output(0, tvm.nd.empty(output_shape)).numpy()

    scores = softmax(tvm_output)
    scores = np.squeeze(scores)
    ranks = np.argsort(scores)[::-1]
    for rank in ranks[0:5]:
        print("class='%s' with probability=%f" % (labels[rank], scores[rank]))
    return module
    
if __name__ == '__main__':
    img_data = get_img()
    tvm_output, module, mod, params = unopt(onnx_model=onnx_model, img_data=img_data)
    time_(module=module)
    labels = post_process(tvm_output=tvm_output)
    opt_module = opt(mod=mod, target='llvm', params=params, labels=labels)
    time_(opt_module)