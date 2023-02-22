import os

import numpy as np
import tvm
from tvm import te, auto_scheduler, topi
from tvm.topi.testing import conv2d_nchw_python

@auto_scheduler.register_workload
def conv2d_layer(N, H, W, CO, CI, KH, KW, stride, padding):
    data = te.placeholder((N, CI, H, W), name="data")
    kernel = te.placeholder((CO, CI, KH, KW), name="kernel")
    bias = te.placeholder((1, CO, 1, 1), name="bias")
    conv = topi.nn.conv2d_nchw(data, kernel, stride, padding, dilation=1, out_dtype="float32")
    out = topi.nn.relu(conv + bias)
    return [data, kernel, bias, out]

# create search task
target = tvm.target.Target("cuda")

# Use the last layer in ResNet-50
N, H, W, CO, CI, KH, KW, strides, padding = 1, 7, 7, 512, 512, 3, 3, (1, 1), (1, 1)
task = auto_scheduler.SearchTask(
    func=conv2d_layer, args=(N, H, W, CO, CI, KH, KW, strides, padding), target=target
)

# Inspect the computational graph
print("Computational DAG:")
print(task.compute_dag)

# measure_ctx launches a different process for measurement to provide isolation. It can protect the main process from GPU crashes during measurement and avoid other runtime conflicts.
# min_repeat_ms defines the minimum duration of one “repeat” in every measurement. 

# This can warmup the GPU, which is necessary to get accurate measurement results. Typically, we recommend a value >= 300 ms.

# num_measure_trials is the number of measurement trials we can use during the search. 
# We only make 10 trials in this tutorial for a fast demonstration. In practice, 1000 is a good value for the search to converge. You can do more trials according to your time budget.

# In addition, we use RecordToFile to dump measurement records into a file conv2d.json. 
# The measurement records can be used to query the history best, resume the search, and do more analyses later.

# see auto_scheduler.TuningOptions, auto_scheduler.LocalRPCMeasureContext for more parameters.

log_file = "conv2d.json"
measure_ctx = auto_scheduler.LocalRPCMeasureContext(min_repeat_ms=300)
tune_option = auto_scheduler.TuningOptions(
    num_measure_trials=10,  # change this to 1000 to achieve the best performance
    runner=measure_ctx.runner,
    measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
    verbose=2,
)

# run search
# Run auto-tuning (search)
task.tune(tune_option)
# Apply the best schedule
sch, args = task.apply_best(log_file)

# Kill the measurement process
del measure_ctx

print("Lowered TIR:")
print(tvm.lower(sch, args, simple_mode=True))

# check correctness and evaluate performance
func = tvm.build(sch, args, target)

# Check correctness
data_np = np.random.uniform(size=(N, CI, H, W)).astype(np.float32)
weight_np = np.random.uniform(size=(CO, CI, KH, KW)).astype(np.float32)
bias_np = np.random.uniform(size=(1, CO, 1, 1)).astype(np.float32)
conv_np = conv2d_nchw_python(data_np, weight_np, strides, padding)
out_np = np.maximum(conv_np + bias_np, 0.0)

dev = tvm.cuda()
data_tvm = tvm.nd.array(data_np, device=dev)
weight_tvm = tvm.nd.array(weight_np, device=dev)
bias_tvm = tvm.nd.array(bias_np, device=dev)
out_tvm = tvm.nd.empty(out_np.shape, device=dev)
func(data_tvm, weight_tvm, bias_tvm, out_tvm)

# Check results
np.testing.assert_allclose(out_np, out_tvm.numpy(), rtol=1e-3)

# Evaluate execution time
evaluator = func.time_evaluator(func.entry_name, dev, min_repeat_ms=500)
print(
    "Execution time of this operator: %.3f ms"
    % (np.median(evaluator(data_tvm, weight_tvm, bias_tvm, out_tvm).results) * 1000)
)

# using the record file
print("Equivalent python schedule:")
print(task.print_best(log_file, print_mode="schedule"))

print("CUDA source code:")
print(task.print_best(log_file, print_mode="cuda"))

# 自己设定search policy and cost model，然后利用log_file运行
def resume_search(task, log_file):
    print("Resume search:")
    cost_model = auto_scheduler.XGBModel()
    cost_model.update_from_file(log_file)
    search_policy = auto_scheduler.SketchPolicy(
        task, cost_model, init_search_callbacks=[auto_scheduler.PreloadMeasuredStates(log_file)]
    )
    measure_ctx = auto_scheduler.LocalRPCMeasureContext(min_repeat_ms=300)
    tune_option = auto_scheduler.TuningOptions(
        num_measure_trials=5,
        runner=measure_ctx.runner,
        measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
    )
    task.tune(tune_option, search_policy=search_policy)

    # Kill the measurement process
    del measure_ctx


resume_search(task, log_file)