# https://tvm.apache.org/docs/tutorial/tvmc_command_line_driver.html

python -m tvm.driver.tvmc compile \
--target "llvm" \
--input-shapes "data:[1,3,224,224]" \
--output resnet50-v2-7-tvm.tar \
../model/resnet50-v2-7.onnx

python ./preprocess.py

python -m tvm.driver.tvmc run \
--inputs imagenet_cat.npz \
--output predictions.npz \
resnet50-v2-7-tvm.tar

python ./postprocess.py

python -m tvm.driver.tvmc tune \
--target "llvm -mcpu=skylake" \
--output resnet50-v2-7-autotuner_records.json \
../model/resnet50-v2-7.onnx
# 跑不了
# RuntimeError: FLOP estimator fails for this operator. 
# Error msg: The length of axis is not constant. . Please use `cfg.add_flop` to manually set FLOP for this operator