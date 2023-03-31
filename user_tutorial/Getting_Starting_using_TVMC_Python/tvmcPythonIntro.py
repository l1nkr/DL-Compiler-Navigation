from tvm.driver import tvmc

# convert to relay



# save time, do not need to convert
# model.save("/Users/fl/github/DL-Compiler-Navigation/model/convertedModel")
import onnx

onnx_model = onnx.load_model(r'/Users/fl/github/DL-Compiler-Navigation/model/resnet50-v2-7.onnx')
onnx_model.graph.input[0].type.tensor_type.shape.dim[0].dim_value = 1
onnx_model.graph.output[0].type.tensor_type.shape.dim[0].dim_value = 1
onnx.checker.check_model(onnx_model)
onnx.save(onnx_model, r'/Users/fl/github/DL-Compiler-Navigation/model/resnet50-v2-7-frozen.onnx')

model = tvmc.load("/Users/fl/github/DL-Compiler-Navigation/model/resnet50-v2-7-frozen.onnx")

print("frozen model saved")


# autotvm
# tvmc.tune(model, target="llvm")
# ansor
# tvmc.tune(model, target="llvm", enable_autoscheduler = True)
# 保存调优日志
tvmc.tune(model, target="llvm", tuning_records="records.log")


# use tuned log
package = tvmc.compile(model, target="llvm", tuning_records="records.log")
# 可以使用package_path参数保存编译后的模型，之后可以快速导入
# tvmc.compile(model, target="llvm", package_path="")
# new_package = tvmc.TVMCPackage(package_path="")



