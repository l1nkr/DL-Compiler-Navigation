from tvm.driver import tvmc

# convert to relay
model = tvmc.load("/Users/fl/github/tvm_tutorial/Getting_Starting_using_TVMC_Python/resnet50-v2-7.onnx")


# save time, do not need to convert
model.save("/Users/fl/github/tvm_tutorial/Getting_Starting_using_TVMC_Python/convertedModel")


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



