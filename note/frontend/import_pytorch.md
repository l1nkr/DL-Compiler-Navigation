## pytorch前端到tvm.IRModule

```
调用流程：relay.frontend.from_pytorch
1 tvm.IRModule(初始化容器，用于保存relay的信息)
2 Prelude(加载辅助函数)
  2.1 import_from_std(加载基础函数)
  2.2 tensor_array_ops.register(加载tensorarray相关函数)
3 PyTorchOpConverter(构建converter，用于算子解析)
4 create inputs && params
  4.1 _get_relay_input_vars(构建inputs)
  4.2 convert_params(构建params)
5 converter.convert_operators(转换算子)
6 set the IRModule
  6.1 analysis.free_vars(确定无依赖参数，例如inputs，params)
  6.2 tvm.relay.Function(用Function包装DAG计算过程)
  6.3 transform.RemoveUnusedFunctions(简单优化去除无用代码)
```

[参考](https://zhuanlan.zhihu.com/p/457039705)