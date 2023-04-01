# pass

## 初始 pass

```python
with tvm.transform.PassContext(opt_level=3):
    lib = relay.build(func, "llvm", params=params)
```

实际上 `tvm.transform.PassContext` 这个接口就定义了一个 Pass

Pass 是 TVM 中基于 Relay IR 进行的一系列优化，类似于 onnx-simplifier 里面用到的 onnxoptimizer，它可以简化计算图，去除一些冗余的算子，提高模型的推理效率。

主要包含 **PassContext，PassInfo，Pass，以及 Sequential**
- **PassContext** 是上面 Python 接口对应的 C++ 实现，它包含了 Pass 执行依赖的一些参数（如优化 level），依赖的其它特定 Pass 以及设置不使用某种指定 Pass 等。
- **PassInfo** 是用来记录 Pass 信息的类，包含 opt_level、name、当前 Pass 需要哪些前置 Pass。
- **Pass** 执行 pass 的主体，这是一个基类。
- **Sequential** 是一个 container，用于装载 Pass。

## pass infrastructure

Relay 和 TVM IR 都包含一系列优化 passes，可提高模型的性能指标。TVM 有一套标准优化方法以及特定于机器学习的优化方法，包括常量折叠、死代码消除、运算符布局更改、算符融合、缓冲区处理和循环变换等。每一个 Pass 都使用在 traversal 期间和 / 或之前收集的分析结果

随着 TVM 的迅速发展，需要一个可以跨 TVM 不同层（如 Relay 和 tir）的 passes 的通用框架来管理这些 passes。许多现有的编译器（GCC、LLVM），采用 **pass manager** 来管理 passes 的执行。现代深度学习框架（如Pytorch、MXNet Gluon），也有分别通过 Sequential 和 Block 启用 pass-style 层构建方案的趋势。

**Relay pass infra** 的设计很大程度上受到 LLVM 中使用的分层 pass manager 和流行的深度学习框架中使用的 block-style 容器的启发。 pass infra 的主要目标包括：

- 编排 optimizer。允许用户灵活地定制 pass 管道。
- 提供一种用户友好的方式来调试 passes。
- 减轻 passes 之间的依赖关系。
- 降低实现新 passes 的难度。

### python frontend

#### PassContext

Python 前端为 `PassContext` 提供了一个包装器，通过覆盖 `__enter__` 和 `__exit__` 来启用 with 语法。 为用户提供了一个 `current` 静态方法来获取在特定范围内使用的上下文。

PassContext 用于配置编译选项，包括优化级别和必需 / 禁用的 pass。 它还可以带一个配置字典，以便不同的 pass 可以方便地获取 passed 的数据，例如回退设备信息和循环展开的步数 / 深度等。 为了能够获取所需的配置，必须通过`TVM_REGISTER_PASS_CONFIG_OPTION`注册关键字。

```python
@tvm._ffi.register_object("transform.PassContext")
class PassContext(tvm.runtime.Object):
    def __enter__(self):
        _transform.EnterPassContext(self)
        return self

    def __exit__(self, ptype, value, trace, config):
        _transform.ExitPassContext(self)

    @staticmethod
    def current():
        """Return the current pass context."""
        return _transform.GetCurrentPassContext()
```

#### Pass Objects

Pass 是所有 pass 对象的基类。 这里的所有方法都只是简单的包装器，仅仅为了api易于使用。 在 pass 基类中只定义了一个__call__来使子类成为可调用对象，以便它们可以很容易地被调用（例如 pass_xx(arg)）来执行。

```python
@register_relay_node
class Pass(RelayNode):
   def __call__(self, mod):
       return _transform.RunPass(self, mod)
```

还有一些辅助 APIs 支持从 Python 前端创建 pass 并让 pass infra 控制执行。 比如module_pass、function_pass、sequential，可以定义自己的 pass 或者 pass 管道。

可以通过装饰器像下面这样构建一个 pass：
```python
 @relay.transform.module_pass(opt_level=2)
 def transform(mod, ctx):
    tp = relay.TensorType((10,), "float32")
    x = relay.var("x", tp)
    gv = relay.GlobalVar("abs")
    func = relay.Function([x], relay.abs(x))
    new_mod = relay.Module({gv: func})
    new_mod.update(mod)
    return new_mod

module_pass = transform
assert isinstance(module_pass, transform.ModulePass)
assert module_pass.info.opt_level == 2
```

这里的transform函数向输入的 module 添加了一个abs 函数，创建此 module_pass 后，用户可以将其应用于任何 Relay 模块。 例如，我们可以构建一个 module 并应用此 pass 来添加 abs 函数。

```python
mod = relay.Module()
mod = module_pass(mod)
```

### c++ backend

#### PassInfo

```c++
class PassInfoNode : public Object {
  String name;
  int opt_level;
  Array<String> required;
};
```

提供了 `PassInfo` 对象来包含一个 pass 所需的基本信息。 
- `name` 是 pass 名称
- `opt_level` 指示将启用 pass 的优化级别。
  - 可用于帮助 pass infra 决定是否需要执行某个 pass。
- `required` 表示执行某个 pass 所需的 pass。
  - `required` 字段可以让 pass infra 解决 pass 依赖关系。

#### PassContext

`PassContext` 带有用于调试 pass 的有用信息。 
- 例如，它包含错误报告系统，pass 的作者可以提供有关优化失败原因的注释。 
- `PassContext` 还旨在替换旧的 `BuildConfig`，它用于帮助用户配置编译选项，包括优化级别、依赖和需要禁用的 pass 等。

#### PassConstructs

`pass infra` 是以分层方式设计的，它可以在不同粒度下工作(Relay/tir)。

因此引入了纯虚类 `PassNode` ，此类包含几个必须由子类在 modules, functions, or sequences of passes 层次实现的虚函数。

```c++
class PassNode : Object {
  virtual PassInfo Info() const = 0;
  virtual Module operator()(const IRModule& mod
                            const PassContext& pass_ctx) const = 0;
};
```

- **Module-Level Passes**
```c++
class ModulePassNode : PassNode {
  PassInfo pass_info;
  runtime::TypedPackedFunc<Module(Module, PassContext)> pass_func;
  Module operator()(const Module& mod, const PassContext& pass_ctx) const final;
};
```

Module Level Passes 主要用于全局和过程间优化 (IPO)。

  - *pass_info* 维护 module-level pass 所需的信息。 
  - *pass_func* 实现了真正的 optimization。 例如，对 module 执行死代码消除，可以在 pass_func 中实现算法并让它在 module 上运行。他被设计为一个 packed function，所以这个优化不仅可以使用 C++ 还可以使用 Python 来实现。

- **Function-Level Passes**
```c++
class FunctionPassNode : PassNode {
  PassInfo pass_info;
  runtime::TypedPackedFunc<Function(Function, Module, PassContext)> pass_func;
  Module operator()(const Module& mod, const PassContext& pass_ctx) const final;
  bool SkipFunction(const Function& func) const;
};
```
Function-level passes 用于为给定的 Relay/tir module 各种内部函数进行优化。 它从 module 的函数列表中获取一个函数进行优化，并生成一个重写的 Relay Function 或 tir PrimFunc。 大多数 pass 可以归入这一类。

此 passes 范围是 **Relay Function** 或 **tir PrimFunc**。 因此，无法通过这些 passes 添加或删除函数，因为不知道全局信息。

  - *pass_info*。 与 Module-Level pass 中描述的相同。 即维护 Function-level pass 所需的信息。 
  - *pass_func*。 需要一个函数进行优化，它还需要一个 Module，因为我们可能会使用它来报告错误。 
  - *SkipOptimization*。一个函数可以用 “SkipOptimization” 注释，以便在优化过程中被忽略。

- **Sequential Passes**
```c++
class SequentialPassNode : PassNode {
  PassInfo pass_info;
  // Passes need to be executed.
  Array<Pass> passes;
  bool PassEnabled(const PassInfo& info) const;
  Module operator()(const Module& mod, const PassContext& pass_ctx) const final;
};
```
SequentialPass 类似于 Pytorch nn.Sequential，它包含许多用于执行的 passes。目前在 Relay 中只有少数 passes 被放入这组中。

```c++
Module SequentialNode::operator()(const Module& module,
                                  const PassContext& pass_ctx) const {
  Module mod = module;
  for (const Pass& pass : passes) {
    ICHECK(pass.defined()) << "Found undefined pass for optimization.";
    const PassInfo& pass_info = pass->Info();
    if (!PassEnabled(pass_info))  continue;
    for (const auto& it : pass_info->required) {
      const auto* name = it.as<tvm::ir::StringImm>();
      ICHECK(name);
      mod = GetPass(name->value)(mod, pass_ctx);
    }
    mod = pass(mod, pass_ctx);
  }
  return mod;
}
```
在调用 pass 时，我们首先检查是否启用了此 pass。 这是通过首先检查用户是否明确禁用该 pass，然后检查它是否被用户指定为必需 pass 来完成的。 如果仍然不确定是否启用了此传递，则将检查其 opt_level。 只有当它的opt_level不低于 pass context 中配置的优化级别时，才会启用并因此执行此 pass。

```c++
Pass GetPass(const std::string& pass_name) {
  using tvm::runtime::Registry;
  std::string fpass_name = "relay._transform." + pass_name;
  const auto* f = Registry::Get(fpass_name);
  ICHECK(f != nullptr) << "Cannot find " << fpass_name
                      << "to create the pass " << pass_name;
  return (*f)();
}
```
要执行 pass，要求 pass name 已经在 TVM packed function 注册表中完成注册。

```c++
Pass CreateFunctionPass();
Pass CreatePrimFuncPass();
Pass CreateModulePass();
Pass Sequential(tvm::Array<Pass> passes, PassInfo pass_info);
```
提供了一些 helper function 来创建上述每种类型的 Pass。 这些 helper function 也暴露给 Python 前端，以便用户可以方便地使用 Python API 来创建特定的 pass 对象

#### Pass Registration

下面以注册 *constant folding* 为例。

```c++
Pass FoldConstant() {
  runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func =
    [=](Function f, IRModule m, PassContext pc) {
      return Downcast<Function>(FoldConstant(f));
  };
  return CreateFunctionPass(pass_func, 2, "FoldConstant", {});
}
```

为了将一个 pass 注册到 pass infra，首先需要决定这个 pass 将在哪个级别执行。 由于常量折叠发生在单个函数上，应该通过 *CreateFunctionPass* 为其创建一个 FunctionPass。 pass_func 作为 packed function 返回，该函数在 IRModule 中的每个 function 上调用 Expr to Expr API。 {} 表示此 pass 不需要先决条件。 否则，pass 开发人员必须识别并列出它们。

```c++
TVM_DLL Pass FoldConstant();
```

为了允许其他 C++ 模块应用此 pass，需要进行以上声明。

参考
- http://www.giantpandacv.com/project/%E9%83%A8%E7%BD%B2%E4%BC%98%E5%8C%96/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E7%BC%96%E8%AF%91%E5%99%A8/%E3%80%90%E4%BB%8E%E9%9B%B6%E5%BC%80%E5%A7%8B%E5%AD%A6%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E7%BC%96%E8%AF%91%E5%99%A8%E3%80%91%E4%B8%83%EF%BC%8C%E4%B8%87%E5%AD%97%E9%95%BF%E6%96%87%E5%85%A5%E9%97%A8TVM%20Pass/