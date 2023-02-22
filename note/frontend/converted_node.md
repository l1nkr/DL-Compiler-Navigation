<!-- todo  -->
<!-- 每次调用_convert_operator都需要重新_get_convert_map。但是_convert_operator通常需要执行很多次，也就是说需要执行很多次_get_convert_map，那么我们能不能只调用一次_get_convert_map呢 -->

onnx算子转化为 tvm relay ir


``_construct_nodes``中会调用``self._convert_operator``将onnx node转化为tvm relay ir

1. 首先获取算子映射表；

2. 如果算子在_identity_list表中，调用get_relay_op得到转换后的算子表达；

3. 如果在算子转换映射表中，调用映射接口转换算子；

4. 否则认为转换异常；

5. 返回转换后的表达式。

```python
def _convert_operator(self, op_name, inputs, attrs, opset):
    """Convert ONNX operator into a Relay operator.
    The converter must specify conversions explicitly for incompatible name, and
    apply handlers to operator attributes.

    Parameters
    ----------
    op_name : str
        Operator name, such as Convolution, FullyConnected
    inputs : list of tvm.relay.function.Function
        List of inputs.
    attrs : dict
        Dict of operator attributes
    opset : int
        Opset version

    Returns
    -------
    sym : tvm.relay.function.Function
        Converted relay function
    """
    convert_map = _get_convert_map(opset)
    if op_name in _identity_list:
        sym = get_relay_op(op_name)(*inputs, **attrs)
    elif op_name in convert_map:
        sym = convert_map[op_name](inputs, attrs, self._params)
    else:
        raise NotImplementedError("Operator {} not implemented.".format(op_name))
    return sym
```

映射表的_get_convert_map接口

- 一对一映射的时候，如果只有名字不同那么使用``Renamer``返回对应的tvm算子表示，再使用``AttrCvt``将onnx属性convert为tvm属性
- 一对多映射的时候(composed)，需要实现特定的转换函数
- 多对一，目前似乎还不支持

```python
# _convert_map defines maps of name to converter functor(callable)
# for 1 to 1 mapping, use Renamer if nothing but name is different
# use AttrCvt if attributes need to be converted
# for 1 to N mapping(composed), use custom callable functions
# for N to 1 mapping, currently not supported(?)
def _get_convert_map(opset):
    return {
        # defs/experimental
        "Identity": Renamer("copy"),
        "Affine": Affine.get_converter(opset),
        "BitShift": BitShift.get_converter(opset),
        "ThresholdedRelu": ThresholdedRelu.get_converter(opset),
        "ScaledTanh": ScaledTanh.get_converter(opset),
        "ParametricSoftplus": ParametricSoftPlus.get_converter(opset),
        "Constant": Constant.get_converter(opset),
        "ConstantOfShape": ConstantOfShape.get_converter(opset),
    ...
```



算子转换类时都是继承了OnnxOpConverter，每个需要转换的算子都有一个或者多个版本的转换接口，例如
```python

class OnnxOpConverter(object):
    """A helper class for holding onnx op converters."""

    @classmethod
    def get_converter(cls, opset):
        """Get converter matches given opset.

        Parameters
        ----------
        opset: int
            opset from model.

        Returns
        -------
        converter, which should be `_impl_vx`. Number x is the biggest
            number smaller than or equal to opset belongs to all support versions.
        """
        # dir(cls) 是在获取子类也就是算子的属性，找到包含字符串_impl_v的属性和方法，
        # 然后将字符串去掉，将剩余部分换成int类型
        versions = [int(d.replace("_impl_v", "")) for d in dir(cls) if "_impl_v" in d]
        # 将传入的opset的版本号加入version表中，并从小到大排序
        versions = sorted(versions + [opset])
        # 找到和opset相等或最接近opset的版本号
        version = versions[max([i for i, v in enumerate(versions) if v == opset]) - 1]
        # 返回该版本的_impl_v方法
        if hasattr(cls, "_impl_v{}".format(version)):
            return getattr(cls, "_impl_v{}".format(version))
        raise NotImplementedError(
            "opset version {} of {} not implemented".format(version, cls.__name__)
        )

```
```python
class Conv(OnnxOpConverter):
    """Operator converter for Conv."""
 
    @classmethod
    def _impl_v1(cls, inputs, attr, params):
        # Use shape of input to determine convolution type.
        # 从传入的inputs参数中获取输入和卷积核数据,并推导各自的形状
        data = inputs[0]
        kernel = inputs[1]
        input_shape = infer_shape(data)
        ndim = len(input_shape)
 
        kernel_type = infer_type(inputs[1])
        kernel_shapes = [get_const_tuple(kernel_type.checked_type.shape)]
        # 如果onnx卷积属性中没有给出卷积核的形状,就使用inputs里面推导出来的形状
        if "kernel_shape" not in attr:
            attr["kernel_shape"] = kernel_shapes[0][2:]
        # 如果onnx卷积算子设置了auto_pad属性
        if "auto_pad" in attr:
            # 对用的tvm卷积算子也使用onnx设置的auto_pad属性值
            attr["auto_pad"] = attr["auto_pad"].decode("utf-8")
            # 根据auto_pad属性值对数据进行填充处理
            if attr["auto_pad"] in ("SAME_UPPER", "SAME_LOWER"):
                # Warning: Convolution does not yet support dynamic shapes,
                # one will need to run dynamic_to_static on this model after import
                # 对输入数据进行填充,得到填充后的数据
                data = autopad(
                    data,
                    attr.get("strides", [1] * (ndim - 2)),
                    attr["kernel_shape"],
                    attr.get("dilations", [1] * (ndim - 2)),
                    mode=attr["auto_pad"],
                )
            elif attr["auto_pad"] == "VALID":
                attr["pads"] = [0 for i in range(ndim - 2)]
            elif attr["auto_pad"] == "NOTSET":
                pass
            else:
                msg = 'Value {} in attribute "auto_pad" of operator Conv is invalid.'
                raise tvm.error.OpAttributeInvalid(msg.format(attr["auto_pad"]))
            attr.pop("auto_pad")
 
        attr["channels"] = kernel_shapes[0][0]
        out = AttrCvt(
            # 返回的op_name是一个函数,返回当前算子对应的tvm算子名称.在AttrCvt.__call__方法中调用该函数，根据当前attr中kernel_shape
            # 属性得到对应的TVM conv1d/conv2d/conv3d算子接口;然后算子接收([data, kernel], attr, params)
            # 参数, 返回转换后的TVM表示out
            op_name=dimension_picker("conv"),
            # 参数转换表
            transforms={
                # 当前属性名 : 转换后的属性名
                "kernel_shape": "kernel_size",
                # 当前属性名 : (转换后的属性名, 转换后的默认值)
                "dilations": ("dilation", 1),
                # 当前属性名 : (转换后的属性名, 转换后的默认值)
                "pads": ("padding", 0),
                # 当前属性名 : (转换后的属性名, 转换后的默认值)
                "group": ("groups", 1),
            },
            custom_check=dimension_constraint(),
        )([data, kernel], attr, params)
 
        use_bias = len(inputs) == 3
        # 如果输入中有偏置参数,则在表达式中添加偏置运算
        if use_bias:
            out = _op.nn.bias_add(out, inputs[2])
        return out
```

AttrCvt.__call__方法大致流程是对参数进行检查，转换，然后调用get_relay_op得到算子对应的tvm接口函数，将当前算子的输入和变换后的参数输入接口，得到onnx node对应的tvm relay ir。

```python
class AttrCvt(object):
    def __init__(
        self,
        op_name,
        transforms=None,
        excludes=None,
        disables=None,
        ignores=None,
        extras=None,
        custom_check=None,
    ):
        # 算子的新名字,op_name可以是一个字符串,也可以是一个返回字符串的函数
        self._op_name = op_name
        # 属性转换表,表项为属性转换字典,形式为"attr_name : new_attr_name", 
        # 或者"attr_name : (new_name, default_value, transform function)"
        self._transforms = transforms if transforms else {}
        # 不允许出现的属性集合,如果出现会抛出异常
        self._excludes = excludes if excludes else []
        # 转换后会被disable的属性集合
        self._disables = disables if disables else []
        # 转换过程中会被忽略的属性集合
        self._ignores = ignores if ignores else []
        # 转换后会被额外返回的属性
        self._extras = extras if extras else {}
        # 转换执行的检测函数,返回False会抛出异常
        self._custom_check = custom_check
 
    def __call__(self, inputs, attrs, *args):
        # 忽略待转换算子的这些属性
        self._ignores.append("_output_shapes")
        self._ignores.append("_input_shapes")
        self._ignores.append("T")
        self._ignores.append("use_cudnn_on_gpu")
        self._ignores.append("_node_name")
        self._ignores.append("is_training")
        self._ignores.append("_target_layout")
 
        # apply custom check
        # 如果算子转换传入了检测函数,则执行该检测函数
        if self._custom_check:
            func, msg = self._custom_check
            if not func(attrs):
                raise RuntimeError("Check failed: {}".format(msg))
        # get new op_name
        # 得到算子转换后的名字
        if isinstance(self._op_name, str):
            op_name = self._op_name
        else:
            assert callable(self._op_name), "op_name can either be string or callable"
            op_name = self._op_name(attrs)
 
        # ignore 'tvm_custom' always
        # 忽略tvm_custom属性
        self._ignores.append("tvm_custom")
 
        # convert attributes
        new_attrs = {}
        # 遍历传入的待转换算子的属性
        for k in attrs.keys():
            # 如果属性在排除表中, 抛出异常
            if k in self._excludes:
                raise NotImplementedError(
                    "Attribute %s in operator %s is not" + " supported.", k, op_name
                )
            # 如果属性是要求disable的,打印debug日志
            if k in self._disables:
                logger.debug("Attribute %s is disabled in relay.sym.%s", k, op_name)
            # 如果属性是要求忽略的,打印debug日志
            elif k in self._ignores:
                if k != "tvm_custom":
                    logger.debug("Attribute %s is ignored in relay.sym.%s", k, op_name)
            # 如果属性在转换表中
            elif k in self._transforms:
                # 从转换表中该属性对应的转换dict,得到属性的新名字,新默认值和转换操作函数
                # 如果转换表中没有给出转换函数,则将转换函数设置为lambda x: x,也就是直接返回参数
                new_name, defaults, transform = self._parse_default(self._transforms[k])
                # 如果没有给出默认值
                if defaults is None:
                    # 那么必须是"attr_name:new_attr_name"形式,获取新属性名
                    new_attr = self._required_attr(attrs, k)
                else:
                    # 从原始的属性表中查找该属性的值,如果没找到,则为新属性为None
                    new_attr = attrs.get(k, None)
                if new_attr is None:
                    # 如果新属性为None,在新的属性表中添加该属性,值为转换表中得到的默认值
                    new_attrs[new_name] = defaults
                else:
                    # 在新的属性表中添加该属性,调用转换函数得到新的属性值
                    new_attrs[new_name] = transform(new_attr)
            else:
                # copy
                # 如果属性不在转换表中,直接原封不动的加入新属性表
                new_attrs[k] = attrs[k]
        # add extras
        # 更新额外的属性
        new_attrs.update(self._extras)
        # 将输入和新属性表传入算子转换接口,返回转换后tvm relay ir
        return get_relay_op(op_name)(*inputs, **new_attrs)

```

这里get_relay_op(conv2d)将返回nn.conv2d。nn.conv2d代码如下：

```python

def get_relay_op(op_name):
    ...
    for candidate in (_op, _op.nn, _op.image, _op.vision, _op.contrib):
            op = getattr(candidate, op_name, None)
            if op is not None:
                break
    ...

# conv2d位于_op.nn中，_op,nn导入了.nn，.nn中对conv2d进行了实现
def conv2d(
    data,
    weight,
    strides=(1, 1),
    padding=(0, 0),
    dilation=(1, 1),
    groups=1,
    channels=None,
    kernel_size=None,
    data_layout="NCHW",
    kernel_layout="OIHW",
    out_layout="",
    out_dtype="",
):
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if isinstance(strides, int):
        strides = (strides, strides)
    if isinstance(dilation, int):
        dilation = (dilation, dilation)
    # TODO enforce 4-way padding in topi/nn/conv2d after #4644 merged
    # convert 2-way padding to 4-way padding
    padding = get_pad_tuple2d(padding)
    return _make.conv2d(
        data,
        weight,
        strides,
        padding,
        dilation,
        groups,
        channels,
        kernel_size,
        data_layout,
        kernel_layout,
        out_layout,
        out_dtype,
    )

```

_make.conv2d会调用到C++代码src/relay/op/nn/convolution_make.h中实现的MakeConv接口：
```c++
template <typename T>
inline Expr MakeConv(Expr data, Expr weight, Array<IndexExpr> strides, Array<IndexExpr> padding,
                     Array<IndexExpr> dilation, int groups, IndexExpr channels,
                     Array<IndexExpr> kernel_size, std::string data_layout,
                     std::string kernel_layout, std::string out_layout, DataType out_dtype,
                     std::string op_name) {
  auto attrs = make_object<T>();
  attrs->strides = std::move(strides);
  attrs->padding = std::move(padding);
  attrs->dilation = std::move(dilation);
  attrs->groups = groups;
  attrs->channels = std::move(channels);
  attrs->kernel_size = std::move(kernel_size);
  attrs->data_layout = std::move(data_layout);
  attrs->kernel_layout = std::move(kernel_layout);
  attrs->out_layout = std::move(out_layout);
  attrs->out_dtype = std::move(out_dtype);
  const Op& op = Op::Get(op_name);
  return Call(op, {data, weight}, Attrs(attrs), {});
}
```

回到_convert_operator

在python/tvm/relay/frontend/onnx.py中，_identity_list表为空，所以_convert_operator中这个分支是走不到的。所有支持的框架里面，只有mxnet里面该表不为空（因为某些算子的属性转换限制，所以单列了这些算子到_identity_list）

get_relay_op的if分支检查下传入的op_name是不是用点号形式给出的，比如relay.op.abs；else分支是到nn，image， vision，contrib目录下去找是否有名为op_name的算子。

两个分支下，任一找到，都会返回算子的定义接口。所以返回的是跟传入的op_name同名的函数地址。例如op_name为abs时，对应的函数定义（python/tvm/relay/op/tensor.py）：

```python
# 注意这里的get_relay_op和上面是同一个函数，但是并不是在同一个地方调用的
def get_relay_op(op_name):
    """Get the callable function from Relay based on operator name.
    Parameters
    ----------
    op_name : str
        The Relay operator name.
    """
    if "." in op_name:
        # explicit hierarchical modules
        op = _op
        try:
            for opn in op_name.split("."):
                op = getattr(op, opn)
        except AttributeError:
            op = None
    else:
        # try search op in various modules
        for candidate in (_op, _op.nn, _op.image, _op.vision, _op.contrib):
            op = getattr(candidate, op_name, None)
            if op is not None:
                break
    if not op:
        raise tvm.error.OpNotImplemented("Unable to map op_name {} to relay".format(op_name))
    return op
```

[refer](https://blog.csdn.net/zx_ros/article/details/122917673)
[refer](https://blog.csdn.net/zx_ros/article/details/123526147)
[关于c++部分的代码讲解](https://blog.csdn.net/zx_ros/article/details/123130815)
[关于c++部分的代码讲解](./register_op.md)

1. 当tvm中实现一个算子时，会调用 RELAY_REGISTER_OP进行注册；

2. 该注册会在 AttrRegistry<OpRegEntry, Op>（这是个单例模式的类）的entry_map_中加入一个OpRegEntry实例；

3. 而tvm处理一个外部输入的模型时，如果遇到这个算子，在Op::Get方法中从entry_map_表中读取对应的OpRegEntry实例：



