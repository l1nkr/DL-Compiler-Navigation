添加一个 pass 所需要的前置知识
- 了解 IR node 属性
- 了解 Visitor 设计模式
- 知道 Schedule 是如何被 lowing 到 IRModule 或者 LLVM Module

pass 其实就是将一个表达式等价的转换为另一个表达式。TVM 提供了两个类给用户用于 IR 的分析以及转换。


#### IR Visitor
我们可以使用`tvm.tir.stmt_functor.post_order_visit(stmt, func)`从IR中获取信息。



#### IR Transformation





[ref](https://tvm.apache.org/docs/how_to/extend_tvm/low_level_custom_pass.html)