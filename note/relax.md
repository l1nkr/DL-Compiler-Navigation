# 非官方relax详解

注意：“《》”中的文本表示当前原型中不存在的功能。

为了开发和测试 Relax，编译器开发人员必须就 Relax 中给定程序的含义以及使其有效的原因达成一致，以便可以独立于任何特定 Relax 实现评估测试用例。 本文档旨在描述 Relax 的语法结构（抽象语法树）、语法语义（不同结构的含义）、Relax 的类型系统和类型检查规则（什么使 Relax 程序有效），以及以详细但仍然非正式的方式推理结构信息（例如张量形状）的规则。 如有必要，我们可能会更正式地对这些规则进行编码，以允许进行更自动化的分析。

虽然本文档将使用 TVMScript 前端作为一些示例，但指定从 Python 的 AST 到 Relax 的 AST 的映射将被推迟，直到 parser 变得更加稳定。

# Table of Contents

- [非官方relax详解](#非官方relax详解)
- [Table of Contents](#table-of-contents)
- [Overview](#overview)
  - [Differences from Relay](#differences-from-relay)
  - [Grammer](#grammer)
    - [Notes on `derive_func`](#notes-on-derive_func)
    - [Notes on `DataType` and Related Terminology](#notes-on-datatype-and-related-terminology)
  - [Expression Survey](#expression-survey)
  - [Purity and Dataflow Blocks](#purity-and-dataflow-blocks)
  - [Structural Information (`StructInfo`) System Survey](#structural-information-structinfo-system-survey)
- [Top-level Program Organization: `IRModule`](#top-level-program-organization-irmodule)
- [Values in Relax](#values-in-relax)
  - [Representation of Values at Run Time](#representation-of-values-at-run-time)
- [Variable Scoping](#variable-scoping)
- [Normal Form](#normal-form)
- [Well-Formedness Criteria](#well-formedness-criteria)
- [Structural Information (`StructInfo`) in Relax](#structural-information-structinfo-in-relax)

# Overview

本节将概述 Relax 的语法，并对不同的组件进行非常简短的描述，包括语义和结构信息 (StructInfo) 系统。 本文档的其余部分将更详细地描述语言的这些方面，包括 StructInfo 系统支持的有效性条件。

## Differences from Relay

根据最初的研讨会论文和后来的报告，Relay 被设计为一种高级函数式语言，用于在高层次上表达深度学习模型。 虽然 Relay 不是完全纯的（`Ref` 类型是根据 SML 和类似函数式语言中的引用类型建模的），但 Relay 中的假设是张量运算符通常是纯的，**这意味着它们不会改变程序状态**，除非产生新值 . 此外，Relay 的类型系统还要求运算符具有类型关系，以推断静态张量类型或得出编译时维度未知的结论 (`Any`)。 注册类型关系和确保运算符纯度的需要使得向 Relay 添加新运算符变得困难，尤其难以直接调用 TIR 或外部库，这些库通常是不纯净的； 任何此类扩展都需要添加新的运算符并抽象出任何杂质。

虽然 Relax 的目标是像 Relay 一样通用和富有表现力，但 Relax 旨在使其更容易与外部库进行互操作，尤其是与 TIR。 特别是，Relax 包括调用任意 TVM `PackedFuncs`（可以调用外部库）的机制和对 TIR 的特殊支持。 因此，该语言并不假定此类操作是纯粹的，尽管这确实需要对别名和类似问题进行推理。 此外，在类型检查期间不再处理张量形状； 除了类型之外，每个表达式都有与其关联的相关结构信息。 在许多情况下，这种结构信息支持关于张量形状的静态推理，但也有助于在不可能时回退到动态检查。 这种形状方法允许在运行时检查更丰富的形状约束和其他结构属性（例如 _symbolic_ shapes，其中某些尺寸是变量），并允许通过避免 需要类型关系。

## Grammer

下面是 Relax 中各种 AST 结构的图表，包括类型。该图将给出 AST 节点的名称及其成员的类型和名称。语义将描述每个构造代表什么计算； AST 只是数据。 Relax 程序由一个 IRModule 组成，该 IRModule 具有绑定到实现感兴趣计算的 Relax 函数的全局变量。

在符号上：
- `[x]` 表示“x 的列表”
- `x?` 表示“可选的 x”
- `{x: y}` 表示“x 到 y 的映射”
- `x | y` 表示“x 或 y”
- `#` 用于注释。

```
# PrimExprs are defined in TIR, see include/tvm/tir/expr.h
# They are intended to have the same semantics as in TIR
PrimExpr ::=
           Var(name: string) # shape variables
         | IntImm(value: int64)
         | Add(a: PrimExpr, b: PrimExpr)
         | Sub(a: PrimExpr, b: PrimExpr)
         | Mul(a: PrimExpr, b: PrimExpr)
         | Div(a: PrimExpr, b: PrimExpr)
         | Min(a: PrimExpr, b: PrimExpr)
         | Max(a: PrimExpr, b: PrimExpr)
         | Not(a: PrimExpr)
         | And(a: PrimExpr, b: PrimExpr)
         | Or(a: PrimExpr, b: PrimExpr)
         | Select(condition: PrimExpr, true_value: PrimExpr, false_value: PrimExpr)
         # (others may be added later, as deemed necessary)

# Also from TIR
DataType ::= Int(bits: int, lanes: int)
           | UInt(bits: int, lanes: int)
           | Float(bits: int, lanes: int)
           | Handle(bits: int, lanes: int)

StructInfo ::= TensorStructInfo(shape: Expr?, dtype: DataType, ndim: int)
             | ShapeStructInfo(values: [PrimExpr]?, ndim: int)
             | PrimStructInfo(dtype: DataType)
             | ObjectStructInfo()
             | TupleStructInfo(fields: [StructInfo])
             | FuncStructInfo(params: [StructInfo]?, ret: StructInfo, derive_func: EnvFunc?*)

# expressions
Expr ::=   Constant(data: NDArray)
           # scoped to functions or SeqExprs
         | Var(name_hint: string, struct_info_annotation: StructInfo?)
           # scoped to DataflowBlocks
         | DataflowVar(name_hint: string, struct_info_annotation: StructInfo?)
         | GlobalVar(name_hint: string)
         | Tuple(fields: [Expr])
         | SeqExpr(blocks: [BindingBlock], body: Expr)
         | PrimValue(value: PrimExpr)
         | StringImm(value: string)
         | DataTypeImm(value: DataType)
         | Function(params: [Var], body: Expr, ret_struct_info: StructInfo?, attrs: Attrs?)
         | If(cond: Expr, true_branch: Expr, false_branch: Expr)
         | ExternFunc(global_symbol: string)
         | Call(op: Expr, args: [Expr], sinfo_args: [StructInfo], attrs: Attrs?)
         | ShapeExpr(values: [PrimExpr])
         | TupleGetItem(tuple_value: Expr, index: int)
         | Op(op_name: string)

# binding blocks (analogous to sequence of statements)
BindingBlock ::= 
           BindingBlock(bindings: [Binding])
         | DataflowBlock(bindings: [Binding])

# bindings (analogous to statements)
Binding ::= 
           VarBinding(var: Var|DataflowVar, value: Expr)
         | MatchCast(var: (Var|DataflowVar)?, struct_info: StructInfo, value: Expr)

# Relax programs are IRModules. Modules may bind global variables either to
# Relax functions or TIR PrimFuncs (specified separately).
# The Relax compiler may analyze and modify the TIR PrimFUncs as well.
Program ::= IRModule(funcs: {GlobalVar: Function|PrimFunc})
```

### Notes on `derive_func`

FuncStructInfo 的 derive_func 字段是元语言中的一个宏：给定一个函数调用和变量映射上下文，返回结果的 StructInfo。 该字段仅在编译时用于推理调用 ExternFuncs 的 StructInfo。

### Notes on `DataType` and Related Terminology

上述 AST 中数据类型 DataType 的表示直接取自 TIR。 然而，数据类型在 Relax 中的使用比在 TIR 中更受限制。

1. Int、UInt 和 Float 数据类型的通道字段必须始终为 1； 我们不直接考虑 Relax 中的向量化值。
2. Handle 数据类型的 lanes 字段必须始终为 0，表示它是 Void（见下文）。 Handle 的位字段应始终设置为 64（Relax 不会使用它）。

我们还为数据类型定义了以下特殊符号，以用于规范的其余部分：

1. Bool()：这是 UInt(bits=1, lanes=1) 的简写，因为 TIR 没有单独的布尔类型。 在此数据类型中，“True”表示值为 1，“false”表示值为 0。为方便起见，我们将在规范中将布尔值称为单独的数据类型，因为它们在 If 节点中很重要。
2. Void()：这是 Handle(bits=64, lanes=0) 的简写。 TIR 使用此数据类型来指代不透明对象； 在 Relax 中，它用于表示未知数据类型。

## Expression Survey

本规范更详细地描述了每个表达式和 StructInfo 代表的内容以及使它们有效的条件。
1. `Constant` 节点构造张量常数（标量的 n 维数组）。
2. `Tuple` 节点构造 Relax 值的元组（不可变的固定大小有序分组）。
3. `Var`、`DataflowVar`和`GlobalVar`节点都是变量，指的是不同种类的命名存储值。 Relax 中的变量必须恰好绑定一次。
   1. `GlobalVar` 绑定在 `IRModule` 本身中，并引用 Relax 函数或 TIR `PrimFunc`。 
   2. `Var` 节点绑定在函数内，它们代表函数参数，或者绑定在 `BindingBlock` 中的 `VarBinding` 或 `MatchCast` 节点中，我们将在下面讨论。 
   3. `DataflowVar` 与 `Var` 类似，只能在 `DataflowBlock` 中绑定。
4. `PrimExpr` 用于表示 `ShapeExpr` 和 `MatchCast` 节点中形状的尺寸。 这些表示对具有自己的 `Var` 节点 (`tir::Var`) 的整数的操作，我们将其称为“形状变量”。 形状变量只能在其他 `PrimExpr` 中使用，并且作用域类似于 `Var` 节点 (`relax::Var`)，我们将其称为“Relax 变量”。
5. `ExternFunc` 节点评估为 `PackedFunc`； 该实现将通过其全局符号查找已注册的 PackedFunc。
6. `PrimValue` 节点从 `PrimExpr` 构造不可变标量值，主要用于与 `ExternFunc` 或运算符交互。 这些标量被封装在 TVM 对象中，允许它们嵌套在 TVM 的容器中。 （相比之下，通过“Constant”定义的零维张量是可变的。）
7. `StringImm` 节点构造字符串，主要用于与 `ExternFunc` 或运算符交互。
8. `DataTypeImm` 节点构造 TIR 数据类型的表示，主要用于与 `ExternFunc` 或运算符交互（例如，对于将数据类型作为输入的 TIR 内在函数）。
9.  `Call` 节点表示函数调用。 被调用方参数（`op`）可以是 `ExternFunc` 节点（表示对 `PackedFunc` 的调用）、`Op` 节点（表示对 Relax 运算符的调用）或任意表达式。
     1. `Op` 节点是指内置的 Relax 运算符，编译器可以根据需要自由实现。 某些运算符会实现重要的操作，例如 `call_tir`（允许调用 TIR `PrimFunc`）。
     2. 任何其他表达式的计算结果必须为 `PackedFunc` 或闭包； 然后将使用给定的参数调用 op 的评估结果。
    
     对 ExternFunc 和运算符的调用可能会产生副作用，因此推断是否允许在 DataflowBlock 中进行函数调用很重要。
10. `If` 节点代表分支控制流。 首先评估条件表达式，它必须评估为布尔标量。 如果条件为真，则计算真分支并使用其结果； 否则，将评估错误分支并使用其结果。
11. `TupleGetItem` 节点表示元组索引。 `tuple_value` 表达式必须评估为至少包含 `index + 1` 项的元组，并且将返回具有给定索引的项。
12. `SeqExpr` 描述了一系列绑定块，后跟一个返回表达式。 `SeqExpr` 打开一个新的范围。 它的绑定块按顺序评估并将新变量添加到范围。 绑定块是普通的“BindingBlock”或“DataflowBlock”，两者都由一系列绑定组成。 `DataflowBlock` 是唯一允许引入与 `DataflowVar` 绑定的类型，它不允许任何具有控制流（`If` 节点或递归调用）或调用（可能）不纯函数的结构。 有两种不同类型的绑定：
     1. `VarBinding`s：首先计算绑定的`value`表达式（绑定的右侧）并绑定到`var`表达式，它必须是一个新的`Var`或`DataflowVar` （在数据流块中）。 新绑定的变量将在范围的其余部分具有该值（`DataflowVar` 的范围仅限于它们出现在其中的 `DataflowBlock`；`Var` 的范围为整个 `SeqExpr`）。
     2. `MatchCast`s：评估`value`表达式，并根据`struct_info`字段中给出的结构信息动态检查结果。
         1. 类型必须匹配：所有`StructInfo`变体都对应一个value值的类别（`TensorStructInfo`对应张量值，`ShapeStructInfo`对应形状值等），所以如果`value`的结构不对应 到 `struct_info`，会触发错误。 `value` 的结构与 `struct_info` 进行递归比较，因此 `value` 的所有组件必须与任何嵌套的结构信息相匹配。 特殊比较规则：
             1. 为了将张量值与 `TensorStructInfo` 进行比较，`ndim` 必须匹配张量值中的维数（除非 `ndim` 为 -1）并且 `dtype` 必须匹配使用的数据类型（除非 `dtype` 为 `Void` `）。 如果指定了 shape ，则值的形状必须与 shape 编码的形状匹配； 如果指定，`shape` 必须是已经绑定在当前范围内的 `Var` 或 `ShapeExpr`。
             2. 为了将形状值与“ShapeStructInfo”进行比较，“ndim”必须与形状值中的维数相匹配（除非“ndim”为 -1）。 如果已指定“values”，则形状值必须与“values”编码的值匹配。
             3.«为了将闭包（函数值）与`FuncStructInfo`进行比较，编译后的程序有必要跟踪闭包的运行时结构信息，因为不可能对闭包进行内省； 该主题将在文档后面进行更详细的讨论。 »
         2. 当将张量值与 `TensorStructInfo` 或形状值与 `ShapeStructInfo` 进行比较时，`TensorStructInfo` 中的 `shape` 的任何成员（如果 `shape` 是 `ShapeExpr`）或 `ShapeStructInfo` 中的 `values` 由以下组成 单个新的（迄今为止未绑定的）形状变量被视为绑定：形状变量绑定到匹配值的相应维度的大小。
         3. 如果提供了变量，则该值绑定到 `var` 表达式（如果省略变量，则执行结构检查并更新任何形状变量，但不引入新的绑定）。 `SeqExpr` 中引入的形状变量与 `SeqExpr` 的范围类似。
    
     `SeqExpr` 的 `body` 表达式允许引用任何在 `SeqExpr` 的绑定块中引入的 `Var`，以及外部作用域中的那些； `body` 表达式在绑定块之后被评估，它的值就是返回的值。 在表达式完成计算后，SeqExpr 中引入的任何 Relax 变量和形状变量都将从范围中删除。
    
13. `ShapeExpr` 节点构造形状文字，它们是形状维度的不可变集合。 其中的 PrimExpr 描述了如何计算每个维度； 他们可以自由使用范围内的任何形状变量。
14. `Function` 节点表示函数定义，接受列出的参数并在新范围内评估函数体表达式（这意味着函数内部定义的任何变量都不能在函数外部引用）。 函数定义可以嵌套在任何其他表达式中，并且它们计算为闭包值，确保函数是一流的。 闭包捕获在其主体中使用的外部作用域中的任何变量，包括 Relax 变量和形状变量。 请注意，函数定义本身是匿名的——函数必须在 `IRModule` 中注册（绑定到 `GlobalVar`）或出现在绑定的右侧以具有名称以便被递归调用。
    
     该函数可以在参数上具有结构注释，在返回值上具有结构注释。 当函数被调用时，参数上的注释会以类似于“MatchCast”的方式根据参数值进行检查，并且可以引入函数范围内的新形状变量。 此外，在调用返回之前，会根据注释检查返回值的结构信息。
    
     «映射绑定到 `GlobalVar` 的函数可以定义一个 `global_symbol` 属性，以指示它应该在外部进行外部链接（可以在 `IRModule` 之外访问）。 绑定到 GlobalVar 的函数定义中缺少 global_symbol 属性表明它是“私有的”，因此只能在 IRModule 中调用。»


## Purity and Dataflow Blocks

如果函数或运算符没有副作用，则称为“纯”函数或运算符，副作用指的是除了返回结果之外程序状态的任何变化。 副作用包括改变它们创建的值以外的值、中止程序或文件 I/O（包括写入控制台）。 纯度是编译器优化的一个有用属性，因为对纯函数的调用可以重新排序或重复，或者（如果结果未使用）消除，而不改变任何其他程序行为。 大多数深度学习运算符都是纯运算符，因为它们对张量执行算术运算并返回包含结果的新张量。

上面提到 `DataflowBlocks` 不允许包含以控制流为特征的构造（`If`节点或对当前函数的递归调用）或对不纯函数的调用。 这确保了 `DataflowBlocks` 代表纯操作的有向无环图，类似于传统深度学习框架的类图抽象。 这使得过去框架中的许多常见优化可以直接适用于 `DataflowBlocks`，而无需对控制流和副作用等更具表现力的功能进行额外推理。

Relax 允许在其他“纯”函数内部产生一个可见的副作用，即退出程序时出现错误。 在以下情况下可能会出现这种情况：

- 转换错误（来自 MatchCast 或调用 Relax 函数时的隐式结构信息检查）
- 由其他纯 Relax 运算符或 PackedFuncs 引发的错误。 由于操作符或 PackedFunc 的纯度必须手动注册，这意味着如果操作符或 PackedFunc 的唯一副作用是在某些情况下发出错误，则允许将其注册为纯操作符或 PackedFunc。

尽管程序异常退出是一个可见的副作用并且删除或重新排序它会改变可观察到的语义，但禁止在 DataflowBlocks 内部进行错误检查的限制太大了。 Relax 没有任何异常处理的概念，因此安全检查失败的唯一后果就是退出程序。 允许编译器重新排序、复制或消除 MatchCast 或其他可能失败的纯操作，前提是这样做不会更改程序返回的值或任何其他可见行为。

## Structural Information (`StructInfo`) System Survey

类似于大多数语言中的类型系统，Relax 跟踪与 Relax 中的值类别相关的结构信息（在实现中称为“StructInfo”）：
1. `TensorStructInfo` 对应张量值，给出标量数据类型、维数（秩）和计算张量形状的表达式（`ShapeExpr` 或 `Var`），所有这些都是可选的 .
2. `TupleStructInfo` 对应于元组值，为元组的每个成员给出`StructInfo`。
3. `PrimStructInfo` 对应于 `PrimValue`（不可变标量值），给出了它们的 TIR 数据类型。
4. `ShapeStructInfo` 对应于形状值，可选地给出形状中的维度数和计算形状维度的表达式（`ShapeExpr` 或 `Var`）。
5. `FunctionStructInfo` 对应于函数值（闭包）和 `PackedFunc`s（外部函数），给出参数的类型、返回类型，«以及函数是否是纯函数。»
6. `ObjectStructInfo` 是所有 Relax `StructInfo` 的父级，对应于上述所有值以及不属于上述类别的 `PackedFunc` 调用返回的任何值。

`StructInfo` 被分配给范围内的每个变量和每种类型的表达式，基于它通过规范后面定义的一组推理规则返回的值，利用子类型分配更通用的 `StructInfo` 当更具体的`StructInfo` 不能 确定。 «Relax 是强类型的，这意味着如果推断出的 `StructInfo` 不如预期的那么具体，则会发出错误，并且需要通过 `MatchCast` 进行显式检查。»

在 Relax 中，张量形状不是在类型系统中静态处理的，尽管编译器将形状信息用于静态优化将大有裨益。 相反，形状信息是使用 Relax 的结构信息系统来跟踪的，其中每个表达式都具有与其相关联的结构信息（如张量形状），这些信息比其类型更具表现力。 `StructInfo` 可以传达有关表达式的更丰富的属性，例如张量形状，并且可以促进更大程度的静态推理。 然而，当编译器无法得出有关结构信息的结论时，可以通过`MatchCast`动态检查此信息。 结构信息本质上是一个扩展类型系统，因此 `MatchCast` 也用于处理类型转换。

---

# Top-level Program Organization: `IRModule`

与 Relay 一样，Relax 程序的顶层组织是 `IRModule`。 `IRModule` 包含全局变量到函数的映射，包括 Relax 函数和 TIR 函数（可以从 Relax 调用）。 称为“main”的全局函数通常被认为是程序的入口点（意味着通过调用该函数开始执行），尽管任何具有“global_symbol”属性的函数都可以在编译期间指定为入口点。 在 AST（见下文）中，`IRModule` 中的 Relax 函数的名称是 `GlobalVar` 节点。

通常，编译器通道仅对特定函数进行操作或向 IRModule 添加新函数，但通道可以通过遍历 IRModule 中的所有函数来对整个 Relax 程序进行操作。

# Values in Relax

以下是 Relax 操作的值类别，这意味着它们可以分配给变量或作为计算表达式的结果。

- *Tensors* 是标量值的 n 维数组（可以是固定位宽的有符号或无符号整数、固定位宽的浮点数或布尔值）。 张量的 *shape* 是每个维度大小的元组； 维数是张量的*rank*。 例如，向量 (1, 2, 3) 是形状为“(3,)”的 rank-1 张量。 请注意，标量是秩为 0 的张量值，这意味着它们的形状为“()”。
- *Tuples* 表示其他 Relax 值（张量、闭包、形状、对象或其他元组，任意嵌套程度）的固定大小不可变分组。 请注意，一个空元组，即 `()`，在函数式编程中也称为“单元”，通常用作不打算返回值的操作的返回值（在某些 `PackedFunc` 或运算符中可能是这种情况 有副作用的调用）。
- *Closures*是评估 Relax 函数表达式的结果； 闭包可以像其他值一样传递，确保函数在 Relax 中是一流的。 Relax 中定义的函数可以从外部范围捕获变量。 [闭包](https://en.wikipedia.org/wiki/Closure_(computer_programming)) 由函数和“捕获”的任何变量的映射组成（这些是函数体中的*自由变量*，来自 既不是参数也不是在函数内定义但在函数中使用的外部范围）到它们的值。 闭包从外部范围捕获 Relax-level 局部变量和形状变量。 当主体包含递归调用时，闭包也会为自己存储一个名称。 «闭包还携带一些*运行时结构信息* (RTSI)，指示它们的参数和结果结构，以促进动态结构检查（因为否则不可能内省包含在闭包中的函数）； RTSI 的精确形式留给编译器实现来确定，只要“MatchCast”可以验证闭包的结构。 可以在调用节点中评估闭包，这会导致使用调用的参数和捕获的值调用函数。»
- *Tensor shapes*（形状值）是描述张量形状的不可变整数元组，通过评估 ShapeExpr 获得。
- *Packed functions*（`PackedFunc`s 或外部函数）表示在 TVM 中实现的任意不透明函数。 也就是说，打包函数是在 Relax 之外定义的例程，无法被编译器检查。 它们可以执行副作用并返回任意值。
- *Primitive values* (`PrimValue`s) 表示主要用于传递给外部过程的不可变标量值，例如调用 `PackedFunc`s。 根据经验，用于算术计算的标量值应该是 0 阶张量，而用作元数据的标量值应该是 PrimValue。
- 此外，还有其他*任意对象*不属于上述类别。 这些可以由 PackedFunc 和运算符返回； 此外，我们将 TIR `PrimFunc` 视为不透明对象。 尽管 `PackedFunc` 和运算符调用以外的 Relax 表达式不能使用这些对象，但 Relax 应该忠实地传递这些值。 将来我们可能会添加更多的值类型以区分不同的对象，但目前我们将这些都视为带有 `ObjectStructInfo` 的任意值。 请注意，目前，字符串和 TIR 数据类型也被视为不透明对象。 此类别中另一个值得注意的值是_null 对象_（在 C++ 中返回空指针或通过 Python FFI 传入“None”的结果），它由“null_value()”运算符返回。

## Representation of Values at Run Time

因为 Relax 支持调用可以在低级别上运行的任意 PackedFunc，所以有必要定义一个约定来说明在运行时如何表示值。 目前，该规范不需要任何特定的表示形式，并允许编译器实现选择自己的表示形式，前提是上面列出的每种值类型都可以在运行时识别（用于动态“StructInfo”检查）。 这意味着直接调用 PackedFunc 的 Relax 程序不能跨编译器实现移植：所使用的 PackedFunc 必须能够对值的运行时表示进行操作。

TVM 对象系统方面的可能规范：

- 张量在运行时表示为 `NDArray`（参见 `include/tvm/NDArray.h`）。
- 元组使用不可变的 TVM `Array`（与 `NDArray` 对比）表示（参见 `include/tvm/runtime/container/array.h`）。
- 在运行时，闭包表示为“ClosureObj”（参见“include/tvm/runtime/container/closure.h”）； 在 Relax VM 中，这些更具体地使用 `VMClosureObj`（参见 [`https://github.com/tlc-pack/relax/blob/relax/include/tvm/runtime/relax_vm/executable.h`]（https： //github.com/tlc-pack/relax/blob/relax/include/tvm/runtime/relax_vm/executable.h))。
- 形状值在运行时表示为“ShapeTuple”（参见“include/tvm/runtime/container/shape_tuple.h”）。
- 字符串使用 TVM 的“字符串”容器表示（参见“include/tvm/runtime/container/string.h”）。
- 我们要求 `PackedFunc` 使用和返回的上述值以外的对象继承自 TVM 的 `Object` 类（在 `include/tvm/runtime/Object.h` 中定义）。 请注意，`PackedFunc` 能够使用和返回所有 TVM POD（普通旧数据）值（参见 `include/tvm/runtimes/packed_func.h`），其中包括一些不继承自 `Object` 的表示。 将来，我们可能会为其他值定义语义，但目前 Relax *不支持*这些值，我们不保证调用使用或返回任何不继承自 Object 的 `PackedFunc` 的语义 .

# Variable Scoping

Relax 中有四个相关的作用域，它们决定了变量在哪里可见并可以使用：

1. 全局：可以从 IRModule 中的任何函数引用 GlobalVar，无论是 Relax 函数还是 TIR PrimFunc。 所有全局函数对彼此和它们自己都是可见的，允许相互递归。
2. 函数：函数的参数（普通的`Var` 节点）可以在该函数的任何地方被引用。 在递归绑定中（“Binding”节点，其中 RHS 是“Function”节点或“GlobalVar”被映射到“IRModule”级别的函数），被绑定的变量也限定在该函数范围内，允许定义 一个递归函数。
3. `SeqExpr`：在 `SeqExpr` 节点中的 `BindingBlock` 中定义的 `Var` 节点可以在同一 `BindingBlock` 中的任何后续绑定中，在该 `SeqExpr` 中任何后续 `BindingBlock` 中的任何绑定中引用 节点，或在“SeqExpr”的正文表达式中。 一旦 `SeqExpr` 返回，在 `BindingBlock` 的离开作用域中定义的变量。
4. `DataflowBlock`：在 `DataflowBlock` 中引入的 `DataflowVar` 可以在该 `DataflowBlock` 中的任何后续绑定中引用，但是一旦 `DataflowBlock` 完成执行*就离开范围*。 旨在离开 `DataflowBlock` 的 `DataflowBlock` 中的定义应绑定到普通的 `Var`。

请注意，Relax 变量必须_exactly_ 绑定一次。 如果全局变量映射到 `IRModule` 中的函数，则全局变量被绑定；如果局部变量作为函数参数出现或出现在绑定（`VarBinding`）的左侧（LHS），则被绑定 或 `MatchCast`）。

«如果有另一个绑定到与已绑定变量同名的局部变量，则该绑定被认为是_shadow_前一个绑定，即它是绑定到一个新的、不同的变量，而这个变量恰好具有相同的名称 名称作为现有变量。 新的阴影变量将仅存在于当前范围内； 如果旧变量是在外部作用域中定义的，那么以后对该名称的使用将引用旧变量。 [有关变量阴影的更多信息，请参阅维基百科页面。](https://en.wikipedia.org/wiki/Variable_shadowing)»

Below is an example of shadowing, in pseudocode:

```python
@R.function
def func(x: Tensor) -> Tensor:
    if True:
        # the true branch will be a nested SeqExpr and hence a new scope
        # this x will shadow the function parameter x
        x = R.const(1)
        R.print(x) # prints 1
        # the inner x goes out of scope
    else:
        R.print("not executed")
    R.print(x) # this x is the function parameter
    return x
```

# Normal Form

为了简化 Relax passes 的编写，我们定义了 Relax 程序的范式，基于 [administrative normal form](https://en.wikipedia.org/wiki/A-normal_form)（A-normal form，或 ANF ). 请参阅 Matt Might 的[这篇文章](https://matt.might.net/articles/a-normalization/)，其中讨论了 ANF 在传统编译中的一些优势； 特别是 ANF 导致程序没有嵌套，这对于编写程序转换非常方便。 因为运算符的 `StructInfo` 检查规则依赖于宏 (`FInferShapeInfo`)，_这意味着程序的结构可以影响 `StructInfo` 推断_。 将程序转换为正常形式（并且没有嵌套）不仅简化了这些宏的编写，而且还确保了这些“StructInfo”检查规则是可预测的，因此在应用“StructInfo”检查之前_需要将程序转换为正常形式_ .

Relax 的正常形式与 ANF 非常相似； 差异将被注意到。 以下是程序成为范式所需的条件：
1. 在 `SeqExpr` 中，任何绑定的右侧（AST 中的 `value` 字段）必须是“叶表达式”或非叶表达式，其中所有子表达式都是叶表达式。 叶表达式如下：变量（`Var`、`DataflowVar` 或 `GlobalVar`）、`Constant`、`ShapeExpr`、`PrimValue`、`StringImm`、`DataTypeImm` 或（_unlike_ ANF）`Tuple` . `Tuple` 节点被认为是“叶”表达式，即使它们包含纯粹为了方便编写传递的嵌套； 许多运算符依赖于使用元组对参数进行分组，因此这是一种允许和预期的嵌套形式。 否则，用作子表达式的非叶表达式必须绑定到变量； 这包括嵌套在“元组”中的任何非叶表达式。
2. `SeqExpr`s 可能只出现在以下位置：
     1. 在 `Function` 节点的 `body` 字段中。
     2. 在“If”节点的“true_branch”和“false_branch”字段中。
3. 事实上，`Function` 节点的`body` 字段和`If` 节点的`true_branch` 和`false_branch` 字段_必须_ 是`SeqExpr`。 如果这些字段不是“SeqExpr”，则它们必须“包装”在“SeqExpr”中。
4. 在“SeqExpr”中，“BindingBlock”必须合并。 例如，如果在另一个“BindingBlock”之后有一个“BindingBlock”，则应将这两个块组合成一个“BindingBlock”，所有绑定的顺序相同。 连续的“DataflowBlock”也应该合并。 应删除空的 BindingBlock。 但是，“DataflowBlock”不能与普通的“BindingBlock”合并。 如果所有的 `BindingBlock` 都是空的，那么 `SeqExpr` 的 `blocks` 字段应该设置为一个空列表。

在执行“StructInfo”检查或进行任何进一步优化之前，应该对被解析的程序进行“规范化”。 请注意，“扁平化”`SeqExpr` 和合并 `BindingBlock` 的过程确实增加了那些 `SeqExpr` 和 `BindingBlock` 中变量的可见性，但这是安全的，因为它不会导致任何变量 在其原始范围之外被引用。 只要最终程序符合上面列出的标准，规范就不需要任何特定的程序规范化方法。 这是一个通用的方法：
1. 对于 IRModule 中的每个函数，确保函数体是一个 SeqExpr。 如果主体不是“SeqExpr”，则将函数主体包装在“SeqExpr”中，创建一个新的“BindingBlock”来为需要绑定到变量的任何非叶表达式保存“VarBinding”。
2. 如果函数体已经是一个`SeqExpr`，合并所有的`BindingBlock`，然后检查`SeqExpr`的`body`字段是否是叶子表达式。 如果不是，则将其绑定到最终“BindingBlock”中的新变量，并用新变量替换“SeqExpr”主体。
3. 如果函数主体不是 `SeqExpr`，则向下递归主体的 AST，将任何嵌套的非叶表达式绑定到当前作用域中的 var（从左到右以广度优先顺序执行此过程将尊重 语义中的评估顺序）。 如果正文本身是一个非叶表达式，最后将它绑定到一个 var 并让最终的 `SeqExpr` 返回新的 var。
4. 如果遇到 `If` 节点，请确保 `true_branch` 和 `false_branch` 字段是 `SeqExpr`（必要时合并 `BindingBlock`）或以与 函数体。
5. 如果遇到作为绑定中的“值”节点的“SeqExpr”节点，通过将其绑定添加到当前范围并用其主体替换“SeqExpr”来“展平”“SeqExpr”。 如果 `SeqExpr` 主体是非叶表达式，则在替换绑定之前以与步骤 3 中相同的方式对其进行递归归一化。 请注意，如果当前作用域（绑定的位置）是一个“DataflowBlock”并且嵌套的“SeqExpr”包含一个普通的“BindingBlock”，则表示程序格式错误。


# Well-Formedness Criteria

在“StructInfo”检查之前，Relax 程序必须符合某些有效的语法标准，其中包括符合上述范式的预期。

以下标准适用于所有程序（包括标准化之前）：
1. `DataflowVar` 只能在 `DataflowBlock` 中绑定。 此外，不能在定义它的“DataflowBlock”之外使用“DataflowVar”。
2. 程序中使用的任何类型的 `Var` 必须是函数参数或在绑定的 LHS 中出现一次。 在定义了 `Var` 的绑定中，仅当绑定正在定义一个函数（即，允许局部函数是递归的）时，才允许在绑定的 RHS 中出现相同的 `Var`。
3. 任何类型的`Var` 在绑定之前都不能出现。 也就是说，如果 `Var` 绑定在 `SeqExpr` 的 `BindingBlock` 中，则该 `Var` 可能不会出现在它出现在 LHS 上的绑定之前的绑定中。
4. «函数的返回结构注释不允许使用任何不在函数定义范围内的形状变量。 也就是说，唯一可以出现在返回结构注释上的形状变量是在外部范围中定义的那些或在参数结构注释中引入的那些。»
5. 在每个函数中，`PrimExpr` 变量（形状变量）同样可能不会出现在形状变量绑定之前（无论是在函数签名还是`MatchCast` 绑定中）的`ShapeExpr` 或形状注释中。 形状变量仅在其自身出现在维度中时才被绑定（例如，由“x”组成的维度将绑定“x”；但是，“2*x”不是绑定，如果“x”被视为错误 ` 尚未绑定）在 `MatchCast` 节点或函数参数形状注释中。
6. 以下构造不允许出现在 `DataflowBlock` 中，它们必须是无副作用和无控制流的：
     1.递归调用当前函数
     2.调用一个与当前函数互递归的全局函数
     3. `If` 节点
    
     «也不允许调用不纯的 Relax 函数、`ExternFuncs` 或 `Op`，但这必须在检查 `StructInfo` 时检测到。»
    
7. «对于包含对自身的递归调用或相互递归全局函数的函数（即函数`a`调用函数`b`和函数`b`调用函数`a`的函数），返回结构注释是*必需的* .»
8. `Op` 节点可能仅作为 `Call` 节点的 `op` 参数出现。
9. 如果变量具有“StructInfo”注释，则任何“TensorStructInfo”和“ShapeStructInfo”的“ndim”必须分别与其“shape”和“values”字段中的维数相匹配。
10. `DataflowBlock` 内的函数定义不得在其主体中使用来自外部作用域的 `DataflowVar`。 我们没有为 `DataflowVar` 定义闭包捕获。
11. «IRModule 中至少有一个全局函数必须被外部链接（具有 global_symbol 属性）才能作为程序入口点。»
12. «如果一个全局函数定义了`global_symbol` 属性，则`global_symbol` 名称必须与`GlobalVar` 的名称提示相同。»
13. 如果给出任何结构注释中的 TensorStructInfo 的 shape 字段，则唯一允许的表达式是 Var（变量必须在注释位置的范围内）或 ShapeExpr（其中任何 使用的形状变量必须已经在范围内，除非`TensorStructInfo` 是`MatchCast` 的`struct_info` 字段，在这种情况下，允许新的形状变量单独出现在维度中）。 此外，如果 `shape` 字段是 `ShapeExpr`，则维数必须与 `ndim` 字段相匹配。
14. 如果给出任何结构注释中 ShapeStructInfo 的 values 字段，则其中使用的任何形状变量必须已经在范围内，除非 ShapeStructInfo 是 MatchCast 的 struct_info 字段，在 在这种情况下，允许新的形状变量作为“值”的成员单独出现。 此外，“ndim”字段必须与“values”的长度相匹配。
15. `params`和`derive_func`字段不能同时定义在`FuncStructInfo`注解中； 也就是说，如果定义了一个，则不能定义另一个。 此外，必须为注解中的每个“FuncStructInfo”至少定义“params”和“derive_func”之一。
16. `PrimValue` 节点仅用于由 TIR `IntImm` 和 `FloatImm` 组成的 `value`（其中 `lanes` 设置为 1）。
17. `PrimStructInfo` 注释在其 `dtype` 字段中应仅使用 `Int`、`UInt` 或 `Float` 数据类型。
18. 根据 [关于 `DataType` 的注释](#notes-on-datatype-and-related-terminology)，任何 `DataType` 注释的 `lanes` 值必须为 1，用于 `Int`、`UInt`、 或 `Float` 数据类型和 `Handle` (`Void`) 数据类型的 `lanes` 值为 0。 此外，对于 Void，“bits”必须为 64。 `Int` 和 `UInt` 支持的位宽是 1、8、16、32 和 64； `Float` 支持的位宽为 16、32 和 64。

此外，[上一节](#normal-form) 中列出的范式标准必须适用于任何已规范化的程序。

# Structural Information (`StructInfo`) in Relax

Relax 中的结构信息旨在加强表达式之间正确传递值的基本保证，同时还以“尽力而为”的方式分析更复杂的属性，如张量形状。 也就是说，任何不能静态证明的东西都可以在运行时检查。 每个 Relax 表达式都有与其关联的结构信息。 Relax 中结构系统的最大努力性质意味着分析可能会在编译时检测到_一些_错误并报告它们，但它可能会在_cannot_得出结论时发出警告，这可能表明应该插入通过`MatchCast` 进行的动态检查。 请注意，静态分析的精度可能会通过一些编译时优化来提高，例如常量传播、函数内联和其他部分评估类转换。

张量形状是在 Relax 中包含结构信息的主要动机，因为形状信息对于内存规划尤为重要。 在 Relay 中，形状是张量类型的一部分，并且在编译时对张量形状进行了大量分析。 虽然这允许 Relay 的类型系统对张量形状做出强有力的保证，但它会导致类型检查更加复杂，并且难以实现新的运算符或处理具有符号形状的张量等情况。 相比之下，Relax 的“StructInfo”系统使用表达式对张量形状进行编码，这允许使用形状变量和算术表达式对各种形状约束进行编码。 但是请注意，结构系统可能会被扩展以编码和分析更多信息，例如张量稀疏性或密度。

[ref](https://github.com/slyubomirsky/relax/blob/spec-draft/relax_spec.md)



