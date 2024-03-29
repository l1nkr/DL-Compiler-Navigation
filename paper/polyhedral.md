# Polyhedral Model

## Tiramisu

采用polyhedral算法路线的编译器

### 调度技术分类

|      | 自动生成schdule                     | 非自动生成schdule               |
| ---- | ------------------------------- | -------------------------- |
| 依赖分析 | PolyMage, Tensor Comprehensions | AlphaZ,CHiLL,URUK,Tiramisu |
| 区间分析 | AutoScheduler(TVM)              | Halide,AutoTVM             |

编译技术根据循环嵌套分析算法的不同可以分为基于依赖分析和基于区间分析的两大类

* 依赖分析也即传统编译器的应用仿射变换优化循环嵌套代码的多面体分析技术，由于深度学习模型的算子在推理阶段的循环控制流是静态可判定的，因此非常适合应用该技术优化计算过程；基于依赖分析的Polyhedral模型的调度描述更加细化、表达力更强，理论上可以将优化做到极致，但缺点是算法原理相对复杂且优化分析的复杂度更高；
* 区间分析针对图像处理领域的常用计算（针对图像矩阵的卷积、池化操作）简化了循环计算过程为循环轴对齐，即简化依赖分析的多面体抽象为长方体抽象，以牺牲一定的资源利用为代价简化常用算子的编译过程。而基于区间分析的调度模型的优势在于，其在图像处理领域的优化效果和前者相差无几，但优化分析的复杂度低很多，缺点则是对于图像处理领域外的代码调度表达力不足，难以优化运行代码到极致性能。

另一种分类方式是调度生成的自动化程度

* 非自动化生成调度的编译器通常会向用户提供一种领域特定语言（DSL），如TVM的Tensor Expression、Tiramisu的Tiramisu Language，用户使用DSL语言描述由算子的计算到具体调度的转化过程；非自动化的方法需要**用户**对目标设备的体系结构有足够理解并提供调度生成模板（AutoTVM）或具体调度过程（Tiramisu），用户在自定义的模板/过程上可以调整调度参数以优化调度过程
* 自动化生成调度的编译器则会内置一套或多套编译准则，这套准则根据用户定义的计算过程描述以及设备性能描述自动生成最优的调度过程。自动化的方法则是对编译准则的**设计者**在计算机体系结构、代码编译原理方面提出了很高的要求，以确保设计的编译准则可以根据给定算子以及运行设备信息生成高效的调度过程。

Tiramisu可以归类于“基于依赖分析的调度非自动化生成”中。

### DSL

Tiramisu定义了一套领域专用语言（DSL），该语言以C++为基础，提供了一套API供用户调用。用户**可基于Tiramisu DSL APIs定义循环优化、内存布局等转化规则以指导算子调度的生成过程**，Tiramisu Compiler进而根据用户定义的规则将原始深度学习模型的所有算子转化为低级别IR，并最终生成运行于设备后端的优化代码。

Tiramisu共定义了4种类型的调度命令：

* 循环嵌套变换命令：这一类型的调度命令包括常见的仿射变换，如循环展开、分割、移位等。
* 循环-硬件关联命令：该类型的调度命令包括循环并行、向量化以及绑定循环到指定计算资源的操作等。
* 数据操作命令：数据操作命令可以分为4种类型：
  1. 分配Tensor空间命令
  2. 设置Tensor属性命令，如设置数据存储位置(host/device/shared)
  3. 数据拷贝命令
  4. 设置数据存取属性命令。数据操作命令也有高级和低级之分，通常用户使用高级命令即可完成一般的调度规划，更细致的规划则需要低级命令参与
* 数据同步操作命令：Tiramisu相比其他Compiler比较有特色的命令，类似于MapReduce的思路。设计者考虑到一次计算的数据量非常大的情况下可能需要多节点共同计算，因此设计了send/recv的调度操作，籍此可以在多节点之间共享数据或数据片段。
