# DL-Complier-Navigation

### IREE

IREE（Intermediate Representation Execution Environment，发音为“eerie”）是一个**基于 MLIR** 的**端到端编译器和运行时**，可将机器学习 (ML) 模型降低为统一的 IR，可向上扩展以满足数据中心和向下的需求以满足移动和边缘部署的约束和特殊考虑。
IREE 仍处于早期阶段。 我们已经确定了总体基础设施，并正在积极改进各种软件组件以及项目物流。 它距离日常使用还有很长的路要走。

### XLA
XLA的全称是Accelerated Linear Algebra，即加速线性代数。作为一种**深度学习编译器**，长期以来被作为Tensorflow框架的一个试验特性被开发，历时至今已经超过两三年了，随着Tensorflow 2.X的发布，XLA也终于从试验特性变成了默认打开的特性。此外， Pytorch社区也在大力推动XLA在Pytorch下的开发，现在已经有推出PyTorch/XLA TPU版本，暂只支持谷歌平台TPU上使用。

XLA使用JIT编译技术来分析用户在运行时创建的 TensorFlow 图，将TensorFlow Op转换成为HLO（High LevelOptimizer）中间表示并在HLO层上完成包括Op Fusion在内的多种图优化，最后基于LLVM完成CPU／GPU等机器代码的自动生成。

XLA的目的是帮助用户做通用，透明的一键式性能优化，提升Training／Inference时的Latency／Throughput等，整个过程完全自动。嵌入到宿主TensorFlow里执行的XLA会在原始计算图上自动圈出能够完整支持的子图，不能支持的个别算子可以继续通过TensorFlow的执行引擎来执行，因此对不同的计算图都有比较好的兼容性和可回退特性。**从应用场景上，XLA不区分training或者inference，与TensorFlow的良好集成使得它可以实现对用户的完全透明**。

参考：[华为开发者论坛](https://developer.huawei.com/consumer/cn/forum/topic/0201750315901780148?fid=0101592429757310384)

https://zhuanlan.zhihu.com/p/163717035

### HLO

HLO (High Level Optimizer) 是XLA的IR。

XLA 的基石是 HLO（高级优化器）IR，它提供了一个精心挑选的操作列表，大部分是相互正交的。它为用这组操作表达的计算提供了一个高效的优化器，并为 CPU、GPU 和 TPU 等硬件平台生成代码。它的目标是提供一个统一的接口来独立于目标设备编译和执行这些优化的 HLO 程序

参考： https://zhuanlan.zhihu.com/p/396309457

### MLIR-HLO

MLIR-HLO：基于 MLIR 的独立“HLO”编译器。

受XLA HLO启发，使用 MLIR 组件的线性代数运算集实现了一个独立的编译器 。旨在提供独立于 TensorFlow 和 XLA 的端到端流程

定义了三种Dialect来支持使用 MLIR 的类似 HLO 的编译管道：- `chlo`：“客户端”HLO 方言，旨在更接近前端（包括隐式广播语义）。

- `mhlo`：“元”-HLO方言；类似于`xla_hlo`，但具有动态形状支持的扩展。

- `lmhlo`: "late"-"meta"-HLO，就是分配缓冲区后的IR。在 XLA 中，缓冲区分配是跟踪这些信息的辅助数据结构，而这种单独的方言在 IR 中将其具体化。

参考： [GitHub - tensorflow/mlir-hloGitHub - tensorflow/mlir-hlo](https://github.com/tensorflow/mlir-hlo)

### Halide

Halide是一种编程语言，主要在**图片处理和矩阵计算**时具有方便快捷高性能的特点。它不是一种独立语言，而是基于C++的DSL(Domain Specified Language)，主要应用在算法的底层加速，并且此优化与算法本身设计无关。Halide思想在传统的图像处理(OpenvCV)和深度学习(TVM)优化加速方面具有较强的借鉴意义。

**Halide的特点是其图像算法的计算的实现（Function和Expression）和这些计算在计算硬件单元上的调度（Schedule）是分离的**，其调度以Function为单位。最终将整个图像算法转换为高效率的多层for循环，for循环的分部数据范围划分和数据加载都是由Halide来完成的，而且可以实现数据的加载和算法计算的Overlay，掩盖数据加载导致的延迟。Halide的Schedule可以由程序员来指定一些策略，指定硬件的buffer大小，缓冲线的相关设置，这样可以根据不同的计算硬件的特性来实现高效率的计算单元的调度，而图像算法的计算实现却不需要修改。

主要使用方法有Reorder(交换)、Split(拆分)、Fuse(融合)、Tile(平铺)、Vector(向量化)、展开(Unrolling)、并行(Parallelizing)，结合这些方法可以实现缓存一致性强、并行度高、额外开销少的图像处理程序。

参考: https://zhuanlan.zhihu.com/p/346468141

https://www.zhihu.com/question/294625837/answer/496218375

### autotvm

### AutoTensorIR
继autotvm, ansor之后第三代方法，但是现在还不可用
### xgb论文

### RAMMER

计算图的调度和算子内部调度是分开互不影响的，这两层分开进行调度会导致运行时的调度开销很大，算子间并行性没有被有效利用，以及忽视了算子内和算子间两种并行性的相互影响。

因此希望提出一种方法，避免这种问题，于是rammer因运而生。提出了一套新的抽象，将原来的算子抽象为rOperator，并将其进一步分解为更小的调度单元rTask，将底层的硬件抽象为由多个虚拟执行单元（virtualized execution units, vEU）组成的 vDevice。在这套新的抽象下，用户可以通过更细的 rTask 粒度将数据流图调度到多个 vEU 之上，兼顾了计算任务中的两种并行性与底层计算资源的协调。整个调度方案在编译期生成并“静态”映射到硬件计算单元上，因此可以天然地消除掉许多原本存在的调度开销。



### ANSOR论文

这篇论文做出了以下贡献：- 一种为计算图生成张量程序的**大型空间分层搜索机制**。

- 一种基于**可学习的代价模型的进化策略**，用来微调张量化程序的性能。

- 一种基于**梯度下降的调度算法**，在优化 DNN 的端到端性能时对重要子图进行优先排序。

- Ansor 系统的实现和综合评估表明，上述技术在各种 DNN 和硬件平台上都优于最先进的系统。

**ANSOR 的 Pipeline 可以分为如下几个步骤：**- **Schedule Task：将完整的计算图划分成多个子图，对热点的子图进行重点优化**

- **Sketch：提取算子中的高层次特征，对算子进行较粗粒度的优化，确定代码的基本结构**

- **Annotation：随机初始化 Tilling Size 和一些 for 循环的策略，获得计算图的完整表示**

- **Evolutionary：训练 Cost Model，根据 Cost Model 对代码的性能进行评估，选取评估中取得高分数的一组实现，再通过运行时模块，获取 ground truth 和实际性能，选取实际性能最优的实现作为 ANSOR 的输出。**

### roller论文

### pet论文

### HALO

Heterogeneity-Aware Lowering and Optimization(HALO)是异构计算加速度基于编译器的技术平台。

它通过称为开放深度学习 API ( **ODLA** )的抽象、可扩展接口来利用针对深度学习领域的异构计算能力。HALO 提供统一的 Ahead-Of-Time 编译解决方案，自动为云、边缘和物联网场景量身定制。

HALO 支持多种编译模式。在提前（AOT）编译模式下，HALO 将 AI 模型编译成用 ODLA API 编写的 C/C++ 代码。编译后的模型可以在具有相应 ODLA 运行时库的任何受支持平台上运行。此外，HALO 能够同时编译主机和异构设备代码。

参考： [异构加速平台Heterogeneity-Aware Lowering and Optimization (HALO)简介-阿里云开发者社区](https://developer.aliyun.com/article/786953)

[GitHub - alibaba/heterogeneity-aware-lowering-and-optimization: heterogeneity-aware-lowering-and-optimization](https://github.com/alibaba/heterogeneity-aware-lowering-and-optimization)

### TensorRT

NVIDIA TensorRT 是一个用于深度学习推理的 SDK 。 TensorRT 提供 api 和解析器来从所有主要的深度学习框架中导入经过训练的模型。然后生成可部署在数据中心、汽车和嵌入式环境中的优化运行时引擎。

TensorRT优化训练好的神经网络模型以产生可部署的运行时推理引擎

TensorRT主要做了下面几件事，来提升模型的运行速度。- **TensorRT支持FP16和INT8的计算**。我们知道深度学习在训练的时候一般是应用32位或者16位数据，TensorRT在推理的时候可以降低模型参数的位宽来进行低精度推理，以达到加速推断的目的。

- **TensorRT对于网络结构进行重构**，把一些能够合并的运算合并在了一起，针对GPU的特性做了优化。**在GPU上跑的函数叫Kernel，TensorRT是存在Kernel的调用的。在绝大部分框架中，比如一个卷积层、一个偏置层和一个reload层，这三层是需要调用三次cuDNN对应的****API****，但实际上这三层的实现完全是可以合并到一起的；再比如说，目前的网络一方面越来越深，另一方面越来越宽，可能并行做若干个相同大小的卷积，这些卷积计算其实也是可以合并到一起来做的。**

- 然后Concat层是可以去掉的，因为TensorRT完全可以实现直接接到需要的地方。

- **Kernel Auto-Tuning**：网络模型在推理计算时，是调用GPU的CUDA核进行计算的。TensorRT可以针对不同的算法，不同的网络模型，不同的GPU平台，进行 CUDA核的调整，以保证当前模型在特定平台上以最优性能计算。

- **Dynamic Tensor Memory** 在每个tensor的使用期间，TensorRT会为其指定显存，避免显存重复申请，减少内存占用和提高重复使用效率。

- 不同的硬件如P4卡还是V100卡甚至是嵌入式设备的卡，TensorRT都会做优化，得到优化后的engine。

参考： [深度学习算法优化系列十七 | TensorRT介绍，安装及如何使用？ - 腾讯云开发者社区-腾讯云](https://cloud.tencent.com/developer/article/1594985)

### nnfusion

### TACO

### Tensor Comprehension
采用polyhedral算法路线的编译器
### FlexTensor

### ngraph

### Learning to optimize tensor programs

### Tensor comprehension: framework-agnostic high-performance machine learning abstractions

### Tiramisu

采用polyhedral算法路线的编译器

#### 调度技术分类

|        | 自动生成schdule| 非自动生成schdule| 
|  ----  | ----  | ----|
| 依赖分析  | PolyMage, Tensor Comprehensions| AlphaZ,CHiLL,URUK,Tiramisu|
| 区间分析  | AutoScheduler(TVM) |Halide,AutoTVM|

编译技术根据循环嵌套分析算法的不同可以分为基于依赖分析和基于区间分析的两大类
* 依赖分析也即传统编译器的应用仿射变换优化循环嵌套代码的多面体分析技术，由于深度学习模型的算子在推理阶段的循环控制流是静态可判定的，因此非常适合应用该技术优化计算过程；基于依赖分析的Polyhedral模型的调度描述更加细化、表达力更强，理论上可以将优化做到极致，但缺点是算法原理相对复杂且优化分析的复杂度更高；
* 区间分析针对图像处理领域的常用计算（针对图像矩阵的卷积、池化操作）简化了循环计算过程为循环轴对齐，即简化依赖分析的多面体抽象为长方体抽象，以牺牲一定的资源利用为代价简化常用算子的编译过程。而基于区间分析的调度模型的优势在于，其在图像处理领域的优化效果和前者相差无几，但优化分析的复杂度低很多，缺点则是对于图像处理领域外的代码调度表达力不足，难以优化运行代码到极致性能。

另一种分类方式是调度生成的自动化程度
* 非自动化生成调度的编译器通常会向用户提供一种领域特定语言（DSL），如TVM的Tensor Expression、Tiramisu的Tiramisu Language，用户使用DSL语言描述由算子的计算到具体调度的转化过程；非自动化的方法需要**用户**对目标设备的体系结构有足够理解并提供调度生成模板（AutoTVM）或具体调度过程（Tiramisu），用户在自定义的模板/过程上可以调整调度参数以优化调度过程
* 自动化生成调度的编译器则会内置一套或多套编译准则，这套准则根据用户定义的计算过程描述以及设备性能描述自动生成最优的调度过程。自动化的方法则是对编译准则的**设计者**在计算机体系结构、代码编译原理方面提出了很高的要求，以确保设计的编译准则可以根据给定算子以及运行设备信息生成高效的调度过程。

Tiramisu可以归类于“基于依赖分析的调度非自动化生成”中。

#### DSL
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
### AKG
采用polyhedral算法路线的编译器
AKG已经用在了MindSpore的图算融合里面
### Diesel

### Stripe

### Polyhedral model

### OpenVino

OpenVINO™ 是一个用于优化和部署人工智能（AI）推理的开源工具平台。它可以：

- 提高计算机视觉、自动语音识别、自然语言处理和其他常见任务的深度学习性能。

- 使用经过 TensorFlow、PyTorch 等流行框架训练的模型。

- 减少资源需求，并在从边缘到云的一系列英特尔®平台上高效部署

### ONNX

### TorchScript

tvm、mlir论文

### Bring You Own Codegen

### 并行计算

### 分布式推理

### AI框架中的IR分析

参考这个https://zhuanlan.zhihu.com/p/263420069。从一个比较顶层的视角来看IR，包括编译器的IR以及深度学习编译的IR。有挺多东西看不懂的，里面说的不同表示能够取得不同的效果，我也不知道是什么原因。然后里面还提到了函数式编程的概念。


**性能上的优化(XLA/TVM/TC)**

性能上的优化思路其实比较统一，就是打开图和算子的边界，进行重新组合优化。

- **XLA**基本上的思路是把图层下发的子图中的算子全部打开成小算子，然后基于这张小算子组成的子图进行编译优化，包括buffer fusion、水平融合等，这里的关键是大算子怎样打开、小算子如何重新融合、新的大的算子(kernel)怎样生成，整体设计主要通过HLO/LLO/LLVM层层lowering实现，所有规则都是手工提前指定。

- **TVM**分为Relay和TVM两层，Relay主要关注图层，TVM主要关注算子层，总体思路与XLA是类似的，也是拿到前端给一张子图进行优化，Relay关注算子间的融合、TVM关注新的算子和kernel的生成，区别在于TVM是一个开放的架构，Relay目标是可以接入各种前端，TVM也是一个可以独立使用的算子开发和编译的工具（基于Halide IR，最新演进到自己定义的TIR），TVM在算子实现方面采用了compute和schedule分离的方案，开发人员通过compute来设计计算的逻辑，通过schedule来指定调度优化的逻辑。

- **TC**(Tensor Comprehensions)：开发者发现算子的计算逻辑的开发是比较容易的，但是schedule的开发非常困难，既要了解算法的逻辑又要熟悉硬件的体系架构，更重要的是，前面提到图算边界打开后，小算子融合后，会生成新的算子和kernel，这些新的算子compute是容易确定的（小算子compute的组合），但是schedule却很难生成，所以传统的方法就是事先定义一大堆schedule模板，万一组合的新算子不在模板之内，性能就可能比较差，甚至出错；那TC则希望通过Polyhedra model实现auto schedule，降低开发门槛，当然这个项目基本已经停更了，但是类似的工作在MLIR、MindSpore上还在不停发展。

### 相关资料
[Deep Learning Systems Course (CMU; video)](https://www.youtube.com/channel/UC3-KIvmiIaZimgXMNt7F99g)

[机器学习系统：设计和实现 (book)](https://openmlsys.github.io/)

[动手学深度学习 (book; video)](https://zh-v2.d2l.ai/)

[实用机器学习 (video)](https://space.bilibili.com/1567748478/channel/collectiondetail?sid=28144)

[TVM官网 (blog)](https://chinese.tvm.wiki/)

[BBuf (blog)](http://giantpandacv.com/project/%E9%83%A8%E7%BD%B2%E4%BC%98%E5%8C%96/)

[Haskell (book)](http://learnyouahaskell.com/syntax-in-functions)

[论文汇总 (github)](https://github.com/merrymercy/awesome-tensor-compilers#open-source-projects)

[LLVM (book)](https://getting-started-with-llvm-core-libraries-zh-cn.readthedocs.io/zh_CN/latest/)