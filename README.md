# DL-Complier-Navigation

## Tuning and Schedule

### Learning to Optimize Tensor Program(AtuoTVM. NeurIPS 2018)

使用机器学习方法优化算子模板

### FlexTensor(ASPLOS 2020)

**用户无需手动编写任何计划或模板。**

FlexTensor 由前端和后端两部分组成。 
- 前端用 Python 编写的张量计算作为输入。采用**静态分析**分析计算模式并生成特定于硬件的**调度空间**。
- 后端利用**启发式和机器学习**相结合的方法来找到优化的调度配置。启发式方法基于**模拟退火**，机器学习算法基于 **Q-Learning**。

### ANSOR(OSDI 2020)

这篇论文做出了以下贡献：- 一种为计算图生成张量程序的**大型空间分层搜索机制**。

- 一种基于**可学习的代价模型的进化策略**，用来微调张量化程序的性能。

- 一种基于**梯度下降的调度算法**，在优化 DNN 的端到端性能时对重要子图进行优先排序。

- Ansor 系统的实现和综合评估表明，上述技术在各种 DNN 和硬件平台上都优于最先进的系统。

**ANSOR 的 Pipeline 可以分为如下几个步骤：**- **Schedule Task：将完整的计算图划分成多个子图，对热点的子图进行重点优化**

- **Sketch：提取算子中的高层次特征，对算子进行较粗粒度的优化，确定代码的基本结构**

- **Annotation：随机初始化 Tilling Size 和一些 for 循环的策略，获得计算图的完整表示**

- **Evolutionary：训练 Cost Model，根据 Cost Model 对代码的性能进行评估，选取评估中取得高分数的一组实现，再通过运行时模块，获取 ground truth 和实际性能，选取实际性能最优的实现作为 ANSOR 的输出。**

### Woodpecker-DL

研究了基于**遗传算法和强化学习**的两种新的自动搜索方法，以寻找针对特定硬件的最佳运算符代码配置

### AdaTune(NeurIPS 2020)

- 提出了一种**自适应评估方法**，可以在统计上提前终止昂贵的硬件测量，而不会损失太多的准确性。
- 设计了一个**具有不确定性量化的代理模型**，使优化能够更好地适应硬件和模型异质性。
- 引入了一个**上下文优化器**，它提供了对 exploration exploitation的自适应控制，以提高转换空间搜索的有效性。

### Chameleon(ICLR 2020)

1. 设计一个**自适应探索模块**，利用**强化学习**来适应新网络的看不见的设计空间，以减少搜索时间，同时获得更好的性能。
2. 提出一种**自适应采样算法**，该算法利用**聚类**来自适应地减少昂贵的硬件测量次数，并设计一种受领域知识启发的**样本合成**，以找到可能产生更好性能的配置。

- 之前的算法，如此长的优化时间是由于模拟退火的低效（虽然它随机保证在大量迭代后能得到合理解决方案），他未能捕捉到设计空间中可以在搜索过程中利用的模式。
- 大部分优化时间都花在了真实硬件上的测量上，这些测量被用作上述搜索的反馈。
- 目前的方法会遭受大量无效配置的困扰，这不仅浪费了编译器开始时有限的硬件测量预算，而且还会产生严重的开销来重置目标硬件以进行后续硬件测量。
- 重要的是，为硬件测量选择潜在配置的采样机制更智能，以确保每次测量都最大限度地提高获得良好解决方案的机会，并避免无效配置。然而，目前的方法依赖于贪婪抽样，它根据成本模型的估计进行被动抽样。这不仅有过度拟合的趋势，而且还忽略了解决方案分布不均匀以及存在许多无效配置。

### DynaTune(ICLR 2021)

考虑了用于张量规划优化问题的多臂老虎机 (MAB) 模型。使用 UCB 来处理基于时隙的优化的决策，并且设计了一个贝叶斯信念模型，该模型允许通过不确定性量化来预测每个算子的潜在性能增益，从而指导优化过程。
能够在获得相同优化结果的情况下，减少优化时间。

- 现有的 DL 编译侧重于加快单个张量算子而不是整个模型的收敛速度，导致收敛速度慢，优化时间长。
- 静态调度对张量程序的了解有限，难以利用实际的优化行为。从执行的角度来看，张量算子的优化是相互独立的，因此我们可以按任意顺序甚至不连续地优化它们。
- 即使有动态信息，也不清楚如何最好地推断估计的性能。鉴于优化结果，有动机采用“预测-然后优化”范式

### Lorien(SoCC 2021)

提出lorien，充当自动调整**深度学习框架和计算资源之间的抽象层**。
1. 提供了一个**分布式系统**来调整来自 Amazon EC2 实例或边缘设备上的各种自动调整框架的大量调整任务
2. 设计了一个通用数据模型，可以**适应来自各种自动调优框架的调优结果**
3. Lorien 中的**性能成本模型是通过自动机器学习**(AutoML) 对高级调度功能进行训练的，**支持零样本调整**（泛化性良好）

### TenSet (NeurIPS 2021)

提出了TenSet，一个大规模张量程序性能数据集，包含从 Intel CPU、AMD CPU、ARM CPU 和 NVIDIA GPU 的真实测量中收集的 5200 万条程序性能记录。

文章的Background里对发展现状总结的挺清晰的。

对各种特征进行建模的思路也值得借鉴。

### a-deep-learning-based-cost-model-for-automatic-code-optimization (MLSys 2021)

有非常大的局限，首先他只能用于为cpu生成模型，其次对于不同的cpu他需要重新生成数据训练模型（自己构造的数据）

优点是不依赖于提取复杂的特征

基于TIRAMISU实现

### One-shot Tuner(CC 2022)

采用受神经预测器启发的方法来减少自动调整开销，并表明在编译之前训练的性能预测器模型可以生成优化的张量操作代码，而**无需重复搜索和硬件测量**。
为了**生成样本高效的训练数据集**，我们扩展输入表示以包含特定于任务的信息，并指导数据采样方法专注于学习高性能代码。

### TLP(ASPLOS 23)

TLP 为张量程序调优设计了一种简单而有效的通用**特征提取**机制。(**转化为了NLP问题**)

MTL-TLP利用**multi-tasking**技术解决**离线成本模型跨硬件不可用**问题。

### AutoTensorIR
继autotvm, ansor之后第三代方法，但是现在还不可用
### 多面体模型


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

### xgb论文

### RAMMER

计算图的调度和算子内部调度是分开互不影响的，这两层分开进行调度会导致运行时的调度开销很大，算子间并行性没有被有效利用，以及**忽视了算子内和算子间两种并行性的相互影响**。

因此希望提出一种方法，避免这种问题，于是rammer因运而生。提出了一套新的抽象，将原来的**算子抽象为rOperator**，并将其进一步分解为**更小的调度单元rTask**，将底层的硬件抽象为由多个虚拟执行单元（virtualized execution units, vEU）组成的 **vDevice**。在这套新的抽象下，用户可以通过更细的 rTask 粒度将数据流图调度到多个 vEU 之上，兼顾了计算任务中的两种并行性与底层计算资源的协调。整个调度方案在编译期生成并“静态”映射到硬件计算单元上，因此可以天然地消除掉许多原本存在的调度开销。

### roller论文

提出的Roller解决了搜索时间长的问题，它有如下几个特点。

- 首先，**Roller不把DNN中的算子计算视为多层嵌套循环，而是视作数据处理管道**。其中数据块(tile) 在具有并行执行单元（如GPU SM）和内存层次结构抽象的硬件上移动和处理。生成高效的Kernel的目标变成了提高流水线吞吐量的目标。
- 然后，**为了使得基于数据块的流水线吞吐量最大化，要求每一级的数据块（Tile）shape都必须匹配（论文中叫对齐）硬件的参数设置**，比如memory bank, memory transaction length, 和 minimum schedulable unit (e.g., warp size in GPUs)这些和内存带宽以及并行度相关的设置。
  这个约束不仅可以使得张量程序在每一级内存中都拥有很好的计算效率，这还大大降低了以前以多重循环为基础的参数搜索空间，从而解决张量编译器在编译时因为搜索Schedule耗费的大量时间。
- 最后，**对齐硬件的数据处理管道的性能是高度可预测的**。因为内存吞吐量可以从硬件规范或者Benchmark测试得出，这大大简化了对不同硬件进行对齐后做性能估计的难度，并不再需要基于硬件去构建复杂的代价模型来估计性能。
  
基于这些想法，Roller提出了rTile，这是一种新的抽象，它封装了硬件加速器的关键特征以及输入张量shape一致的数据块（Tile）shape。然后**将数据处理管道描述为基于rTile的程序（rProgram），由Load, Store, Compute 三个接口组成，作用于rTile。**

### pet论文

现有的框架在图层做优化一般都是基于等价变换，也就时说变换前后的程序是完全等价的。这里等价的意思是给定相同的输入，那么变换前后的程序一定可以得到相同的输出。而这篇论文挖了一个新坑，即做了一个新的框架PET，在优化过程中允许出现部分等价的变换，并且设计了一个高效的搜索算法去组合完全等价以及部分等价的变换以探索更大的搜索空间。并且最终结果也比较好。

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

### Nimble (NeurIPS 2020)
现有的 DL 框架存在大的**调度开销**和不必要的**串行执行**的问题，导致效率低下

Nimble 引入了一种称为提前 (AoT) 调度的新技术。调度在执行 GPU 内核之前完成，从而消除了运行时的大部分调度开销。

核心思想就是充分利用gpu的并行能力，通过充分挖掘算子间的并行潜力实现。

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