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

### xgb论文

### RAMMER

计算图的调度和算子内部调度是分开互不影响的，这两层分开进行调度会导致运行时的调度开销很大，算子间并行性没有被有效利用，以及忽视了算子内和算子间两种并行性的相互影响。

因此希望提出一种方法，避免这种问题，于是rammer因运而生。提出了一套新的抽象，将原来的算子抽象为rOperator，并将其进一步分解为更小的调度单元rTask，将底层的硬件抽象为由多个虚拟执行单元（virtualized execution units, vEU）组成的 vDevice。在这套新的抽象下，用户可以通过更细的 rTask 粒度将数据流图调度到多个 vEU 之上，兼顾了计算任务中的两种并行性与底层计算资源的协调。整个调度方案在编译期生成并“静态”映射到硬件计算单元上，因此可以天然地消除掉许多原本存在的调度开销。

这个机制可以看懂，但是为什么这样做就可以消除掉之前的缺点呢？

*这个东西和哪个nnfusion之间到底是什么关系呢？明明是这篇论文先发表，但是这篇论文里面说nnfusion是rammer的基础。nnfusion发表在2022的OSDI里面，rammer在2020就已经发表了。*

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

### FlexTensor

### ngraph

### Learning to optimize tensor programs

### Tensor comprehension: framework-agnostic high-performance machine learning abstractions

### Tiramisu

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

参考这个[https://zhuanlan.zhihu.com/p/263420069。从一个比较顶层的视角来看IR，包括编译器的IR以及深度学习编译的IR。有挺多东西看不懂的，里面说的不同表示能够取得不同的效果，我也不知道是什么原因。然后里面还提到了函数式编程的概念。](https://zhuanlan.zhihu.com/p/263420069%E3%80%82%E4%BB%8E%E4%B8%80%E4%B8%AA%E6%AF%94%E8%BE%83%E9%A1%B6%E5%B1%82%E7%9A%84%E8%A7%86%E8%A7%92%E6%9D%A5%E7%9C%8BIR%EF%BC%8C%E5%8C%85%E6%8B%AC%E7%BC%96%E8%AF%91%E5%99%A8%E7%9A%84IR%E4%BB%A5%E5%8F%8A%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E7%BC%96%E8%AF%91%E7%9A%84IR%E3%80%82%E6%9C%89%E6%8C%BA%E5%A4%9A%E4%B8%9C%E8%A5%BF%E7%9C%8B%E4%B8%8D%E6%87%82%E7%9A%84%EF%BC%8C%E9%87%8C%E9%9D%A2%E8%AF%B4%E7%9A%84%E4%B8%8D%E5%90%8C%E8%A1%A8%E7%A4%BA%E8%83%BD%E5%A4%9F%E5%8F%96%E5%BE%97%E4%B8%8D%E5%90%8C%E7%9A%84%E6%95%88%E6%9E%9C%EF%BC%8C%E6%88%91%E4%B9%9F%E4%B8%8D%E7%9F%A5%E9%81%93%E6%98%AF%E4%BB%80%E4%B9%88%E5%8E%9F%E5%9B%A0%E3%80%82%E7%84%B6%E5%90%8E%E9%87%8C%E9%9D%A2%E8%BF%98%E6%8F%90%E5%88%B0%E4%BA%86%E5%87%BD%E6%95%B0%E5%BC%8F%E7%BC%96%E7%A8%8B%E7%9A%84%E6%A6%82%E5%BF%B5%E3%80%82)

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