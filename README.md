# DL-Complier-Navigation

## 还未读

### Simulating Execution Time of Tensor Programs using Graph Neural Networks (arxiv)

基于TVM，提出学习surrogate model来克服搜索configuration space耗时的问题。模型基于AST进行训练，使用graph convolutional network（GraphNN）来挖掘graph中的结构信息。它的优势是使用可学习的基于图的处理比基于heuristic的特征提取有优势。AST中的每个节点会通过一个shared encoder编码成一个固定长的特征，节点间的信息通过GraphNN来传递。这些信息会整合进一个固定长的向量，最后通过一个预测函数对性能进行预测。

### Tuna: A Static Analysis Approach to Optimizing Deep Neural Networks (arxiv)

使用张量计算性能的静态分析（编译时）方法，基于analytical cost model预测tensor program性能。它基于TVM实现（重用了TE, IR与codegen）。静态分析的优点是可以在多核CPU机器上运行。它基于目标硬件特点构建特征。Tuna既分析program IR也分析low-level生成代码（assembly codes），来提取目标硬件相关的特征（如SIMD指令数量、CPU cache局部性、GPU shared memory使用等）。如GPU cost model会考虑PTX指令数量，thread level parallelism, workload per thread, SM occupancy, warp latency hiding, shared memory bank conflict。Program的performance score这些特征的线性模型（其参数是per-hardware的，通过profiling确定）。根据这些特征就可以预测tensor program上一组transformation的相对性能。搜索方法使用了evolution strategy（ES）。实验中的metrics包括compilation time/cost与inference latency。与AutoTVM相比，它可以只用1.65%的编译时间达到相当的性能。

### Bayesian Optimization for auto-tuning GPU kernels (pmbs)

采用Bayesian optimization（BO）用于GPU平台上的kernel auto-tuning。文中引入了一种contextual variance exploration factor与一种新的acquisition function，结合acquisition function选择机制来克服搜索空间离散且复杂，以及包含invalid configurations等挑战。BO包含几个基本组件：objective function，surrogate model与acquisition function。由于objective function未知且评估通常很费时，Surrogate model用于模拟objective function，一般求值没有objective function那么费时。本文方法使用Gaussian processes（GP）作为surrogate model。GP中的covariance function中的kernel使用Matérn kernel。Acquisition function在surrogate model上优化，给出下一次要在objective function上评估的parameter configuration。它需要考虑exploration与exploitation间的trade-off，原则上它会选取未知的区域或是有性能提升潜力的配置。Acquisition function通常有Probability of Improvement, Expected Improvement和Upper Confidence Bound。Acquisition function中的exploration factor一般置为常数，文中采用contextual variance在每一次评估中根据surrogate model的状态设置该参数。文中提出multi与advanced multi两种acquisition function，一开始，会有多个acquisition functions，在搜索过程中当给出重复建议时会将其忽略。这样就可以根据特定问题选取最优的acquisition function。实验中与Kernel Tuner框架中其它的方法（Simulated Annealing, Multi-start Local Search, Genetic Algorithm），以及其它的BO框架（BayesianOptimization，Scikit-optimize）作了比较。在OpenCL GEMM，CUDA 2D Convolution，及异构point-in-polygon三个GPU kernel上，本中方法有更优的表现。

### DietCode: Automatic Optimization for Dynamic Tensor Programs (mlsys 22)

基于TVM中的Ansor，TVM中现有的自动调优机制Ansor要求workload是static的。对于dynamic-shape workload则会导致tuning时间很长（对每个shape都tuning一把）。一些用于dynamic shape支持的扩展面临不够自动或者产生的kernel性能较差的问题。如Selective tuning需要专家经验。Nimble针对large shape进行tuning，然后将得到的schedule应用到所有的shape上。但在large shape上最优的schedule未必对其它也是最优的（因为padding等原因）。Bucketing会引入padding开销与冗余计算。DietCode基于完整程序是由micro-kernels组成的这个观察，为dynamic shape构建由micro-kernels组成的shape-generic搜索空间，然后对于所有shape在这个统一的空间中进行联合搜索。相应地，文中构建了micro-kernel-based cost model。它分为两个部分：一部分是shape-generic的cost function，预测micro-kernel的cost；另一部分对应shape-dependent的adaption cost function（这部分可通过occupancy与padding ratio计算，不需要feature extraction）。当调优结束，会输出一组micro-kernels。为了将所有可能的shape dispatch到这些micro-kernels，Automatic dispatching机制基于cost model针对特定shape对每个micro-kernel进行投票，然后训练decision tree自动地将选取同一micro-kernel的shape归为一类，并生成源码用于运行时shape到micro-kernel的分派。对于dynamic shape的workload它比现有的auto-scheduling机制tuning时间减少数倍，同时性能上也比现有auto-scheduling与vendor library有所提升。

### AKG: Automatic Kernel Generation for Neural Processing Units Using Polyhedral Transformations

AKG使用auto-tuner来找tiling策略。它使用machine learning guided sampling approach来减少tuning空间。AKG先根据一定的策略计算包含合法tiling参数的tuning空间，然后第一轮采样一些样本，并测其性能。这些样本用来训练机器学习模型，并产生第二批样本。第二轮的样本是从第一轮中最好的N个加上改动而来。这个改动以一定概率根据机器学习模型往高性能的方向前进，或者随机选取。第二批样本也会测量其性能并用来更新模型

## Tuning and Schedule

### Learning to Optimize Tensor Program (AtuoTVM. NeurIPS 2018)

解空间：`S`
具体调度：`s`
算子表达式：`e`
生成的表达式：`x=g(e,s)`
硬件上的性能：`f(g(e,s))`
成本模型上的性能：`f^(g(e,s))`
最终的目标就是`min(f(g(e,s)))`

* explore得到的`s`将会生成`f^(g(e,s))`，`f^()`是`模拟退火`的`energy function`，所以一些s会被筛选掉，将一些比较有希望的放到硬件上生成`f(g(e,s))`。那么此时我们就有了一个打上了标签的数据，再利用此数据训练更新cost model。cost model训练的越好，通过模拟退火得到的配置就越好。当得到一定数量的数据后（数量取决于模拟退火迭代次数），从中选取性能最好的配置作为最终结果。
* 使用了并行的马尔科夫模型来提升cost model的吞吐量
* 迁移学习
  * The key to transfer learning is to create a **transferable representation** that is **invariant** to the source and target domains
  * To leverage this invariance, our cost model f^(x) takes the **low-level loop AST x** as input.

### FlexTensor: An Automatic Schedule Exploration and Optimization Framework for Tensor Computation on Heterogeneous System (ASPLOS 2020)

**用户无需手动编写任何计划或模板。**

FlexTensor 由前端和后端两部分组成。 

- 前端用 Python 编写的张量计算作为输入。采用**静态分析**分析计算模式并生成特定于硬件的**调度空间**。
- 后端利用**启发式和机器学习**相结合的方法来找到优化的调度配置。启发式方法基于**模拟退火**，机器学习算法基于 **Q-Learning**。

### ANSOR: Generating High-Performance Tensor Programs for Deep Learning (OSDI 2020)

这篇论文做出了以下贡献：- 一种为计算图生成张量程序的**大型空间分层搜索机制**。

- 一种基于**可学习的代价模型的进化策略**，用来微调张量化程序的性能。

- 一种基于**梯度下降的调度算法**，在优化 DNN 的端到端性能时对重要子图进行优先排序。

- Ansor 系统的实现和综合评估表明，上述技术在各种 DNN 和硬件平台上都优于最先进的系统。

**ANSOR 的 Pipeline 可以分为如下几个步骤：**- **Schedule Task：将完整的计算图划分成多个子图，对热点的子图进行重点优化**

- **Sketch：提取算子中的高层次特征，对算子进行较粗粒度的优化，确定代码的基本结构**

- **Annotation：随机初始化 Tilling Size 和一些 for 循环的策略，获得计算图的完整表示**

- **Evolutionary：训练 Cost Model，根据 Cost Model 对代码的性能进行评估，选取评估中取得高分数的一组实现，再通过运行时模块，获取 ground truth 和实际性能，选取实际性能最优的实现作为 ANSOR 的输出。**

### FamilySeer: Towards Optimized Tensor Codes by Exploiting Computation Subgraph Similarity (arxiv)

是一个基于TVM的auto-tuning框架。文中指出现有方法的问题是training sample与time budget无法很好地利用。FamilySeer将subgraph聚类成family，同类中的subgraph可以共享training sample与time budget。Ansor中的cost model对所有的subgraph是一个，忽略了subgraph的不同特性。
它对Ansor进行了改进，基本思想是挖掘subgraph间的相似性，将subgraph合并成family，对每个subgraph family构建cost model。在训练每个subgraph family 的时候，充分考虑了 time budget 的问题，优先选择优化潜力最大的 graph 所在的 family 进行调优。这样一个subgraph的tuning也能用于同一faimily的另外subgraph。另外，还实现了cost model的并行训练，以及GPU上的cost measurement。与Ansor相比，FamilySeer在搜索效率上可以有平均约2-3x的性能提升，同时达到相同性能。

### Woodpecker-DL (gtc)

研究了基于**遗传算法和强化学习**的两种新的自动搜索方法，以寻找针对特定硬件的最佳运算符代码配置

### AdaTune: Adaptive tensor program compilation made efficient (NeurIPS 2020)

- 提出了一种**自适应评估方法**，可以在统计上提前终止昂贵的硬件测量，而不会损失太多的准确性。
- 设计了一个**具有不确定性量化的代理模型**，使优化能够更好地适应硬件和模型异质性。
- 引入了一个**上下文优化器**，它提供了对 exploration exploitation的自适应控制，以提高转换空间搜索的有效性。

### ProTuner: Tuning Programs with Monte Carlo Tree Search (arxiv)

搜索算法使用蒙特卡洛树

### Chameleon: Adaptive Code Optimization for Expedited Deep Neural Network Compilation (ICLR 2020)

1. 设计一个**自适应探索模块**，利用**强化学习**来适应新网络的看不见的设计空间，以减少搜索时间，同时获得更好的性能。
2. 提出一种**自适应采样算法**，该算法利用**聚类**来自适应地减少昂贵的硬件测量次数，并设计一种受领域知识启发的**样本合成**，以找到可能产生更好性能的配置。
- 之前的算法，如此长的优化时间是由于模拟退火的低效（虽然它随机保证在大量迭代后能得到合理解决方案），他未能捕捉到设计空间中可以在搜索过程中利用的模式。
- 大部分优化时间都花在了真实硬件上的测量上，这些测量被用作上述搜索的反馈。
- 目前的方法会遭受大量无效配置的困扰，这不仅浪费了编译器开始时有限的硬件测量预算，而且还会产生严重的开销来重置目标硬件以进行后续硬件测量。
- 重要的是，为硬件测量选择潜在配置的采样机制更智能，以确保每次测量都最大限度地提高获得良好解决方案的机会，并避免无效配置。然而，目前的方法依赖于贪婪抽样，它根据成本模型的估计进行被动抽样。这不仅有过度拟合的趋势，而且还忽略了解决方案分布不均匀以及存在许多无效配置。

### DYNATUNE: Dynamic Tensor Program Optimization in Deep Neural Network Compilation (ICLR 2021)

考虑了用于张量规划优化问题的多臂老虎机 (MAB) 模型。使用 UCB 来处理基于时隙的优化的决策，并且设计了一个贝叶斯信念模型，该模型允许通过不确定性量化来预测每个算子的潜在性能增益，从而指导优化过程。
能够在获得相同优化结果的情况下，减少优化时间。

- 现有的 DL 编译侧重于加快单个张量算子而不是整个模型的收敛速度，导致收敛速度慢，优化时间长。
- 静态调度对张量程序的了解有限，难以利用实际的优化行为。从执行的角度来看，张量算子的优化是相互独立的，因此我们可以按任意顺序甚至不连续地优化它们。
- 即使有动态信息，也不清楚如何最好地推断估计的性能。鉴于优化结果，有动机采用“预测-然后优化”范式

主要针对time allocation问题。即如何充分利用编译的时间，时间花可能大提升性能的地方，从而提升整个模型的tuning收敛速度。分析了DL编译器几点挑战：1. 现有方法关注单算子收敛速度（非整模型）。2. 静态调度不够。3. 即使用动态信息，难以extrapolate estimated performance。针对这些问题，DynaTune使用MAB（Multi-Armed Bandit）模型，目标是设计scheduler使cumulative regret最小。UCB（upper confidence bound）用来处理time-slot-based optimization中的决策问题。模型中需要知道maximum latency reduction的算子，而它实际中无法得到。因此文中设计了Bayesian belief model（+MCMC）来预测每个算子的潜在的performance gain，其中的uncertainty信息可以用于指导搜索。实验中它与static schedule（Random, Round-robin与Linear）和dynamic scheme（Dynamic allocation + Random selection, Dynamic allocation + Round-robin selection）做了比较。

### Lorien: Efficient Deep Learning Workloads Delivery (SoCC 2021)

提出lorien，充当自动调整**深度学习框架和计算资源之间的抽象层**。

1. 提供了一个**分布式系统**来调整来自 Amazon EC2 实例或边缘设备上的各种自动调整框架的大量调整任务
2. 设计了一个通用数据模型，可以**适应来自各种自动调优框架的调优结果**
3. Lorien 中的**性能成本模型是通过自动机器学习**(AutoML) 对高级调度功能进行训练的，**支持零样本调整**（泛化性良好）

### TenSet: A Large-scale Program Performance Dataset for Learned Tensor Compilers (NeurIPS 2021)

提出了TenSet，一个大规模张量程序性能数据集，包含从 Intel CPU、AMD CPU、ARM CPU 和 NVIDIA GPU 的真实测量中收集的 5200 万条程序性能记录。

文章的Background里对发展现状总结的挺清晰的。

对各种特征进行建模的思路也值得借鉴。

### a-deep-learning-based-cost-model-for-automatic-code-optimization (MLSys 2021)

有非常大的局限，首先他只能用于为cpu生成模型，其次对于不同的cpu他需要重新生成数据训练模型（自己构造的数据）

优点是不依赖于提取复杂的特征

基于TIRAMISU实现

### Value Learning for Throughput Optimization of Deep Neural Workloads(MLSys 2021)

将调度过程建模为一系列优化选择，并提出了一种新技术来准确预测部分调度的预期性能。

使用 LSTM 建模算子和当前调度选择的特征。利用这些预测，能够做出优化决策，并且无需在目标硬件上执行任何操作，即可快速确定有效的调度。

这篇文章提出一种预测partial schedule期望性能的方法。具体地，它将选取最优schedule的问题建模为Markov Decision Process（MDP），状态s_i为模型前i层的schedule，即partial schedule。动作a_i为给定s_i下第i + 1层的合法scheduling选项。为了求解MDP，就需要一个值函数的估计。这个值函数的作用是预测给定状态下如果后续都采用最优schedule能达到的最小执行时间。实现中它基于LSTM，输入为两类特征：与schedule无关的intrinsic特征，以及schedule相关acquired特征。然后通过强化学习中的value iteration方法对值函数进行近似。其中对于每个状态，基于之前的值函数版本执行beam search，然后通过benchmarking得到性能，再对值函数进行修正。有了这个值函数后，就可以使用贪心方法进行优化了 ，且不需要在目标平台上执行，从而快速找到高效的schedule。实验中它找到的schedule执行性能优于Halide和TVM，同时搜索时间有2到3个数量级的加速（从小时级到秒级）。

### One-shot tuner for deep learning compilers (CC 2022)

采用受神经预测器启发的方法来减少自动调整开销，并表明在编译之前训练的性能预测器模型可以生成优化的张量操作代码，而**无需重复搜索和硬件测量**。
为了**生成样本高效的训练数据集**，我们扩展输入表示以包含特定于任务的信息，并指导数据采样方法专注于学习高性能代码。

### TLP: A Deep Learning-based Cost Model for Tensor Program Tuning (ASPLOS 23)

TLP 为张量程序调优设计了一种简单而有效的通用**特征提取**机制。(**转化为了NLP问题**)

MTL-TLP利用**multi-tasking**技术解决**离线成本模型跨硬件不可用**问题。

### AutoTensorIR

继autotvm, ansor之后第三代方法，但是现在还不可用

## Compiler and IR

### AI框架中的IR分析

参考这个https://zhuanlan.zhihu.com/p/263420069。从一个比较顶层的视角来看IR，包括编译器的IR以及深度学习编译的IR。有挺多东西看不懂的，里面说的不同表示能够取得不同的效果，我也不知道是什么原因。然后里面还提到了函数式编程的概念。

**性能上的优化(XLA/TVM/TC)**

性能上的优化思路其实比较统一，就是打开图和算子的边界，进行重新组合优化。

- **XLA**基本上的思路是把图层下发的子图中的算子全部打开成小算子，然后基于这张小算子组成的子图进行编译优化，包括buffer fusion、水平融合等，这里的关键是大算子怎样打开、小算子如何重新融合、新的大的算子(kernel)怎样生成，整体设计主要通过HLO/LLO/LLVM层层lowering实现，所有规则都是手工提前指定。

- **TVM**分为Relay和TVM两层，Relay主要关注图层，TVM主要关注算子层，总体思路与XLA是类似的，也是拿到前端给一张子图进行优化，Relay关注算子间的融合、TVM关注新的算子和kernel的生成，区别在于TVM是一个开放的架构，Relay目标是可以接入各种前端，TVM也是一个可以独立使用的算子开发和编译的工具（基于Halide IR，最新演进到自己定义的TIR），TVM在算子实现方面采用了compute和schedule分离的方案，开发人员通过compute来设计计算的逻辑，通过schedule来指定调度优化的逻辑。

- **TC**(Tensor Comprehensions)：开发者发现算子的计算逻辑的开发是比较容易的，但是schedule的开发非常困难，既要了解算法的逻辑又要熟悉硬件的体系架构，更重要的是，前面提到图算边界打开后，小算子融合后，会生成新的算子和kernel，这些新的算子compute是容易确定的（小算子compute的组合），但是schedule却很难生成，所以传统的方法就是事先定义一大堆schedule模板，万一组合的新算子不在模板之内，性能就可能比较差，甚至出错；那TC则希望通过Polyhedra model实现auto schedule，降低开发门槛，当然这个项目基本已经停更了，但是类似的工作在MLIR、MindSpore上还在不停发展。

### Rammer: Enabling Holistic Deep Learning Compiler Optimizations with rTasks (OSDI 2020)

计算图的调度和算子内部调度是分开互不影响的，这两层分开进行调度会导致运行时的调度开销很大，算子间并行性没有被有效利用，以及**忽视了算子内和算子间两种并行性的相互影响**。

因此希望提出一种方法，避免这种问题，于是rammer因运而生。提出了一套新的抽象，将原来的**算子抽象为rOperator**，并将其进一步分解为**更小的调度单元rTask**，将底层的硬件抽象为由多个虚拟执行单元（virtualized execution units, vEU）组成的 **vDevice**。在这套新的抽象下，用户可以通过更细的 rTask 粒度将数据流图调度到多个 vEU 之上，兼顾了计算任务中的两种并行性与底层计算资源的协调。整个调度方案在编译期生成并“静态”映射到硬件计算单元上，因此可以天然地消除掉许多原本存在的调度开销。

### PET: Optimizing Tensor Programs with Partially Equivalent Transformations and Automated Corrections(OSDI 2021)

现有的框架在图层做优化一般都是基于等价变换，也就时说变换前后的程序是完全等价的。这里等价的意思是给定相同的输入，那么变换前后的程序一定可以得到相同的输出。而这篇论文挖了一个新坑，即做了一个新的框架PET，在优化过程中允许出现部分等价的变换，并且设计了一个高效的搜索算法去组合完全等价以及部分等价的变换以探索更大的搜索空间。并且最终结果也比较好。

### Roller: Fast and Efficient Tensor Compilation for Deep Learning (OSDI 2022)

提出的Roller解决了搜索时间长的问题，它有如下几个特点。

- 首先，**Roller不把DNN中的算子计算视为多层嵌套循环，而是视作数据处理管道**。其中数据块(tile) 在具有并行执行单元（如GPU SM）和内存层次结构抽象的硬件上移动和处理。生成高效的Kernel的目标变成了提高流水线吞吐量的目标。
- 然后，**为了使得基于数据块的流水线吞吐量最大化，要求每一级的数据块（Tile）shape都必须匹配（论文中叫对齐）硬件的参数设置**，比如memory bank, memory transaction length, 和 minimum schedulable unit (e.g., warp size in GPUs)这些和内存带宽以及并行度相关的设置。
  这个约束不仅可以使得张量程序在每一级内存中都拥有很好的计算效率，这还大大降低了以前以多重循环为基础的参数搜索空间，从而解决张量编译器在编译时因为搜索Schedule耗费的大量时间。
- 最后，**对齐硬件的数据处理管道的性能是高度可预测的**。因为内存吞吐量可以从硬件规范或者Benchmark测试得出，这大大简化了对不同硬件进行对齐后做性能估计的难度，并不再需要基于硬件去构建复杂的代价模型来估计性能。

基于这些想法，Roller提出了rTile，这是一种新的抽象，它封装了硬件加速器的关键特征以及输入张量shape一致的数据块（Tile）shape。然后**将数据处理管道描述为基于rTile的程序（rProgram），由Load, Store, Compute 三个接口组成，作用于rTile。**

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

### IREE

IREE（Intermediate Representation Execution Environment，发音为“eerie”）是一个**基于 MLIR** 的**端到端编译器和运行时**，可将机器学习 (ML) 模型降低为统一的 IR，可向上扩展以满足数据中心和向下的需求以满足移动和边缘部署的约束和特殊考虑。
IREE 仍处于早期阶段。 我们已经确定了总体基础设施，并正在积极改进各种软件组件以及项目物流。 它距离日常使用还有很长的路要走。

### XLA

两个主流编译框架XLA（针对访存密集算子）和TVM（针对计算密集算子）

XLA采取了了一种相对比较朴素的技术路径。对于对自动CodeGen要求较高的计算密集型算子，如MatMul/Convolution等，会直接调用cuBLAS/cuDNN等Vendor Library；而对于除此之外的访存密集型算子，XLA会进行完全自动的Op Fusion和底层代码生成（CodeGen）。

除编译本身外，XLA还包含了一套静态的执行引擎。这个静态性体现在静态的Fixed Shape编译（ 即，在运行时为每一套输入shape进行一次完整编译并保留编译结果 ），静态的算子调度顺序，静态的显存/内存优化等方面，以期望获得更好的性能/存储优化结果。

XLA的主要性能收益来源可以概括为如下几个方面：

- 访存密集型算子的Op Fusion收益，这是目前在大多数业务中XLA最主要的收益来源；
- Fixed Shape架构下，TensorFlow计算图中的shape计算相关的子图会在编译时被分析为静态的编译时常量，节省执行时的节点数量。
- HLO层在比TensorFlow Graph的颗粒度上可能存在更大的图优化空间。

此外，XLA的架构也可以方便开发者扩展更多的图优化Pass，包括Layout优化和并发调度优化等等。

XLA的目的是帮助用户做通用，透明的一键式性能优化，提升Training／Inference时的Latency／Throughput等，整个过程完全自动。嵌入到宿主TensorFlow里执行的XLA会在原始计算图上自动圈出能够完整支持的子图，不能支持的个别算子可以继续通过TensorFlow的执行引擎来执行，因此对不同的计算图都有比较好的兼容性和可回退特性。从应用场景上，XLA不区分training或者inference，与TensorFlow的良好集成使得它可以实现对用户的完全透明。

[参考](https://zhuanlan.zhihu.com/p/163717035)

XLA的全称是Accelerated Linear Algebra，即加速线性代数。作为一种**深度学习编译器**，长期以来被作为Tensorflow框架的一个试验特性被开发，历时至今已经超过两三年了，随着Tensorflow 2.X的发布，XLA也终于从试验特性变成了默认打开的特性。此外， Pytorch社区也在大力推动XLA在Pytorch下的开发，现在已经有推出PyTorch/XLA TPU版本，暂只支持谷歌平台TPU上使用。

XLA使用JIT编译技术来分析用户在运行时创建的 TensorFlow 图，将TensorFlow Op转换成为HLO（High LevelOptimizer）中间表示并在HLO层上完成包括Op Fusion在内的多种图优化，最后基于LLVM完成CPU／GPU等机器代码的自动生成。

XLA的目的是帮助用户做通用，透明的一键式性能优化，提升Training／Inference时的Latency／Throughput等，整个过程完全自动。嵌入到宿主TensorFlow里执行的XLA会在原始计算图上自动圈出能够完整支持的子图，不能支持的个别算子可以继续通过TensorFlow的执行引擎来执行，因此对不同的计算图都有比较好的兼容性和可回退特性。**从应用场景上，XLA不区分training或者inference，与TensorFlow的良好集成使得它可以实现对用户的完全透明**。

参考：[华为开发者论坛](https://developer.huawei.com/consumer/cn/forum/topic/0201750315901780148?fid=0101592429757310384)


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

### nnfusion

### TACO

### Tensor Comprehension

采用polyhedral算法路线的编译器

### ngraph

### Tensor comprehension: framework-agnostic high-performance machine learning abstractions

## Graph-level Optimization

### Nimble: Lightweight and Parallel GPU Task Scheduling for Deep Learning (NeurIPS 2020)

现有的 DL 框架存在大的**调度开销**和不必要的**串行执行**的问题，导致效率低下

Nimble 引入了一种称为提前 (AoT) 调度的新技术。调度在执行 GPU 内核之前完成，从而消除了运行时的大部分调度开销。

核心思想就是充分利用gpu的并行能力，通过充分挖掘算子间的并行潜力实现。

## Graph Neural Network

由于GNN计算过程中的稀疏性，使得计算访存比的提升成为了优化的主要目标，而ML编译优化中的算子融合成为了典型的优化方案。
这方面的工作不少，比如DGL团队的featgraph、Graphiler、以及港中文James Chen团队的Seastar，编译优化后既能省内存，又能加速计算。从我在华为的工作经验来看，效果还是相当不错的。只不过现在的GNN训练中模型计算时间消耗本身占比整个训练pipeline的时间比较低，训练的性能瓶颈核心还是在图引擎上。

https://zhuanlan.zhihu.com/p/423889406

### Graphiler: Optimizing Graph Neural Networks with Message Passing Data Flow Graph (MLSys22)

图神经网络 (GNN) 是一类新的强大的机器学习模型，但简单的编程和高效的计算往往是矛盾的。 当前的 GNN 框架基于消息传递范式，并允许使用内置原语和用户定义函数 (UDF) 简洁地表达 GNN 模型。 虽然内置原语提供了高性能，但它们的表现力有限； UDF 很灵活，但通常性能低下并且使用过多的内存。 在本文中，我们提出了 Graphiler，这是一种用于 GNN 的编译器堆栈，它在提供 UDF 编程接口的灵活性的同时实现了高性能。 Graphiler 的核心是一种称为消息传递数据流图 (MP-DFG) 的新颖抽象，它可以实现优化，从而大大减少计算冗余和内存占用，并在统一框架下优化同构和异构 GNN。 实验表明，Graphiler 可以将 UDF GNN 加速多达两个数量级，并实现接近或优于专家实现的性能，并且可以节省大量内存。

https://zhuanlan.zhihu.com/p/570329623

### Seastar: vertex-centric programming for graph neural networks (Eurosys21)

图神经网络 (GNN) 在节点分类、链接预测和图聚类等图分析方面取得了突破性的性能。 已经开发了许多 GNN 训练框架，但它们通常被设计为一组手动编写的、GNN 特定的运算符插入现有的深度学习系统，这导致内存消耗高、数据局部性差以及算法设计和实现之间的语义差距大 . 本文提出了 Seastar 系统，它为 GPU 上的 GNN 训练提供了一个以顶点为中心的编程模型，并提供了惯用的 Python 结构，以便轻松开发新颖的同构和异构 GNN 模型。 我们还提出了新颖的优化方法，以产生高效的融合 GPU 内核，用于 GNN 训练中的前向和后向传递。 与最先进的 GNN 系统 DGL 和 PyG 相比，Seastar 实现了更好的可用性，内存消耗分别减少了 2 倍和 8 倍，执行速度分别提高了 14 倍和 3 倍。

### FeatGraph: A Flexible and Efficient Backend for Graph Neural Network Systems (SC20)

图神经网络 (GNN) 作为一种很有前途的图机器学习方法越来越受欢迎。 与每个顶点/边都与标量相关联的传统图形工作负载不同，GNN 将特征张量附加到每个顶点/边。 这种额外的特征维度，以及随之而来的更复杂的顶点和边缘计算，对局部性和并行性具有巨大影响，而现有的图形处理系统无法利用这些影响。

本文提出 FeatGraph 通过共同优化图遍历和特征维度计算来加速 GNN 工作负载。 FeatGraph 通过在每个顶点/边上将粗粒度的稀疏模板与细粒度的用户定义函数 (UDF) 组合在一起，提供了一个灵活的编程接口来表达不同的 GNN 模型。 FeatGraph 将图形遍历的优化合并到稀疏模板中，并允许用户使用特征维度表 (FDS) 指定 UDF 的优化。 FeatGraph 在 CPU 上将端到端 GNN 训练和推理速度提高了 32 倍，在 GPU 上提高了 7 倍。

https://leiblog.wang/%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB%EF%BD%9CFeatGraph/

### AKG

采用polyhedral算法路线的编译器
AKG已经用在了MindSpore的图算融合里面

### Diesel

### Stripe

## Polyhedral Model

### Tiramisu

采用polyhedral算法路线的编译器

#### 调度技术分类

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

### OpenVino

OpenVINO™ 是一个用于优化和部署人工智能（AI）推理的开源工具平台。它可以：

- 提高计算机视觉、自动语音识别、自然语言处理和其他常见任务的深度学习性能。

- 使用经过 TensorFlow、PyTorch 等流行框架训练的模型。

- 减少资源需求，并在从边缘到云的一系列英特尔®平台上高效部署

### xgb论文

### ONNX

### TorchScript

### mlir论文

### Bring You Own Codegen

### 并行计算

### 分布式推理

### Dynamic Shape

Static 编译会在运行时捕捉待编译子图的实际输入shape组合，并且为每一个输入shape组合生成一份编译结果。

Static Shape Compiler的优势显而易见，编译期完全已知静态shape信息的情况下，Compiler可以作出更好的优化决策并得到更好的CodeGen性能，同时也能够得到更好的显存/内存优化plan和调度执行plan；然而，Static Shape Compiler的缺点也十分明显，具体包括：

编译开销的增加。对于训练业务，编译开销导致训练迭代速度不稳定，训练初期显著负优化，甚至整个训练过程的时间开销负优化；对于Inference业务，很多业务实际部署和迭代时不允许出现性能抖动，而离线的预编译预热又会使得部署的过程变复杂。
内存显存占用的增加。除编译开销的问题之外，当shape变化范围特别大的时候，编译缓存额外占用的内存显存，经常导致实际部署环境下的内存/显存OOM，直接阻碍业务的实际落地。
对于一部分业务场景，shape变化范围可能非常大甚至是趋于无穷的，比较常见的包括广告推荐类业务中常见的稀疏化模型，还有例如分布式训练下的embedding切片等等。在这种情况下，编译缓存永远也无法收敛，用户也就不可能通过compiler获取到性能收益了。
上述问题在部分情况下，可以通过人工干预Compiler的圈图过程来缓解，即，将shape变化剧烈的子图排除在编译范围之外。然而，这种解决办法对用户非常不友好，大大降低了Compiler应用的通用性和透明性，这要求做部署和优化的同学同时对模型结构和compiler非常了解，且每一次模型结构迭代时，都需要花费额外的工作量来调整圈图获得可以接受的性能效果。

[参考](https://zhuanlan.zhihu.com/p/305546437)
## 编译器场景下的资源调度问题

### VELTAIR: Towards High-Performance Multi-tenant Deep Learning Services via Adaptive Compilation and Scheduling (ASPLOS 22)

- 粒度调度算法。提升资源利用率，降低调度conflict
  - 按model进行调度cpu利用率太低，按layer进行调度conflict太多。因此提出按block apdaptive进行调度
- 自适应编译策略。可以动态智能地选择独占或共享资源的程序，以减少整体干扰引起的性能损失。
  - 分析了一组深度学习层的易受干扰版本和干扰容忍代码版本之间的关系。 这些不同的版本本质上是并行性和局部性之间权衡的帕累托均衡。因此，提出了一种基于现有自动调度程序的single pass compiling strategy。 扩展的自动调度器会编译出用于不同干扰水平的不同版本。
- 场景：推理的计算量很小，所以通常一个服务器上会部署多个推理任务
- tvm之类的编译器只考虑单租户情况，在多租户情况下，性能下降很快

## 相关资料

[Deep Learning Systems Course (CMU; video)](https://www.youtube.com/channel/UC3-KIvmiIaZimgXMNt7F99g)

[机器学习系统：设计和实现 (book)](https://openmlsys.github.io/)

[动手学深度学习 (book; video)](https://zh-v2.d2l.ai/)

[实用机器学习 (video)](https://space.bilibili.com/1567748478/channel/collectiondetail?sid=28144)

[TVM官网 (blog)](https://chinese.tvm.wiki/)

[BBuf (blog)](http://giantpandacv.com/project/%E9%83%A8%E7%BD%B2%E4%BC%98%E5%8C%96/)

[Haskell (book)](http://learnyouahaskell.com/syntax-in-functions)

[论文汇总 (github)](https://github.com/merrymercy/awesome-tensor-compilers#open-source-projects)

[LLVM (book)](https://getting-started-with-llvm-core-libraries-zh-cn.readthedocs.io/zh_CN/latest/)
