
# Tuning and Schedule

## Learning to Optimize Tensor Program (AtuoTVM. NeurIPS 2018)

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

## FlexTensor: An Automatic Schedule Exploration and Optimization Framework for Tensor Computation on Heterogeneous System (ASPLOS 2020)

**用户无需手动编写任何计划或模板。**

FlexTensor 由前端和后端两部分组成。 

- 前端用 Python 编写的张量计算作为输入。采用**静态分析**分析计算模式并生成特定于硬件的**调度空间**。
- 后端利用**启发式和机器学习**相结合的方法来找到优化的调度配置。启发式方法基于**模拟退火**，机器学习算法基于 **Q-Learning**。

## ANSOR: Generating High-Performance Tensor Programs for Deep Learning (OSDI 2020)

这篇论文做出了以下贡献：

- 一种为计算图生成张量程序的**大型空间分层搜索机制**。

- 一种基于**可学习的代价模型的进化策略**，用来微调张量化程序的性能。

- 一种基于**梯度下降的调度算法**，在优化 DNN 的端到端性能时对重要子图进行优先排序。

- Ansor 系统的实现和综合评估表明，上述技术在各种 DNN 和硬件平台上都优于最先进的系统。

ANSOR 的 Pipeline 可以分为如下几个步骤：

- **Schedule Task**：将完整的计算图划分成多个子图，对热点的子图进行重点优化

- **Sketch**：提取算子中的高层次特征，对算子进行较粗粒度的优化，确定代码的基本结构

- **Annotation**：随机初始化 Tilling Size 和一些 for 循环的策略，获得计算图的完整表示

- **Evolutionary**：训练 Cost Model，根据 Cost Model 对代码的性能进行评估，选取评估中取得高分数的一组实现，再通过运行时模块，获取 ground truth 和实际性能，选取实际性能最优的实现作为 ANSOR 的输出。

## FamilySeer: Towards Optimized Tensor Codes by Exploiting Computation Subgraph Similarity (arxiv)

是一个基于TVM的auto-tuning框架。文中指出现有方法的问题是training sample与time budget无法很好地利用。FamilySeer将subgraph聚类成family，同类中的subgraph可以共享training sample与time budget。Ansor中的cost model对所有的subgraph是一个，忽略了subgraph的不同特性。

它对Ansor进行了改进，基本思想是挖掘subgraph间的相似性，将subgraph合并成family，对每个subgraph family构建cost model。在训练每个subgraph family 的时候，充分考虑了 time budget 的问题，优先选择优化潜力最大的 graph 所在的 family 进行调优。这样一个subgraph的tuning也能用于同一faimily的另外subgraph。另外，还实现了cost model的并行训练，以及GPU上的cost measurement。与Ansor相比，FamilySeer在搜索效率上可以有平均约2-3x的性能提升，同时达到相同性能。

## Woodpecker-DL (gtc)

研究了基于**遗传算法和强化学习**的两种新的自动搜索方法，以寻找针对特定硬件的最佳运算符代码配置

## AdaTune: Adaptive tensor program compilation made efficient (NeurIPS 2020)

- 提出了一种**自适应评估方法**，可以在统计上提前终止昂贵的硬件测量，而不会损失太多的准确性。
- 设计了一个**具有不确定性量化的代理模型**，使优化能够更好地适应硬件和模型异质性。
- 引入了一个**上下文优化器**，它提供了对 exploration exploitation的自适应控制，以提高转换空间搜索的有效性。

## ProTuner: Tuning Programs with Monte Carlo Tree Search (arxiv)

搜索算法使用蒙特卡洛树

## Chameleon: Adaptive Code Optimization for Expedited Deep Neural Network Compilation (ICLR 2020)

1. 设计一个**自适应探索模块**，利用**强化学习**来适应新网络的看不见的设计空间，以减少搜索时间，同时获得更好的性能。
2. 提出一种**自适应采样算法**，该算法利用**聚类**来自适应地减少昂贵的硬件测量次数，并设计一种受领域知识启发的**样本合成**，以找到可能产生更好性能的配置。
- 之前的算法，如此长的优化时间是由于模拟退火的低效（虽然它随机保证在大量迭代后能得到合理解决方案），他未能捕捉到设计空间中可以在搜索过程中利用的模式。
- 大部分优化时间都花在了真实硬件上的测量上，这些测量被用作上述搜索的反馈。
- 目前的方法会遭受大量无效配置的困扰，这不仅浪费了编译器开始时有限的硬件测量预算，而且还会产生严重的开销来重置目标硬件以进行后续硬件测量。
- 重要的是，为硬件测量选择潜在配置的采样机制更智能，以确保每次测量都最大限度地提高获得良好解决方案的机会，并避免无效配置。然而，目前的方法依赖于贪婪抽样，它根据成本模型的估计进行被动抽样。这不仅有过度拟合的趋势，而且还忽略了解决方案分布不均匀以及存在许多无效配置。

## DYNATUNE: Dynamic Tensor Program Optimization in Deep Neural Network Compilation (ICLR 2021)

考虑了用于张量规划优化问题的多臂老虎机 (MAB) 模型。使用 UCB 来处理基于时隙的优化的决策，并且设计了一个贝叶斯信念模型，该模型允许通过不确定性量化来预测每个算子的潜在性能增益，从而指导优化过程。
能够在获得相同优化结果的情况下，减少优化时间。

- 现有的 DL 编译侧重于加快单个张量算子而不是整个模型的收敛速度，导致收敛速度慢，优化时间长。
- 静态调度对张量程序的了解有限，难以利用实际的优化行为。从执行的角度来看，张量算子的优化是相互独立的，因此我们可以按任意顺序甚至不连续地优化它们。
- 即使有动态信息，也不清楚如何最好地推断估计的性能。鉴于优化结果，有动机采用“预测-然后优化”范式

主要针对time allocation问题。即如何充分利用编译的时间，时间花可能大提升性能的地方，从而提升整个模型的tuning收敛速度。分析了DL编译器几点挑战：

1. 现有方法关注单算子收敛速度（非整模型）。
2. 静态调度不够。
3. 即使用动态信息，难以extrapolate estimated performance。

针对这些问题，DynaTune使用MAB（Multi-Armed Bandit）模型，目标是设计scheduler使cumulative regret最小。UCB（upper confidence bound）用来处理time-slot-based optimization中的决策问题。模型中需要知道maximum latency reduction的算子，而它实际中无法得到。因此文中设计了Bayesian belief model（+MCMC）来预测每个算子的潜在的performance gain，其中的uncertainty信息可以用于指导搜索。实验中它与static schedule（Random, Round-robin与Linear）和dynamic scheme（Dynamic allocation + Random selection, Dynamic allocation + Round-robin selection）做了比较。

## TenSet: A Large-scale Program Performance Dataset for Learned Tensor Compilers (NeurIPS 2021)

提出了TenSet，一个大规模张量程序性能数据集，包含从 Intel CPU、AMD CPU、ARM CPU 和 NVIDIA GPU 的真实测量中收集的 5200 万条程序性能记录。

文章的Background里对发展现状总结的挺清晰的。

对各种特征进行建模的思路也值得借鉴。

## a-deep-learning-based-cost-model-for-automatic-code-optimization (MLSys 2021)

有非常大的局限，首先他只能用于为cpu生成模型，其次对于不同的cpu他需要重新生成数据训练模型（自己构造的数据）

优点是不依赖于提取复杂的特征

基于TIRAMISU实现

## Value Learning for Throughput Optimization of Deep Neural Workloads(MLSys 2021)

将调度过程建模为一系列优化选择，并提出了一种新技术来准确预测部分调度的预期性能。

使用 LSTM 建模算子和当前调度选择的特征。利用这些预测，能够做出优化决策，并且无需在目标硬件上执行任何操作，即可快速确定有效的调度。

这篇文章提出一种预测partial schedule期望性能的方法。具体地，它将选取最优schedule的问题建模为Markov Decision Process（MDP），状态s_i为模型前i层的schedule，即partial schedule。动作a_i为给定s_i下第i + 1层的合法scheduling选项。为了求解MDP，就需要一个值函数的估计。这个值函数的作用是预测给定状态下如果后续都采用最优schedule能达到的最小执行时间。实现中它基于LSTM，输入为两类特征：与schedule无关的intrinsic特征，以及schedule相关acquired特征。然后通过强化学习中的value iteration方法对值函数进行近似。其中对于每个状态，基于之前的值函数版本执行beam search，然后通过benchmarking得到性能，再对值函数进行修正。有了这个值函数后，就可以使用贪心方法进行优化了 ，且不需要在目标平台上执行，从而快速找到高效的schedule。实验中它找到的schedule执行性能优于Halide和TVM，同时搜索时间有2到3个数量级的加速（从小时级到秒级）。

## One-shot tuner for deep learning compilers (CC 2022)

采用受神经预测器启发的方法来减少自动调整开销，并表明在编译之前训练的性能预测器模型可以生成优化的张量操作代码，而**无需重复搜索和硬件测量**。
为了**生成样本高效的训练数据集**，我们扩展输入表示以包含特定于任务的信息，并指导数据采样方法专注于学习高性能代码。

## TLP: A Deep Learning-based Cost Model for Tensor Program Tuning (ASPLOS 23)

TLP 为张量程序调优设计了一种简单而有效的通用**特征提取**机制。(**转化为了NLP问题**)

MTL-TLP利用**multi-tasking**技术解决**离线成本模型跨硬件不可用**问题。

## Metaschedule. Tensor Program Optimization with Probabilistic Programs (NeurIPS 22)

优化效果依赖于搜索空间大小和搜索算法的效率。但是大多数现有的方法无法灵活的让领域专家扩大搜索空间。
MetaSchedule，一种特定于领域的**概率编程语言抽象**，用于构建构建搜索空间。
- 允许领域专家分析程序，并以模块（将多个transformation组合起来）为单位进行随机选择，组合起来对程序进行转换。
- 构建了一个端到端的learning-driven框架对给定的搜索空间进行搜索。 

当前问题

- **手动调度**：开发人员通过手动调用调度原语来优化他们的程序，即在循环中探索设计空间中的点，是一种繁琐且容易出错的方法
- **AutoTVM**：要求用户定义“调度模板”作为调度空间。
- **AutoScheduler (Ansor)**：它根据一组预定义的“搜索规则”自动生成计划模板作为设计空间。 然而，将 AutoScheduler 扩展到新的调度原语（张量化、循环分区、软件流水线）并非易事。
上面的三个系统都有独立的 API 集，这些 API 有几层自己的抽象，不仅难以学习，而且定制起来也需要大量工程。

meta schedule的好处

- 简洁的语法、与 TensorIR 调度一致的 API，没有其他抽象层。
- 提供统一的API，用于实现手动调度、AutoTVM和AutoScheduler（Ansor）。
- 对所有调度原语的可扩展性，包括张量化和循环分区。 在自动调整中使用新原语几乎不需要额外的努力。
- 自动化基础设施在每一层都是可定制的。

## Heron: Automatically Constrained High-Performance Library Generation for Deep Learning Accelerators (ASPLOS 23)

无法从大型但质量低下的搜索空间中找到最优或接近最优的程序，因为硬件的大量固有约束无法准确表征。

关键是在整个程序生成（包括空间生成和空间探索）中自动实施大量复杂而准确的约束（剪枝）。 

- 通过对计算进行静态分析，自动生成约束，从而修剪无效候选以产生高质量的搜索空间。 
- 为了有效地探索搜索空间，进一步提出了一种新的基于约束的遗传算法，其特点是进化过程是在制定的约束满足问题而不是具体解决方案上进行的。因此，在整个探索过程中严格保留了搜索空间的复杂约束。

## Tuna: A Static Analysis Approach to Optimizing Deep Neural Networks (arxiv)

第一次见到使用静态分析的。

现在许多深度学习模型都使用动态分析，依赖于在设备上sample来构建cost model。 需要在编译时访问目标硬件，而且会导致机器资源的巨大成本

提出一种方法，该方法通过按顺序基于目标硬件构建特征来分析程序。 使用张量运算的相对性能的静态分析来优化深度学习程序。 实验表明，与具有相同编译时间的基于动态分析的方法相比，我们的方法可以实现高达 11 倍的性能。

## Bayesian Optimization for auto-tuning GPU kernels (arxiv)

我们演示了如何处理包含无效配置的粗糙、离散、受限的搜索空间。 我们引入了一种新的上下文方差探索因子，以及具有改进的可扩展性的新采集函数，并结合了一种知情的采集函数选择机制。

通过将我们的贝叶斯优化实现在各种测试用例上的性能与 Kernel Tuner 中的现有搜索策略以及其他贝叶斯优化实现进行比较，我们证明我们的搜索策略具有很好的泛化能力，并且在很大程度上始终优于其他搜索策略。