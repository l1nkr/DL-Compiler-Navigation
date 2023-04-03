
## AKG

采用polyhedral算法路线的编译器
AKG已经用在了MindSpore的图算融合里面

## Diesel

## Stripe


## xgb论文

## ONNX

## TorchScript

## mlir论文


## 并行计算

## 分布式推理


## Tuna: A Static Analysis Approach to Optimizing Deep Neural Networks (arxiv)

使用张量计算性能的静态分析（编译时）方法，基于analytical cost model预测tensor program性能。它基于TVM实现（重用了TE, IR与codegen）。静态分析的优点是可以在多核CPU机器上运行。它基于目标硬件特点构建特征。Tuna既分析program IR也分析low-level生成代码（assembly codes），来提取目标硬件相关的特征（如SIMD指令数量、CPU cache局部性、GPU shared memory使用等）。如GPU cost model会考虑PTX指令数量，thread level parallelism, workload per thread, SM occupancy, warp latency hiding, shared memory bank conflict。Program的performance score这些特征的线性模型（其参数是per-hardware的，通过profiling确定）。根据这些特征就可以预测tensor program上一组transformation的相对性能。搜索方法使用了evolution strategy（ES）。实验中的metrics包括compilation time/cost与inference latency。与AutoTVM相比，它可以只用1.65%的编译时间达到相当的性能。

## Bayesian Optimization for auto-tuning GPU kernels (pmbs)

采用Bayesian optimization（BO）用于GPU平台上的kernel auto-tuning。文中引入了一种contextual variance exploration factor与一种新的acquisition function，结合acquisition function选择机制来克服搜索空间离散且复杂，以及包含invalid configurations等挑战。BO包含几个基本组件：objective function，surrogate model与acquisition function。由于objective function未知且评估通常很费时，Surrogate model用于模拟objective function，一般求值没有objective function那么费时。本文方法使用Gaussian processes（GP）作为surrogate model。GP中的covariance function中的kernel使用Matérn kernel。Acquisition function在surrogate model上优化，给出下一次要在objective function上评估的parameter configuration。它需要考虑exploration与exploitation间的trade-off，原则上它会选取未知的区域或是有性能提升潜力的配置。Acquisition function通常有Probability of Improvement, Expected Improvement和Upper Confidence Bound。Acquisition function中的exploration factor一般置为常数，文中采用contextual variance在每一次评估中根据surrogate model的状态设置该参数。文中提出multi与advanced multi两种acquisition function，一开始，会有多个acquisition functions，在搜索过程中当给出重复建议时会将其忽略。这样就可以根据特定问题选取最优的acquisition function。实验中与Kernel Tuner框架中其它的方法（Simulated Annealing, Multi-start Local Search, Genetic Algorithm），以及其它的BO框架（BayesianOptimization，Scikit-optimize）作了比较。在OpenCL GEMM，CUDA 2D Convolution，及异构point-in-polygon三个GPU kernel上，本中方法有更优的表现。

## DietCode: Automatic Optimization for Dynamic Tensor Programs (mlsys 22)

基于TVM中的Ansor，TVM中现有的自动调优机制Ansor要求workload是static的。对于dynamic-shape workload则会导致tuning时间很长（对每个shape都tuning一把）。一些用于dynamic shape支持的扩展面临不够自动或者产生的kernel性能较差的问题。如Selective tuning需要专家经验。Nimble针对large shape进行tuning，然后将得到的schedule应用到所有的shape上。但在large shape上最优的schedule未必对其它也是最优的（因为padding等原因）。Bucketing会引入padding开销与冗余计算。DietCode基于完整程序是由micro-kernels组成的这个观察，为dynamic shape构建由micro-kernels组成的shape-generic搜索空间，然后对于所有shape在这个统一的空间中进行联合搜索。相应地，文中构建了micro-kernel-based cost model。它分为两个部分：一部分是shape-generic的cost function，预测micro-kernel的cost；另一部分对应shape-dependent的adaption cost function（这部分可通过occupancy与padding ratio计算，不需要feature extraction）。当调优结束，会输出一组micro-kernels。为了将所有可能的shape dispatch到这些micro-kernels，Automatic dispatching机制基于cost model针对特定shape对每个micro-kernel进行投票，然后训练decision tree自动地将选取同一micro-kernel的shape归为一类，并生成源码用于运行时shape到micro-kernel的分派。对于dynamic shape的workload它比现有的auto-scheduling机制tuning时间减少数倍，同时性能上也比现有auto-scheduling与vendor library有所提升。

## AKG: Automatic Kernel Generation for Neural Processing Units Using Polyhedral Transformations

AKG使用auto-tuner来找tiling策略。它使用machine learning guided sampling approach来减少tuning空间。AKG先根据一定的策略计算包含合法tiling参数的tuning空间，然后第一轮采样一些样本，并测其性能。这些样本用来训练机器学习模型，并产生第二批样本。第二轮的样本是从第一轮中最好的N个加上改动而来。这个改动以一定概率根据机器学习模型往高性能的方向前进，或者随机选取。第二批样本也会测量其性能并用来更新模型