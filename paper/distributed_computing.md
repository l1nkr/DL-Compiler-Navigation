# 分布式场景

## Supporting Very Large Models using Automatic Dataﬂow Graph Partitioning(eurosys 2019)

本文介绍了 Tofu，这是一个将非常大的 DNN 模型跨多个 GPU 设备进行分区以减少每个 GPU 内存占用的系统。 Tofu 旨在对 MXNet 和 TensorFlow 等平台使用的细粒度张量运算符的数据流图进行分区。 为了自动划分每个运算符，我们建议用一种受 Halide 启发的简单语言来描述运算符的语义。 为了在数据流图中最佳地划分不同的操作符，Tofu 使用递归搜索算法来最小化总通信成本。 我们在 8 GPU 机器上的实验表明，Tofu 可以训练非常大的 CNN 和 RNN 模型。 与训练超大型模型的替代方法相比，它还实现了 25% - 400% 的加速。

## Lorien: Efficient Deep Learning Workloads Delivery (SoCC 21)

提出lorien，充当自动调整**深度学习框架和计算资源之间的抽象层**。

1. 提供了一个**分布式系统**来调整来自 Amazon EC2 实例或边缘设备上的各种自动调整框架的大量调整任务
2. 设计了一个通用数据模型，可以**适应来自各种自动调优框架的调优结果**
3. Lorien 中的**性能成本模型是通过自动机器学习**(AutoML) 对高级调度功能进行训练的，**支持零样本调整**（泛化性良好）

## Varuna: Scalable, Low-cost Training of Massive Deep Learning Models (EuroSys 22)

由于机器学习硬件资源需要，通常会在大规模集群上进行训练。集群间的连接如NV-Link和Infiniband会成为性能瓶颈，导致
- scalability limits on job parallelism; 
- resource fragmentation across hyperclusters.
  
提出了Varuna
- 允许在商用网络上训练大规模深度学习模型。
- Varuna能降低网络资源使用并能够自动配置用户的训练任务来充分利用硬件资源。
- Varuna 能够利用成本比专用 GPU 便宜约 5 倍的“低优先级”虚拟机，从而显着降低训练大规模模型的成本。

## Distributed halide(ppopp16)

Halide 使用简单的语言结构来表达要计算的内容，并使用单独的调度协同语言来表达何时何地执行计算，这种方法已证明性能与手动优化代码相当或更好。 然而，到目前为止，Halide 一直局限于并行共享内存执行，限制了它在内存带宽受限管道或大规模图像处理任务中的性能。

我们对 Halide 进行了扩展，以支持复杂模板管道的分布式内存并行执行。 这些扩展与 Halide 中现有的调度结构相结合，允许表达复杂的计算和通信策略。 现有的 Halide 应用程序可以通过最小的更改进行分发，允许程序员轻松探索重新计算和通信之间的权衡。 对于 200 行、99 阶段的应用程序，只需要大约 10 行新代码。 

在九个图像处理基准测试中，我们的扩展通过减轻非均匀内存访问的影响，在单个节点上比具有相同数量内核的常规多线程执行速度提高了 1.4 倍。 

## Breaking the Computation and Communication Abstraction Barrier in Distributed Machine Learning Workloads(ASPLOS 2022)

分布式训练模型必须解锁计算和通信方面的优化以获得最佳性能。然而，手动应用这些优化需要为每个场景修改底层计算和通信库，这既耗时又容易出错。

提出了 CoCoNet，它包含 (i) 一种领域特定语言，以计算和通信操作的形式表达分布式机器学习程序，(ii) 一组语义保留转换来优化程序，以及 (iii) 一个 编译器生成联合优化的通信和计算 GPU 内核。 

将计算和通信作为first class构造提供，允许用户进行高级抽象并应用强大的优化，例如通信和计算的融合或重叠。 CoCoNet 使我们能够仅用几行代码优化大型语言模型中的数据、模型和管道并行工作负载。 

## DISTAL: The Distributed Tensor Algebra Compiler(pldi 22)

我们介绍了 DISTAL，一种针对现代分布式和异构系统的密集张量代数编译器。**DISTAL 允许用户独立描述张量和计算如何通过单独的格式和调度语言映射到目标机器上**。

数据和计算分布选择的组合创造了一个很大的设计空间，其中包括许多过去的算法（例如 Cannon 算法）和现在的算法（例如 COSMA）。 DISTAL 将张量代数领域特定语言编译为**基于分布式任务**的运行时系统，并支持具有多核 CPU 和多个 GPU 的节点。 

DISTAL 生成的代码可与 Lassen 超级计算机 256 个节点上的矩阵乘法优化代码相媲美，并且在高阶张量运算上比现有系统高出 1.8 到 3.7 倍（异常值为 45.7 倍）。

## Alpa: Automating Inter- and Intra-Operator Parallelism for Distributed Deep Learning(OSDI 2022)

Alpa 通过生成统一的数据、运算符和管道并行的执行计划，自动对大型深度学习 (DL) 模型进行model-parallel训练。 现有的model-parallel训练系统要么需要用户手动创建并行化计划，要么从有限的模型并行配置空间自动生成一个。 它们不足以在分布式计算设备上扩展复杂的 DL 模型。 

Alpa 通过将并行性视为inter-operator and intra-operator parallelisms两个层次级别来分发大型 DL 模型的训练。 在此基础上，Alpa 为海量模型并行执行计划构建了一个新的层次空间。 

Alpa 设计了许多编译过程，以在每个并行级别自动派生出高效的并行执行计划。 Alpa 实现了一个高效的运行时来协调分布式计算设备上的两级并行执行。 

我们的评估表明，Alpa 生成的并行化计划即使在为它们设计的模型上也能匹配或优于手动调整的模型并行训练系统。 与专用系统不同，Alpa 还可以泛化到具有异构架构的模型和无需手动设计计划的模型。