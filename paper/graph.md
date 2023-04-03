# Graph-level Optimization

## Nimble: Lightweight and Parallel GPU Task Scheduling for Deep Learning (NeurIPS 2020)

现有的 DL 框架存在大的**调度开销**和不必要的**串行执行**的问题，导致效率低下

Nimble 引入了一种称为提前 (AoT) 调度的新技术。调度在执行 GPU 内核之前完成，从而消除了运行时的大部分调度开销。

核心思想就是充分利用gpu的并行能力，通过充分挖掘算子间的并行潜力实现。

