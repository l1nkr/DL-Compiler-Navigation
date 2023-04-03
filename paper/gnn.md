# Graph Neural Network

由于GNN计算过程中的稀疏性，使得计算访存比的提升成为了优化的主要目标，而ML编译优化中的算子融合成为了典型的优化方案。
这方面的工作不少，比如DGL团队的featgraph、Graphiler、以及港中文James Chen团队的Seastar，编译优化后既能省内存，又能加速计算。从我在华为的工作经验来看，效果还是相当不错的。只不过现在的GNN训练中模型计算时间消耗本身占比整个训练pipeline的时间比较低，训练的性能瓶颈核心还是在图引擎上。

https://zhuanlan.zhihu.com/p/423889406

## Graphiler: Optimizing Graph Neural Networks with Message Passing Data Flow Graph (MLSys22)

图神经网络 (GNN) 是一类新的强大的机器学习模型，但简单的编程和高效的计算往往是矛盾的。 当前的 GNN 框架基于消息传递范式，并允许使用内置原语和用户定义函数 (UDF) 简洁地表达 GNN 模型。 虽然内置原语提供了高性能，但它们的表现力有限； UDF 很灵活，但通常性能低下并且使用过多的内存。 在本文中，我们提出了 Graphiler，这是一种用于 GNN 的编译器堆栈，它在提供 UDF 编程接口的灵活性的同时实现了高性能。 Graphiler 的核心是一种称为消息传递数据流图 (MP-DFG) 的新颖抽象，它可以实现优化，从而大大减少计算冗余和内存占用，并在统一框架下优化同构和异构 GNN。 实验表明，Graphiler 可以将 UDF GNN 加速多达两个数量级，并实现接近或优于专家实现的性能，并且可以节省大量内存。

https://zhuanlan.zhihu.com/p/570329623

## Seastar: vertex-centric programming for graph neural networks (Eurosys21)

图神经网络 (GNN) 在节点分类、链接预测和图聚类等图分析方面取得了突破性的性能。 已经开发了许多 GNN 训练框架，但它们通常被设计为一组手动编写的、GNN 特定的运算符插入现有的深度学习系统，这导致内存消耗高、数据局部性差以及算法设计和实现之间的语义差距大 . 本文提出了 Seastar 系统，它为 GPU 上的 GNN 训练提供了一个以顶点为中心的编程模型，并提供了惯用的 Python 结构，以便轻松开发新颖的同构和异构 GNN 模型。 我们还提出了新颖的优化方法，以产生高效的融合 GPU 内核，用于 GNN 训练中的前向和后向传递。 与最先进的 GNN 系统 DGL 和 PyG 相比，Seastar 实现了更好的可用性，内存消耗分别减少了 2 倍和 8 倍，执行速度分别提高了 14 倍和 3 倍。

## FeatGraph: A Flexible and Efficient Backend for Graph Neural Network Systems (SC20)

图神经网络 (GNN) 作为一种很有前途的图机器学习方法越来越受欢迎。 与每个顶点/边都与标量相关联的传统图形工作负载不同，GNN 将特征张量附加到每个顶点/边。 这种额外的特征维度，以及随之而来的更复杂的顶点和边缘计算，对局部性和并行性具有巨大影响，而现有的图形处理系统无法利用这些影响。

本文提出 FeatGraph 通过共同优化图遍历和特征维度计算来加速 GNN 工作负载。 FeatGraph 通过在每个顶点/边上将粗粒度的稀疏模板与细粒度的用户定义函数 (UDF) 组合在一起，提供了一个灵活的编程接口来表达不同的 GNN 模型。 FeatGraph 将图形遍历的优化合并到稀疏模板中，并允许用户使用特征维度表 (FDS) 指定 UDF 的优化。 FeatGraph 在 CPU 上将端到端 GNN 训练和推理速度提高了 32 倍，在 GPU 上提高了 7 倍。

https://leiblog.wang/%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB%EF%BD%9CFeatGraph/