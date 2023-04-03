
# 编译器场景下的资源调度问题

## VELTAIR: Towards High-Performance Multi-tenant Deep Learning Services via Adaptive Compilation and Scheduling (ASPLOS 22)

- 粒度调度算法。提升资源利用率，降低调度conflict
  - 按model进行调度cpu利用率太低，按layer进行调度conflict太多。因此提出按block apdaptive进行调度
- 自适应编译策略。可以动态智能地选择独占或共享资源的程序，以减少整体干扰引起的性能损失。
  - 分析了一组深度学习层的易受干扰版本和干扰容忍代码版本之间的关系。 这些不同的版本本质上是并行性和局部性之间权衡的帕累托均衡。因此，提出了一种基于现有自动调度程序的single pass compiling strategy。 扩展的自动调度器会编译出用于不同干扰水平的不同版本。
- 场景：推理的计算量很小，所以通常一个服务器上会部署多个推理任务
- tvm之类的编译器只考虑单租户情况，在多租户情况下，性能下降很快