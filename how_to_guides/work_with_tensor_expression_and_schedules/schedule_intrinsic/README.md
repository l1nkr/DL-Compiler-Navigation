### cache_read

cache_read将tensor读入指定存储层次scope的cache，这个设计的意义在于显式利用现有计算设备的on-chip memory hierarchy。这个例子中，会先将A的数据load到shared memory中，然后计算B。在这里，我们需要引入一个stage的概念，一个op对应一个stage，也就是通过cache_read会新增一个stage。

```python
produce B {
  for (i, 0, 1024)
    B[i] = 0f
    for (k, 0, 1024)
      B[i] = (B[i] + A[((i*1024) + k)])

---------cutting line---------

// attr [A.shared] storage_scope = "shared"
allocate A.shared[float32 * 1048576]

produce A.shared
  for (ax0, 0, 1024)
    for (ax1, 0, 1024)
      A.shared[((ax0*1024) + ax1)] = A[((ax0*1024) + ax1)]
produce B
  for (i, 0, 1024)
    B[i] = 0f
    for (k, 0, 1024)
      B[i] = (B[i] + A.shared[((i*1024) + k)])
```

### cache_write

cache_write和cache_read对应，是先在shared memory中存放计算结果，最后将结果写回到global memory。当然在真实的场景中，我们往往是会将结果先放着register中，最后写回。

```python
produce B
  for (i, 0, 1024)
    B[i] = 0f
    for (k, 0, 1024)
      B[i] = (B[i] + A[((i*1024) + k)])

---------cutting line---------

// attr [B.local] storage_scope = "local"
allocate B.local[float32 * 1024]
produce B.local
  for (i.c, 0, 1024)
    B.local[i.c] = 0f
    for (k, 0, 1024)
      B.local[i.c] = (B.local[i.c] + A[((i.c*1024) + k)])
produce B
  for (i, 0, 1024)
    B[i] = B.local[i]
```

### set_scope

set_scope指定stage计算结果所在的存储层次，为tensor选择最优的存储位置，适用于设置线程间的共享内存。事实上，set_scope是cache_read的子操作。

```python
// attr [B] storage_scope = "global"
allocate B[float32 * 1024]
produce B
  for (i, 0, 1024)
    B[i] = 0f
    for (k, 0, 1024)
      B[i] = (B[i] + A[((i*1024) + k)])
produce C
  for (i, 0, 1024)
    C[i] = (B[i] + 10f)

---------cutting line---------

// attr [B] storage_scope = "shared"
allocate B[float32 * 1024]
produce B
  for (i, 0, 1024)
    B[i] = 0f
    for (k, 0, 1024)
      B[i] = (B[i] + A[((i*1024) + k)])
produce C
  for (i, 0, 1024)
    C[i] = (B[i] + 10f)
```

### storage_align

storage_align把stage对应的存储空间以factor为单位、以offset为偏置重新对齐，以避免GPU共享访问时的bank conflict。

```python
// attr [A.shared] storage_scope = "shared"
allocate A.shared[float32 * 1048576]
produce A.shared
  for (ax0, 0, 1024)
    for (ax1, 0, 1024)
      A.shared[((ax0*1024) + ax1)] = A[((ax0*1024) + ax1)]
produce B
  for (i, 0, 1024)
    B[i] = 0f
    for (k, 0, 1024)
      B[i] = (B[i] + A.shared[((i*1024) + k)])

---------cutting line---------

// attr [A.shared] storage_scope = "shared"
allocate A.shared[float32 * 1134592]
produce A.shared
  for (ax0, 0, 1024)
    for (ax1, 0, 1024)
      A.shared[((ax0*1108) + ax1)] = A[((ax0*1024) + ax1)]
produce B
  for (i, 0, 1024)
    B[i] = 0f
    for (k, 0, 1024)
      B[i] = (B[i] + A.shared[((i*1108) + k)])
```

### fuse

fuse用于融合两个iter，将两层循环合并到一层，其返回值为iter类型，可以多次合并。

```python
produce B
  B[0] = 0f
  for (k.outer, 0, 32)
    for (k.inner, 0, 32)
      B[0] = (B[0] + A[((k.outer*32) + k.inner)])

---------cutting line---------
produce B
  B[0] = 0f
  for (k.outer.k.inner.fused, 0, 1024)
    B[0] = (B[0] + A[k.outer.k.inner.fused])
```

### split

split是fuse的反操作，把iter以factor为间隔分离成outer与inner两层迭代，增加循环层数，用于将循环操作分割为更小的子任务。事实上，以CUDA为例，gridDim和blockDim都可以最多是三维，所以通过split可以产生新的维度用于绑定到grid和block上

```python
produce B
  B[0] = 0f
  for (k, 0, 1024)
    B[0] = (B[0] + A[k])

---------cutting line---------
produce B
  B[0] = 0f
  for (k.outer, 0, 128)
    for (k.inner, 0, 8)
      B[0] = (B[0] + A[((k.outer*8) + k.inner)])
```

### reorder

reorder用于重置循环iter的内外顺序，根据局部性原理，最大化利用cache中的现有数据，减少反复载入载出的情况。注意，这里到底怎样的顺序是最优化的是一个很有趣的问题。以矩阵乘法为例，M, N, K三维，往往是将K放在最外层可以最大程度利用局部性。

```python
produce C
  for (i.outer, 0, 64)
    for (i.inner, 0, 16)
      for (j.outer, 0, 64)
        for (j.inner, 0, 16)
          let cse_var_1: int32 = ((((i.outer*16384) + (i.inner*1024)) + (j.outer*16)) + j.inner)
          C[cse_var_1] = (A[cse_var_1] + B[cse_var_1])

---------cutting line---------
produce C
  for (i.outer, 0, 64)
    for (j.outer, 0, 64)
      for (j.inner, 0, 16)
        for (i.inner, 0, 16)
          let cse_var_1: int32 = ((((i.outer*16384) + (i.inner*1024)) + (j.outer*16)) + j.inner)
          C[cse_var_1] = (A[cse_var_1] + B[cse_var_1])
```

### tile

**tile是可以由split和reorder来实现的**
tile将stage的两个维度按照各自的factor拆分，并以固定顺序依次返回两个outer和两个inner的iter，从而增加循环层数，形成更小的计算任务，tile是矩阵乘法和卷积计算的重要schedule。

```python
produce C
  for (i, 0, 1024)
    for (j, 0, 1024)
      C[((i*1024) + j)] = 0f
      for (K, 0, 1024)
        C[((i*1024) + j)] = (C[((i*1024) + j)] + (A[((i*1024) + K)]*B[((K*1024) + j)]))

---------cutting line---------
produce C {
  for (i.outer, 0, 64) {
    for (j.outer, 0, 128)
      for (i.inner, 0, 16)
        for (j.inner, 0, 8)
          C[((((i.outer*16384) + (i.inner*1024)) + (j.outer*8)) + j.inner)] = 0f32
          for (K: int32, 0, 1024) {
            let cse_var_3: int32 = (j.outer*8)
            let cse_var_2: int32 = ((i.outer*16384) + (i.inner*1024))
            let cse_var_1: int32 = ((cse_var_2 + cse_var_3) + j.inner)
            C[cse_var_1] = (C[cse_var_1] + (A[(cse_var_2 + K)]*B[(((K*1024) + cse_var_3) + j.inner)]))
```

### rfactor(tensor, axis, factor_axis=0)

rfactor对原tensor在axis方向以factor_axis为间隔做reduction操作。

```python
produce B
  B[0] = 0f
  for (k.outer, 0, 64)
    for (k.inner, 0, 16)
      B[0] = (B[0] + A[((k.outer*16) + k.inner)])

---------cutting line---------
// attr [B.rf] storage_scope = "global"
allocate B.rf[float32 * 16]
produce B.rf
  for (k.inner, 0, 16)
    B.rf[k.inner] = 0f
    for (k.outer, 0, 64)
      B.rf[k.inner] = (B.rf[k.inner] + A[((k.outer*16) + k.inner)])
produce B
  B[0] = 0f
  for (k.inner.v, 0, 16)
    B[0] = (B[0] + B.rf[k.inner.v])
```

### unroll

unroll是一种常见的循环优化方法，减分支预测失败减少，如果循环体内语句没有数据相关，增加了并发执行的机会，也有利于指令流水线的调度。

```python
produce C
  for (i.outer, 0, 256)
    for (i.inner, 0, 4)
      for (j, 0, 1024)
        C[(((i.outer*4096) + (i.inner*1024)) + j)] = (A[(((i.outer*4096) + (i.inner*1024)) + j)] + B[(((i.outer*4096) + (i.inner*1024)) + j)])

---------cutting line---------
produce C
  for (i.outer, 0, 256)
    for (j, 0, 1024)
      C[((i.outer*4096) + j)] = (A[((i.outer*4096) + j)] + B[((i.outer*4096) + j)])
    for (j, 0, 1024)
      C[(((i.outer*4096) + j) + 1024)] = (A[(((i.outer*4096) + j) + 1024)] + B[(((i.outer*4096) + j) + 1024)])
    for (j, 0, 1024)
      C[(((i.outer*4096) + j) + 2048)] = (A[(((i.outer*4096) + j) + 2048)] + B[(((i.outer*4096) + j) + 2048)])
    for (j, 0, 1024)
      C[(((i.outer*4096) + j) + 3072)] = (A[(((i.outer*4096) + j) + 3072)] + B[(((i.outer*4096) + j) + 3072)])
```

### vectorize

vectorize把iter方向上的循环迭代替换成ramp，从而通过SIMD指令实现数据的批量计算，并且只有在数据size为常数、且分割的iter为2的幂（即满足SIMD的计算数量）时才会发生替换，否则vectorize没有效果，是SIMD计算设备的常用schedule。

```python
produce C
  for (x.outer, 0, 32)
    for (y.outer, 0, 32)
      for (x.inner, 0, 32)
        for (y.inner, 0, 32)
          C[((((x.outer*32768) + (x.inner*1024)) + (y.outer*32)) + y.inner)] = (A[((((x.outer*32768) + (x.inner*1024)) + (y.outer*32)) + y.inner)] + B[((((x.outer*32768) + (x.inner*1024)) + (y.outer*32)) + y.inner)])

---------cutting line---------
produce C
  for (x.outer, 0, 32)
    for (y.outer, 0, 32)
      for (x.inner, 0, 32)
        C[ramp((((x.outer*32768) + (x.inner*1024)) + (y.outer*32)), 1, 32)] = (A[ramp((((x.outer*32768) + (x.inner*1024)) + (y.outer*32)), 1, 32)] + B[ramp((((x.outer*32768) + (x.inner*1024)) + (y.outer*32)), 1, 32)])
```

### parallel

parallel将指定iter的for循环替换为parallel操作，从而在GPU以外的CPU等设备上实现并行。

```python
produce B
  for (i, 0, 1024)
    B[i] = 0f
    for (l, 0, 1024)
      B[i] = (B[i] + A[((i*1024) + l)])

---------cutting line---------
produce B
  for (i, 0, 1024)
    B[i] = 0f
    parallel (l, 0, 1024)
      B[i] = (B[i] + A[((i*1024) + l)])
```

### tensorize

tensorize将计算作为整体，编译为一个tensor_intrin函数中。这是因为很多计算属于常用计算，针对这些计算已经有了很好的built-in的schedule，通过tensorize可以直接调用这些内置的intrinsic，其实这也就是intrinsic在计算机科学中的本意

[code](./src/tensorize.py)

### pragma

pragma用于添加编译注释，使编译器遵循pragma的要求，实现unroll, vectorize等调度功能。事实上一个新的优化规则，都可以看做是一种gragma，也被称作directive

```python
s[B].pragma(ki, "unroll")
```

### prefetch

prefetch利用数据的空间局部性，用于使得前面的计算与后面访存overlap起来，提高访存和计算的并行度，减少耗时。本质上是软件流水线的概念，不是硬件prefetch。

```python
produce B
  for (i, 0, 1024)
    B[i] = 0f
    for (k, 0, 1024)
      B[i] = (B[i] + A[((i*1024) + k)])

---------cutting line---------
produce B
  for (i, 0, 1024)
    B[i] = 0f
    for (k, 0, 1024)
      for (prefetch.A.1, 0, 1)
        for (prefetch.A.0, 0, 1)
          prefetch(tvm_address_of(A[(((k*1024) + i) + 1024)]), 0, 3, 1)
      B[i] = (B[i] + A[((i*1024) + k)])
```

### set_store_predicate

set_store_predicate设置了store的条件，适用于在多线程调度中预防写操作之间的冲突。

```python
// attr [B.rf] storage_scope = "global"
allocate B.rf[float32 * 1]
produce B
  B[0] = 0f
  for (k.inner.v, 0, 16)
    produce B.rf
      B.rf[0] = 0f
      for (k.outer, 0, 64)
        B.rf[0] = (B.rf[0] + A[((k.outer*16) + k.inner.v)])
    B[0] = (B[0] + B.rf[0])

---------cutting line---------
// attr [B.rf] storage_scope = "global"
allocate B.rf[float32 * 1]
produce B
  B[0] = 0f
  for (k.inner.v, 0, 16)
    produce B.rf
      B.rf[0] = 0f
      for (k.outer, 0, 64)
        B.rf[0] = (B.rf[0] + A[((k.outer*16) + k.inner.v)])
    if ((threadIdx.x == 0))
      B[0] = (B[0] + B.rf[0])
```

### create_group(outputs, inputs, include_inputs=False)

create_group对从inputs到outputs的所有stage创建group，group本质上是一个虚拟stage，可以通过操作这个虚拟stage来一起操作这个group里的所有stage。本例中，通过compute_at使这个group中的D和E，一起附着到指定操作中。

```python
// attr [D] storage_scope = "global"
allocate D[float32 * 1048576]
// attr [F] storage_scope = "global"
allocate F[float32 * 1024]
produce D
  for (i, 0, 1024)
    for (j, 0, 1024)
      D[((i*1024) + j)] = (A[((i*1024) + j)] + B[((i*1024) + j)])
produce E
  for (i, 0, 1024)
    for (j, 0, 1024)
      E[((i*1024) + j)] = (D[((i*1024) + j)] + B[((i*1024) + j)])
produce F
  for (i, 0, 1024)
    F[i] = 0f
    for (k, 0, 1024)
      F[i] = (F[i] + E[((i*1024) + k)])

---------cutting line---------
// attr [F] storage_scope = "global"
allocate F[float32 * 1024]
// attr [D] storage_scope = "global"
allocate D[float32 * 1]
produce F
  for (i, 0, 1024)
    F[i] = 0f
    for (k, 0, 1024)
      produce D
        D[0] = (A[((i*1024) + k)] + B[((i*1024) + k)])
      produce E
        E[((i*1024) + k)] = (D[0] + B[((i*1024) + k)])
      F[i] = (F[i] + E[((i*1024) + k)])
```

### bind

bind将iter绑定到block或thread的index上，从而把循环的任务分配到线程，实现并行化计算，这是针对CUDA后端最核心的部分。

```python
produce B
  B[0] = 0f
  for (k.outer, 0, 64)
    for (k.inner, 0, 16)
      B[0] = (B[0] + A[((k.outer*16) + k.inner)])

---------cutting line---------
produce B  
  // attr [IterVar(blockIdx.x: int32, (nullptr), "ThreadIndex", "blockIdx.x")] "thread_extent" = 64;
  allocate(reduce_temp0: Pointer(local float32), float32, [1]), storage_scope = local;
  // attr [IterVar(threadIdx.x: int32, (nullptr), "ThreadIndex", "threadIdx.x")] "thread_extent" = 16
  // attr [meta[tir.CommReducer][0]] "reduce_scope" = @tir.reinterpret(0u64, dtype=handle);
  @tir.tvm_thread_allreduce(1u32, A[((blockIdx.x*16) + threadIdx.x)], True, reduce_temp0_1: Buffer(reduce_temp0, float32, [1], [], scope="local")[0], blockIdx.x, threadIdx.x, dtype=handle)
  B[0] = reduce_temp0_1[0]
```

### compute_at

compute_at将当前的stage附着到目标stage的指定iter方向上，同时与目标stage采用相同的并行方式，在其内部完成当前stage的计算。往往compute_at会与cache_read和cache_write一起使用。

```python
for (k.inner: int32, 0, 16)
    B.rf_1: Buffer(B.rf, float32, [16], [])[k.inner] = 0f32
    for (k.outer: int32, 0, 64)
    B.rf_1[k.inner] = (B.rf_1[k.inner] + A[((k.outer*16) + k.inner)])
B[0] = 0f32
for (k.inner.v: int32, 0, 16) 
    B[0] = (B[0] + B.rf_1[k.inner.v])

---------cutting line---------
B[0] = 0f32
for (k.inner.v: int32, 0, 16)
    B.rf_1: Buffer(B.rf, float32, [1], [], align=4)[0] = 0f32
    for (k.outer: int32, 0, 64)
    B.rf_1[0] = (B.rf_1[0] + A[((k.outer*16) + k.inner.v)])
    B[0] = (B[0] + B.rf_1[0])
```

### compute_inline

compute_inline把独立的计算操作转化成内联函数形式，在使用到原计算结果时再调用内联函数完成运算，通过compute_inline来减少一个stage。

```python
// attr [Apad] storage_scope = "global"
allocate Apad[float32 * 1056784]
produce Apad
  for (yy, 0, 1028)
    for (xx, 0, 1028)
      Apad[((yy*1028) + xx)] = tvm_if_then_else(((((2 <= yy) && (yy < 1026)) && (2 <= xx)) && (xx < 1026)), A[(((yy*1024) + xx) - 2050)], 0f)
produce B
  for (yy, 0, 1026)
    for (xx, 0, 1026)
      B[((yy*1026) + xx)] = 0f
      for (ry, 0, 3)
        for (rx, 0, 3)
          B[((yy*1026) + xx)] = (B[((yy*1026) + xx)] + (Apad[((((yy*1028) + (ry*1028)) + xx) + rx)]*W[((ry*3) + rx)]))

---------cutting line---------
produce B
  for (yy, 0, 1026)
    for (xx, 0, 1026)
      B[((yy*1026) + xx)] = 0f
      for (ry, 0, 3)
        for (rx, 0, 3)
          B[((yy*1026) + xx)] = (B[((yy*1026) + xx)] + (tvm_if_then_else(((((2 <= (yy + ry)) && ((yy + ry) < 1026)) && (2 <= (xx + rx))) && ((xx + rx) < 1026)), A[(((((yy*1024) + (ry*1024)) + xx) + rx) - 2050)], 0f)*W[((ry*3) + rx)]))
```

### compute_root

compute_root是compute_at的反操作。因为不做任何schedule的话，每一个stage默认就是compute_root的，这个schedule相当于注释了对之前对一个stage的compute操作。

```python
produce B {
  // attr [iter_var(threadIdx.x, , threadIdx.x)] thread_extent = 32
  // attr [B.rf] storage_scope = "local"
  allocate B.rf[float32 * 1]
  // attr [reduce_temp0] storage_scope = "local"
  allocate reduce_temp0[float32 * 1]
  produce B.rf {
    B.rf[0] = 0f
    for (k.outer, 0, 32) {
      B.rf[0] = (B.rf[0] + A[((k.outer*32) + threadIdx.x)])
    }
  }
  // attr [comm_reducer(result=[(x + y)], lhs=[x], rhs=[y], identity_element=[0f])] reduce_scope = reinterpret((uint64)0)
  tvm_thread_allreduce((uint32)1, B.rf[0], (bool)1, reduce_temp0, threadIdx.x)
  B[0] = reduce_temp0[0]
}

---------cutting line---------
// attr [B.rf] storage_scope = "global"
allocate B.rf[float32 * 32]
produce B.rf {
  for (k.inner, 0, 32) {
    B.rf[k.inner] = 0f
    for (k.outer, 0, 32) {
      B.rf[k.inner] = (B.rf[k.inner] + A[((k.outer*32) + k.inner)])
    }
  }
}
produce B {
  // attr [iter_var(threadIdx.x, , threadIdx.x)] thread_extent = 32
  // attr [reduce_temp0] storage_scope = "local"
  allocate reduce_temp0[float32 * 1]
  // attr [comm_reducer(result=[(x + y)], lhs=[x], rhs=[y], identity_element=[0f])] reduce_scope = reinterpret((uint64)0)
  tvm_thread_allreduce((uint32)1, B.rf[threadIdx.x], (bool)1, reduce_temp0, threadIdx.x)
  B[0] = reduce_temp0[0]
}
```

参考：
- https://zhuanlan.zhihu.com/p/94846767
- https://github.com/StrongSpoon/tvm.schedule
