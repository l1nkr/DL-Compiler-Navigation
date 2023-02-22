# split: splits a specified axis into two axises by the defined factor.
# tile: tiles will split a computation across two axes by the defined factors.
# fuse: fuses two consecutive axises of one computation.
# reorder: can reorder the axises of a computation into a defined order.
# bind: can bind a computation to a specific thread, useful in GPU programming.
# compute_at: by default, TVM will compute tensors at the outermost level of the function, or the root, by default. 
#             compute_at specifies that one tensor should be computed at the first axis of computation for another operator.
# compute_inline: when marked inline, a computation will be expanded then inserted into the address where the tensor is required.
# compute_root: moves a computation to the outermost layer, or root, of the function. 
#               This means that stage of the computation will be fully computed before it moves on to the next stage.

# https://tvm.apache.org/docs/tutorial/tensor_expr_get_started.html

# Preparation and Performance Baseline

import tvm
import tvm.testing
from tvm import te
import numpy
import timeit
# The size of the matrix
# (M, K) x (K, N)
# You are free to try out different shapes, sometimes TVM optimization outperforms numpy with MKL.
M = 1024
K = 1024
N = 1024

# The default tensor data type in tvm
dtype = "float32"

target = tvm.target.Target(target="llvm", host="llvm")
dev = tvm.device(target.kind.name, 0)

# Random generated tensor for testing
a = tvm.nd.array(numpy.random.rand(M, K).astype(dtype), dev)
b = tvm.nd.array(numpy.random.rand(K, N).astype(dtype), dev)
c = tvm.nd.array(numpy.zeros((M, N), dtype=dtype), dev)
# Repeatedly perform a matrix multiplication to get a performance baseline
# for the default numpy implementation
np_repeat = 100
np_running_time = timeit.timeit(
    setup="import numpy\n"
    "M = " + str(M) + "\n"
    "K = " + str(K) + "\n"
    "N = " + str(N) + "\n"
    'dtype = "float32"\n'
    "a = numpy.random.rand(M, K).astype(dtype)\n"
    "b = numpy.random.rand(K, N).astype(dtype)\n",
    stmt="answer = numpy.dot(a, b)",
    number=np_repeat,
)
print("Numpy running time: %f" % (np_running_time / np_repeat))

answer = numpy.dot(a.numpy(), b.numpy())


def evaluate_operation(s, vars, target, name, optimization, log):
    func = tvm.build(s, vars, target=target, name="mmult")
    assert func

    c = tvm.nd.array(numpy.zeros((M, N), dtype=dtype), dev)
    func(a, b, c)
    tvm.testing.assert_allclose(c.numpy(), answer, rtol=1e-5)

    evaluator = func.time_evaluator(func.entry_name, dev, number=10)
    mean_time = evaluator(a, b, c).mean
    print("%s: %f" % (optimization, mean_time))
    log.append((optimization, mean_time))
    
# TVM Matrix Multiplication using TE
def use_TE():
    config = {
        "block": True,
        "vectorize": True,
        "permutation": True,
    }
    k = te.reduce_axis((0, K), "k")
    A = te.placeholder((M, K), name="A")
    B = te.placeholder((K, N), name="B")
    C = te.compute((M, N), lambda x, y: te.sum(A[x, k] * B[k, y], axis=k), name="C")

    # Default schedule
    s = te.create_schedule(C.op)
    
    if config["block"]:
        bn = 32
        # Blocking by loop tiling
        xo, yo, xi, yi = s[C].tile(C.op.axis[0], C.op.axis[1], bn, bn)
        (k,) = s[C].op.reduce_axis
        ko, ki = s[C].split(k, factor=4)

        # Hoist reduction domain outside the blocking loop
        s[C].reorder(xo, yo, ko, ki, xi, yi)
        
        if config["permutation"]:
            s[C].reorder(xo, yo, ko, xi, ki, yi)
        if config["vectorize"]:
            s[C].vectorize(yi)

    func = tvm.build(s, [A, B, C], target=target, name="mmult")
    func(a, b, c)
    tvm.testing.assert_allclose(c.numpy(), answer, rtol=1e-5)

    log = []

    evaluate_operation(s, vars=[A, B, C], target=target, name="mmult", optimization="none", log=log)
    print(tvm.lower(s, [A, B, C], simple_mode=True))
    
def arrayPacking():
    k = te.reduce_axis((0, K), "k")
    A = te.placeholder((M, K), name="A")
    B = te.placeholder((K, N), name="B")
    bn = 32
    # We have to re-write the algorithm slightly.
    packedB = te.compute((N / bn, K, bn), lambda x, y, z: B[y, x * bn + z], name="packedB")
    C = te.compute(
        (M, N),
        lambda x, y: te.sum(A[x, k] * packedB[y // bn, k, tvm.tir.indexmod(y, bn)], axis=k),
        name="C",
    )

    s = te.create_schedule(C.op)

    xo, yo, xi, yi = s[C].tile(C.op.axis[0], C.op.axis[1], bn, bn)
    (k,) = s[C].op.reduce_axis
    ko, ki = s[C].split(k, factor=4)

    s[C].reorder(xo, yo, ko, xi, ki, yi)
    s[C].vectorize(yi)

    x, y, z = s[packedB].op.axis
    s[packedB].vectorize(z)
    s[packedB].parallel(x)
    log = []

    evaluate_operation(s, [A, B, C], target=target, name="mmult", optimization="array packing", log=log)

    print(tvm.lower(s, [A, B, C], simple_mode=True))
    
def cache():
    k = te.reduce_axis((0, K), "k")
    A = te.placeholder((M, K), name="A")
    B = te.placeholder((K, N), name="B")
    bn = 32
    # We have to re-write the algorithm slightly.
    packedB = te.compute((N / bn, K, bn), lambda x, y, z: B[y, x * bn + z], name="packedB")
    C = te.compute(
        (M, N),
        lambda x, y: te.sum(A[x, k] * packedB[y // bn, k, tvm.tir.indexmod(y, bn)], axis=k),
        name="C",
    )
    s = te.create_schedule(C.op)
    
    # Allocate write cache
    CC = s.cache_write(C, "global")

    xo, yo, xi, yi = s[C].tile(C.op.axis[0], C.op.axis[1], bn, bn)

    # Write cache is computed at yo
    s[CC].compute_at(s[C], yo)

    # New inner axes
    xc, yc = s[CC].op.axis

    (k,) = s[CC].op.reduce_axis
    ko, ki = s[CC].split(k, factor=4)
    s[CC].reorder(ko, xc, ki, yc)
    s[CC].unroll(ki)
    s[CC].vectorize(yc)

    x, y, z = s[packedB].op.axis
    s[packedB].vectorize(z)
    s[packedB].parallel(x)
    log = []
    evaluate_operation(s, [A, B, C], target=target, name="mmult", optimization="block caching", log=log)

    # Here is the generated IR after write cache blocking.
    print(tvm.lower(s, [A, B, C], simple_mode=True))
    
    
use_TE()
arrayPacking()
cache()