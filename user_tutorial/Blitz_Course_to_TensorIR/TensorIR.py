# https://tvm.apache.org/docs/tutorial/tensor_ir_blitz_course.html

# Based on the design of TensorIR and IRModule, we are able to create a new programming method:

# Write a program by TVMScript in a python-AST based syntax.
# Transform and optimize a program with python api.
# Interactively inspect and try the performance with an imperative style transformation API.


import tvm
from tvm.ir.module import IRModule
from tvm.script import tir as T
import numpy as np

@tvm.script.ir_module
class MyModule:
    @T.prim_func
    def main(a: T.handle, b: T.handle):
        # We exchange data between function by handles, which are similar to pointer.
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        # Create buffer from handles.
        A = T.match_buffer(a, (8,), dtype="float32")
        B = T.match_buffer(b, (8,), dtype="float32")
        for i in range(8):
            # A block is an abstraction for computation.
            with T.block("B"):
                # Define a spatial block iterator and bind it to value i.
                vi = T.axis.spatial(8, i)
                B[vi] = A[vi] + 1.0

def same_pre_in_te():
    from tvm import te

    A = te.placeholder((8,), dtype="float32", name="A")
    B = te.compute((8,), lambda *i: A(*i) + 1.0, name="B")
    func = te.create_prim_func([A, B])
    ir_module_from_te = IRModule({"main": func})
    print(ir_module_from_te.script())

ir_module = MyModule
print(type(ir_module))
print(ir_module.script())
mod = tvm.build(ir_module, target="llvm")  # The module for CPU backends.
print(type(mod))

a = tvm.nd.array(np.arange(8).astype("float32"))
b = tvm.nd.array(np.zeros((8,)).astype("float32"))
mod(a, b)
print(a)
print(b)

# schedule

sch = tvm.tir.Schedule(ir_module)
print(type(sch))
# Get block by its name
block_b = sch.get_block("B")
# Get loops surrounding the block
(i,) = sch.get_loops(block_b)
# Tile the loop nesting.
i_0, i_1, i_2 = sch.split(i, factors=[2, 2, 2])
print(sch.mod.script())

sch.reorder(i_0, i_2, i_1)
print(sch.mod.script())

# Transform to a GPU program

sch.bind(i_0, "blockIdx.x")
sch.bind(i_2, "threadIdx.x")
print(sch.mod.script())

ctx = tvm.cuda(0)
cuda_mod = tvm.build(sch.mod, target="cuda")
cuda_a = tvm.nd.array(np.arange(8).astype("float32"), ctx)
cuda_b = tvm.nd.array(np.zeros((8,)).astype("float32"), ctx)
cuda_mod(cuda_a, cuda_b)
print(cuda_a)
print(cuda_b)