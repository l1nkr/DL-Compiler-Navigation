import tvm
from tvm import te
n = 1024
dtype = "float32"
A = te.placeholder((n, n), dtype=dtype, name='A')
k = te.reduce_axis((0, n), name='k')
B = te.compute((n,), lambda i: te.sum(A[i, k], axis=k), name='B')
C = te.compute((n,), lambda i: B[i] + 10, name='C')

s = te.create_schedule(C.op)

print(tvm.lower(s, [A, C], simple_mode=True))
print("---------cutting line---------")

s[B].set_scope('shared')

print(tvm.lower(s, [A, C], simple_mode=True))
