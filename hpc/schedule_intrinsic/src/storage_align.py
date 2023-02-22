import tvm
from tvm import te
n = 1024
factor =100
offset =8
dtype = "float32"
A = te.placeholder((n, n), dtype=dtype, name='A')
k = te.reduce_axis((0, n), name='k')
B = te.compute((n,), lambda i: te.sum(A[i, k], axis=k), name='B')

s = te.create_schedule(B.op)
AA = s.cache_read(A, "shared", [B])

print(tvm.lower(s, [A, B], simple_mode=True))
print("---------cutting line---------")

s[AA].storage_align(AA.op.axis[0], factor, offset)

print(tvm.lower(s, [A, B], simple_mode=True))