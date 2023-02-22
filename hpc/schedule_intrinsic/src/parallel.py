import tvm
from tvm import te

n = 1024
m = 1024

A = te.placeholder((n, m), name='A')
l = te.reduce_axis((0, m), name = 'l')

B = te.compute((n,), lambda i: te.sum(A[i, l], axis=l), name='B')

s = te.create_schedule(B.op)

print(tvm.lower(s, [A, B], simple_mode=True))
print("---------cutting line---------")

s[B].parallel(B.op.reduce_axis[0])
print(tvm.lower(s, [A, B], simple_mode=True))