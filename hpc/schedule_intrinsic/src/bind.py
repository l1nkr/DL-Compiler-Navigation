import tvm
from tvm import te

n = 1024
A = te.placeholder((n,), name='A')
k = te.reduce_axis((0, n), name='k')

B = te.compute((1,), lambda i: te.sum(A[k], axis=k), name='B')

s = te.create_schedule(B.op)

ko, ki = s[B].split(B.op.reduce_axis[0], factor=16)

print(tvm.lower(s, [A, B], simple_mode=True))
print("---------cutting line---------")

s[B].bind(ko, te.thread_axis("blockIdx.x"))
s[B].bind(ki, te.thread_axis("threadIdx.x"))

print(tvm.lower(s, [A, B], simple_mode=True))