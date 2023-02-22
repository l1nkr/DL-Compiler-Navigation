import tvm
from tvm import te
import numpy as np

# 构造用于进行pass的样本

n = tvm.tir.const(128, "int32")
a = te.placeholder((n, ), name="a")
b = te.placeholder((n, ), name="b")
c = te.compute((n, ), lambda i : a[i] + b[i], name="c")

sch = te.create_schedule(c.op)
ir = tvm.lower(sch, [a, b, c])
print(ir)
