import tvm.testing
from tvm import te
import numpy as np

from tvm_te_operators.geometric.recursive_gaussian import (
    DericheFilter3D,
    RecursiveGaussianFilter
)


if __name__ == "__main__":
    from tvm.contrib import tedd
    a0 = DericheFilter3D(3.0, 0, 0)
    a1 = DericheFilter3D(3.0, 1, 0)
    a2 = DericheFilter3D(3.0, 2, 0)
    r = RecursiveGaussianFilter(3, order=(0, None, None))
    s0 = te.create_schedule([op.op for op in a0.computation_context["result"]])
    s1 = te.create_schedule([op.op for op in a1.computation_context["result"]])
    s2 = te.create_schedule([op.op for op in a2.computation_context["result"]])
    s3 = te.create_schedule([op.op for op in r.computation_context["result"]])
    tedd.viz_dataflow_graph(s0, dot_file_path="deriche0.dot")
    tedd.viz_dataflow_graph(s1, dot_file_path="deriche1.dot")
    tedd.viz_dataflow_graph(s2, dot_file_path="deriche2.dot")
    tedd.viz_dataflow_graph(s3, dot_file_path="rgf.dot")
    tgt = tvm.target.Target(target="llvm", host="llvm")
    aux = [a0.computation_context["input"], *a0.computation_context["result"]]
    d0 = tvm.build(s0,  aux, tgt, name="d0")
    aux = [a1.computation_context["input"], *a1.computation_context["result"]]
    d1 = tvm.build(s1, aux, tgt, name="d1")
    aux = [a2.computation_context["input"], *a2.computation_context["result"]]
    d2 = tvm.build(s2, aux, tgt, name="d2")
    aux = [r.computation_context["input"], *r.computation_context["result"]]
    rgf = tvm.build(s3, aux, tgt, name="rgf")
    d0.export_library("d0.so")
    d1.export_library("d1.so")
    d2.export_library("d2.so")
    rgf.export_library("rgf.so")

    with open("D0fa_wop.txt", "w") as f:
        aux = [a0.computation_context["input"], *a0.computation_context["result"]]
        f.write(str(tvm.lower(s0, aux, simple_mode=True)))

    with open("D1fa_wop.txt", "w") as f:
        aux = [a1.computation_context["input"], *a1.computation_context["result"]]
        f.write(str(tvm.lower(s1, aux, simple_mode=True)))

    with open("D2fa_wop.txt", "w") as f:
        aux = [a2.computation_context["input"], *a2.computation_context["result"]]
        f.write(str(tvm.lower(s2, aux, simple_mode=True)))
