import tvm.testing
from tvm import te
import numpy as np

from tvm_te_operators.geometric.gst import (
    GradientStructureTensor,
)

from tvm_te_operators.utils import get_name

if __name__ == "__main__":
    from tvm.contrib import tedd
    from time import perf_counter

    build_start = perf_counter()
    x = te.var("x")
    y = te.var("y")
    z = te.var("z")
    X = te.placeholder((x, y, z), dtype="float64", name=get_name("X"))
    wrapper = GradientStructureTensor(
        computation_context={"x": x, "y": y, "z": z, "X": X}
    )

    tgt = tvm.target.Target(target="llvm", host="llvm")
    schedule = te.create_schedule(
        [result.op for result in wrapper.computation_context["result"]]
    )
    tedd.viz_dataflow_graph(schedule, dot_file_path="gst.dot")
    gst = tvm.build(
        schedule,
        [X, *[result for result in wrapper.computation_context["result"]]],
        tgt,
        name="gst",
    )
    build_time = perf_counter() - build_start
    gst.export_library("gst.so")
    print(f"Build: {build_time}")
    