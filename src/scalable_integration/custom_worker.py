import os

from distributed import Worker, get_worker
import tvm

from tvm_te_operators.operator_generic import TVMOperator
from torch_operators.operator_generic import TorchOperator
from jax_operators.operator_generic import JAXOperator
from baseline.operator_generic import BaselineOperator

operators = {
    "tvm": TVMOperator,
    "torch_c": TorchOperator,
    "torch_n": TorchOperator,
    "jax": JAXOperator,
    "baseline": BaselineOperator,
}

def get_operator():
    worker = get_worker()
    return worker.get_operator()



class DaskOperatorWorker(Worker):
    def __init__(
        self,
        *args,
        # locks,
        rt,
        device,
        attribute,
        base_path = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if rt == "tvm":
            dev = tvm.cuda() if device == "gpu" else tvm.cpu()
            module = os.path.join(base_path, f"{attribute}_{device}.so")
            self._operator = operators[rt](module_path=module, dev=dev)
        else:
            self._operator = operators[rt](operator=attribute)

    def get_operator(self):
        return self._operator
