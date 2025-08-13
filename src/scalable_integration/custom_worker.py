import os

from distributed import Worker, get_worker

from torch_operators.operator_generic import TorchOperator
from jax_operators.operator_generic import JAXOperator
from baseline.operator_generic import BaselineOperator

operators = {
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
        rt,
        device,
        attribute,
        base_path = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if rt == "tvm":
            import tvm # avoids issues with torch.compile
            from tvm_te_operators.operator_generic import TVMOperator
            dev = tvm.cuda() if device == "gpu" else tvm.cpu()
            module = os.path.join(base_path, f"{attribute}_{device}.so")
            self._operator = TVMOperator(module_path=module, dev=dev)
        elif rt == "torch_n":
            self._operator = operators[rt](operator=attribute, compile=False)
        else:
            self._operator = operators[rt](operator=attribute)

    def get_operator(self):
        return self._operator
