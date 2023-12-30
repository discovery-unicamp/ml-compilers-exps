import tvm


class TVMOperator:
    def __init__(self, module_path, outputs=1):
        self._operator = tvm.runtime.load_module(module_path)
        self.outputs = outputs

    def transform(self, *args):
        self._operator(*args)
        return args[-self.outputs :]
