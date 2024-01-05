import tvm


class TVMOperator:
    def __init__(self, module_path, dev, outputs=1):
        self._operator = tvm.runtime.load_module(module_path)
        self._dev = dev
        self.outputs = outputs

    def transform(self, *args):
        self._operator(*args)
        self._dev.sync()
        return args[-self.outputs :]
