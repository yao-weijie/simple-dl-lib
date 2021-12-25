from collections import OrderedDict
import numpy as np


class Module(object):
    def __init__(self, *args, **kwargs):
        self.input = None
        self.output = None
        self.training = True
        self._parameters = OrderedDict()
        self._modules = OrderedDict()

    def forward(self, *args):
        """前向传递"""
        raise NotImplementedError

    def backward(self, delta):
        """反向传播"""
        raise NotImplementedError

    def zero_grad(self):
        """"梯度清零"""
        raise NotImplementedError

    def train(self):
        """训练模式"""
        for m in self._modules.values():
            m.training = True

    def eval(self):
        """测试模式"""
        for m in self._modules.values():
            m.training = False
    
    def parameters(self):
        return self._parameters.values()

    def __call__(self, *args):
        """对象当成函数可调用"""
        return self.forward(*args)

    def __str__(self):
        return self.__class__.__name__ + '\n'


class Sequential(Module):
    def __init__(self, *modules):
        super(Sequential, self).__init__()
        self.register_modules(modules)

    def register_modules(self, modules):
        for i, m in enumerate(modules):
            assert isinstance(m, Module)
            
            module_name = f'module-{i+1}-{m.__class__.__name__}'
            self._modules[module_name] = m

            for param_key in m._parameters:
                if 'weight' in param_key:
                    param_name = f'module-{i+1}-weight'
                elif 'bias' in param_key:
                    param_name = f'module-{i+1}-bias'
                else:
                    param_name = f'module-{i+1}-{param_key}'
                self._parameters[param_name] = m._parameters[param_key]

    def forward(self,  x):
        for module in self._modules.values():
            x = module(x)

        return x

    def backward(self, delta):
        """delta: 后层传来的梯度"""
        for m in reversed(self._modules.values()):
            delta = m.backward(delta)
        return delta

    def __str__(self):
        s = ''
        for i, (k, v) in enumerate(self._modules.items()):
            s += f'module-{i+1}-{v}'
        return s
