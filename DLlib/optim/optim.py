class SGD(object):
    def __init__(self, parameters, lr):
        super(SGD, self).__init__()
        self.parameters = parameters
        self.lr = lr

    def step(self):
        for p in self.parameters:
            p -= self.lr * p.grad

    def zero_grad(self):
        for p in self.parameters:
            p.zero_grad()
