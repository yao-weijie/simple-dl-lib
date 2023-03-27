# simple-dl-lib

一个模仿 pytorch api 的简易深度学习库, 用 numpy 实现.

目录结构如下

```
DLlib
├── mydata
│   ├── dataloader.py
│   ├── dataset.py
│   └── __init__.py
├── mynn
│   ├── activation.py
│   ├── basemodule.py
│   ├── conv.py
│   ├── __init__.py
│   ├── init.py
│   ├── linear.py
│   ├── loss.py
│   ├── pool.py
│   ├── regularization.py
│   └── variable.py
└── myoptim
    ├── __init__.py
    └── optim.py
```

基本实现了:

- `nn.Sequential` 方式搭建网络;
- 常见的`linear`, `pool`, `Conv` , `activation`, `Loss`, `optim` 层等;
- 除了`Conv` 层之外的**前向**, **反向**传播

MLP 的 demo :

```python
import DLlib.mydata as mydata
import DLlib.mynn as mynn
import DLlib.myoptim as myoptim

dataset_root = "~/Datasets"
train_loader = mydata.DataLoader(
    dataset=mydata.MNIST(root=dataset_root, train=True), batch_size=100, shuffle=True
)
test_loader = mydata.DataLoader(
    dataset=mydata.MNIST(root=dataset_root, train=False), batch_size=500, shuffle=False
)

net = mynn.Sequential(
    mynn.Flatten(),
    mynn.Linear(in_features=28 * 28, out_features=56),
    mynn.ReLU(),
    mynn.Linear(in_features=56, out_features=15),
    mynn.ReLU(),
    mynn.Linear(in_features=15, out_features=10),
)

loss_fn = mynn.CrossEntropyLoss()
optimizer = myoptim.SGD(net.parameters(), lr=0.01)

train_epochs = 100
for epoch in range(train_epochs):
    net.train()
    loss_sum = 0.0
    for images, labels in train_loader:
        output = net(images)
        loss = loss_fn(output, labels)
        loss_sum += loss.item()
        loss_grad = loss_fn.backward()
        net.backward(loss_grad)
        optimizer.step()
        optimizer.zero_grad()

    net.eval()
    acc_sum = 0.0
    for images, labels in test_loader:
        output = net(images)
        acc_sum += sum(output.argmax(axis=1) == labels)
    acc = acc_sum / len(test_loader.dataset)
    print(f"train epoch: {epoch+1}, loss: {loss_sum:.3f}, test acc: {acc:.3f}")
```

```
module-1-Flatten
module-2-Linear: in_features=784, out_features=56, bias=True
module-3-ReLU
module-4-Linear: in_features=56, out_features=15, bias=True
module-5-ReLU
module-6-Linear: in_features=15, out_features=10, bias=True

train epoch: 1, loss: 74549.336, test acc: 0.902
train epoch: 2, loss: 16239.636, test acc: 0.926
train epoch: 3, loss: 12061.245, test acc: 0.941
train epoch: 4, loss: 10211.680, test acc: 0.949
train epoch: 5, loss: 9248.711, test acc: 0.948
train epoch: 6, loss: 8239.660, test acc: 0.954
train epoch: 7, loss: 7529.682, test acc: 0.952
train epoch: 8, loss: 6993.986, test acc: 0.957
train epoch: 9, loss: 6675.860, test acc: 0.941
train epoch: 10, loss: 6261.238, test acc: 0.954

...

train epoch: 91, loss: 2350.065, test acc: 0.963
train epoch: 92, loss: 2252.226, test acc: 0.960
train epoch: 93, loss: 2807.070, test acc: 0.959
train epoch: 94, loss: 2443.823, test acc: 0.962
train epoch: 95, loss: 2189.885, test acc: 0.963
train epoch: 96, loss: 2368.156, test acc: 0.962
train epoch: 97, loss: 2652.633, test acc: 0.962
train epoch: 98, loss: 2568.504, test acc: 0.961
train epoch: 99, loss: 3924.259, test acc: 0.963
train epoch: 100, loss: 2632.744, test acc: 0.965

```

TODO:

- `Conv2d` 层的前向传播应该没什么问题, 反向传播还有 bug
