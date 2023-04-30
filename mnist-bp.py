import DLlib.mydata as mydata
import DLlib.nn as nn
import DLlib.optim as optim

dataset_root = "~/Datasets"
train_loader = mydata.DataLoader(
    dataset=mydata.MNIST(root=dataset_root, train=True), batch_size=100, shuffle=True
)
test_loader = mydata.DataLoader(
    dataset=mydata.MNIST(root=dataset_root, train=False), batch_size=500, shuffle=False
)

net = nn.Sequential(
    nn.Flatten(),
    nn.Linear(in_features=28 * 28, out_features=56),
    nn.ReLU(),
    nn.Linear(in_features=56, out_features=15),
    nn.ReLU(),
    nn.Linear(in_features=15, out_features=10),
)
print(net)

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01)

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

    optimizer.lr *= 0.99

    net.eval()
    acc_sum = 0.0
    for images, labels in test_loader:
        output = net(images)
        acc_sum += sum(output.argmax(axis=1) == labels)
    acc = acc_sum / len(test_loader.dataset)
    print(f"train epoch: {epoch+1}, loss: {loss_sum:.3f}, test acc: {acc:.3f}")
