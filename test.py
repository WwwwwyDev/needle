import needle as ndl
import numpy as np
x1 = ndl.Tensor([[0.1, 0.7, 0.9, 0.3, 1.0, 0.8, 0.2, 0.05, 0.16, 0.89]], dtype='float32')
x2 = ndl.Tensor([[0.1, 0.1, 0.2, 0.3, 0.1, 0.4, 0.2, 0.05, 0.16, 0.2]], dtype='float32')
x3 = ndl.Tensor([[0.56, 0.3, 0.4, 0.38, 0.4, 0.8, 0.8, 0.3, 0.10, 0.97]], dtype='float32')
X = [x1,x2,x3]
y1 = ndl.Tensor(3)
y2 = ndl.Tensor(1)
y3 = ndl.Tensor(2)
Y = [y1,y2,y3]
model = ndl.nn.Sequential(ndl.nn.Linear(10, 6),
                          ndl.nn.LayerNorm1d(6),
                          ndl.nn.ReLU(),
                          ndl.nn.Dropout(),
                          ndl.nn.Linear(6,4),
                          ndl.nn.LayerNorm1d(4))
                        #   ndl.nn.ReLU(),
                        #   ndl.nn.Dropout(0.4),
                        #   ndl.nn.Linear(6,4))
optim = ndl.optim.Adam(model.parameters(),lr = 0.001)
Lossf =  ndl.nn.SoftmaxLoss()
model.train()
# y_hat = model(x.reshape([2, 10]))
# loss = Lossf(y_hat, y.reshape([2,4]))
# loss.backward()
# print(model.parameters()[0].grad)
# optim.step()
# print(model.parameters()[0].grad)
for epoch in range(1,10000):
    for i,x in enumerate(X):
        o = model(x)
        loss = Lossf(o, Y[i])
        loss.backward()
        optim.step()
    if epoch % 100 == 0:
        print(loss)

model.eval()
print(model(x1).numpy())
print(model(x2).numpy())
print(model(x3).numpy())
print(np.argmax(model(x1).numpy()))
print(np.argmax(model(x2).numpy()))
print(np.argmax(model(x3).numpy()))
