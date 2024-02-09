# Reference: https://medium.com/@rizqinur2010/partial-derivatives-chain-rule-using-torch-autograd-grad-a8b5917373fa

import torch
import numpy as np

# 모델 정의
class SimpleModel(torch.nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.linear = torch.nn.Linear(3, 2)

    def forward(self, x):
        return self.linear(x)

# 모델 및 입력 데이터 생성
model  = SimpleModel()
x      = torch.tensor([[2.0, 3.0, 1.0], [1.0, 2.0, 0.5]], requires_grad=True)
target = torch.tensor([[1.0, 0.0], [0.0, 0.5]])

# 출력 계산
y = model(x)
print(y, '\n')

# 로스 계산
L      = torch.nn.functional.mse_loss(y, target)
dydx1   = torch.autograd.grad(outputs=y, inputs=x, grad_outputs=torch.ones_like(y), create_graph=True, retain_graph=True)  # dy/dx * grad_output = dy/dx
dLdy   = torch.autograd.grad(outputs=L, inputs=y                                 , create_graph=True, retain_graph=True)  # dL/dy

y[0][0].backward(retain_graph=True)
dydx2 = x.grad  # cf. y.grad = dL/dy
print("▶ When still backwarding ◀ \n dy/dx2 (y[0][0].backward) :\n", dydx2[0].detach().numpy()  ,'\n')
print("▶ When still backwarding ◀ \n dLdy   :\n", dLdy[0].detach().numpy()  ,'\n')

y[0][1].backward(retain_graph=True)
dydx3 = x.grad  # cf. y.grad = dL/dy

# 결과 출력
print("dy/dx1 ( torch.autograd.grad ) :\n", dydx1[0].detach().numpy()  ,'\n')
print("dy/dx2 ( y[0][0].backward    ) :\n", dydx2[0].detach().numpy()  ,'\n')
print("dy/dx3 ( y[0][1].backward    ) :\n", dydx3[0].detach().numpy()  ,'\n')

print('\n\n-----------------------')
print("dLdy   :\n", dLdy[0].detach().numpy()  ,'\n')
