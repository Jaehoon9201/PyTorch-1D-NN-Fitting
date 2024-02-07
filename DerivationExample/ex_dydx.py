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
print(y)
# 로스 계산
L = torch.nn.functional.mse_loss(y, target)

dydx               = torch.autograd.grad(outputs=y, inputs=x, grad_outputs=torch.ones_like(y), create_graph=True, retain_graph=True)  # dy/dx * grad_output = dy/dx
dLdy               = torch.autograd.grad(outputs=L, inputs=y                                 , create_graph=True, retain_graph=True)  # dL/dy

# 그래디언트 계산 방법 1, 2
dLdx               = torch.autograd.grad(outputs=y, inputs=x, grad_outputs=dLdy[0]                              , retain_graph=True)  # dy/dx * dL/dy = dL/dx
dLdx2              = torch.autograd.grad(outputs=L, inputs=x                                                    , retain_graph=True)  # dL/dx

# 그래디언트 계산 방법 3
L.backward(retain_graph=True)
dLdx3 = x.grad  # cf. y.grad = dL/dy

# 결과 출력 1
print("dy/dx                   :\n", dydx[0].detach().numpy()  ,'\n')
print("dL/dy                   :\n", dLdy[0].detach().numpy()  ,'\n')
print("dL/dx (=dy/dx * dL/dy)  :\n", dLdx[0].detach().numpy()  ,'\n')
# print('dy/dx * dL/dy           :\n', np.dot(  np.transpose(dLdy[0].detach().numpy())   ,   (dydx[0].detach().numpy())   ) ,'\n') # ???

# 결과 출력 2
print("dL/dx                   :\n", dLdx2[0].detach().numpy() ,'\n')

# 결과 출력 3
print("dL/dx                   :\n", dLdx3.detach().numpy()    ,'\n')

