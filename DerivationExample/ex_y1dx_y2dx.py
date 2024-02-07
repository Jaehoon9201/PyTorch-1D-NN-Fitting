import torch
from torch.autograd import grad

device = 'cuda'
# .pt load
class SimpleModel(torch.nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.linear = torch.nn.Linear(10, 2)

    def forward(self, x):
        return self.linear(x)

# 모델 및 입력 데이터 생성
model  = SimpleModel()
model = model.to(device)
model = model.eval()
# ex data
x = torch.tensor([[4.1547e-03,  4.4907e-02, -2.7384e-03, 9.9574e-04, 3.1319e-02,
                   9.3274e-02, -8.1103e-01, -2.3757e-01, 1.2081e-01, 1.0000e+00]],
                 device='cuda:0', requires_grad=True)

output    = model(x.to(device))
print(output)
print(output[0][0])
print(output[0][1])

# torch.autograd.grad --> calculating gradients
gradients  = torch.autograd.grad(output.to(device), x.to(device), grad_outputs=torch.tensor([[1, 1]]).to(device), retain_graph = True)
gradients1 = torch.autograd.grad(output[0][0].to(device), x.to(device), grad_outputs=torch.tensor(1).to(device), retain_graph = True)
gradients2 = torch.autograd.grad(output[0][1].to(device), x.to(device), grad_outputs=torch.tensor(1).to(device), retain_graph = True)

# outputs
print("dy/dx  : \n", gradients)
print("dy1/dx : \n", gradients1)
print("dy2/dx : \n", gradients2)
