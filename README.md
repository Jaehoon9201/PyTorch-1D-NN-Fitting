# PyTorch-1D-NN-Fitting

- Author : Jaehoon Shim


# Derivation Example
## ex_dydx.py

dydx = torch.autograd.grad(outputs=y, inputs=x, grad_outputs=torch.ones_like(y))

```python
dydx = torch.autograd.grad(outputs=y, inputs=x, grad_outputs=torch.ones_like(y), create_graph=True, retain_graph=True) 
```

![image](https://github.com/Jaehoon9201/PyTorch-1D-NN-Fitting/assets/71545160/253285e0-e912-4446-a8fa-30faec444a72)

## ex_y1dx_y2dx.py

gradients = gradients1 + gradients2

```python
gradients  = torch.autograd.grad(output.to(device), x.to(device), grad_outputs=torch.tensor([[1, 1]]).to(device), retain_graph = True)
gradients1 = torch.autograd.grad(output[0][0].to(device), x.to(device), grad_outputs=torch.tensor(1).to(device), retain_graph = True)
gradients2 = torch.autograd.grad(output[0][1].to(device), x.to(device), grad_outputs=torch.tensor(1).to(device), retain_graph = True)
```

![image](https://github.com/Jaehoon9201/PyTorch-1D-NN-Fitting/assets/71545160/f5018cb4-8881-477f-b118-6860d0221a74)

## ex_dydx_methodf2(not_recommended_without_cloning).py

dydx = dydx3[0].detach().numpy()  == dydx2 = x.grad  

```python
y[0][0].backward(retain_graph=True)
dydx2 = x.grad  # cf. y.grad = dL/dy
y[0][1].backward(retain_graph=True)
dydx3 = x.grad  # cf. y.grad = dL/dy
```

## ex_dydx_methodf2(recommended_with_cloning).py

dydx2[0].detach().numpy() + dydx3[0].detach().numpy()  
== x.grad   
== torch.autograd.grad(outputs=y, inputs=x)

```python
y[0][0].backward(retain_graph=True)
dydx2 = x.grad.clone().detach().numpy()
x.grad.zero_()
y[0][1].backward(retain_graph=True)
dydx3 = x.grad.clone().detach().numpy()
```

![image](https://github.com/Jaehoon9201/PyTorch-1D-NN-Fitting/assets/71545160/1d9b65f7-e2ba-4800-bbae-a5936e8b2c4c)


## SUMMARY

Below 'dydx' have all the same results !

```python
y[0][0].backward(retain_graph=True)
y[0][1].backward(retain_graph=True)
dydx = x.grad
``` 

```python
dydx = torch.autograd.grad(outputs=y, inputs=x, grad_outputs=torch.ones_like(y))
```

```python
dydx1 = torch.autograd.grad(outputs=y[0][0], inputs=x, grad_outputs=torch.ones_like(y), create_graph=True, retain_graph=True) 
dydx2 = torch.autograd.grad(outputs=y[0][1], inputs=x, grad_outputs=torch.ones_like(y), create_graph=True, retain_graph=True) 
dydx = dydx1+ dydx2
```



