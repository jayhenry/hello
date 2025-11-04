import torch

a = torch.tensor([2., 3.], requires_grad=True)
b = torch.tensor([6., 4.], requires_grad=True)

Q = 3*a**3 - b**2  # Q is vector

# 因为Q是向量，所以backward需要提供一个gradient tensor
Q_grad = torch.tensor([1., 1.])  # vector
Q.backward(gradient=Q_grad)  # do the vector-Jacobian product
# More about vector-Jacobian product: https://docs.pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html#optional-reading-vector-calculus-using-autograd

print(f"9*a**2 == a.grad: {9*a**2 == a.grad}")
print(f"a.grad: {a.grad}")
print("-"*10)
print(f"-2*b == b.grad: {-2*b == b.grad}")
print(f"b.grad: {b.grad}")