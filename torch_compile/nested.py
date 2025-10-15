import torch


def complex_conjugate(z):
    return torch.conj(z)

@torch.compiler.disable(recursive=False)
def complex_function(real, imag):
    # Assuming this function cause problems in the compilation
    z = torch.complex(real, imag)
    return complex_conjugate(z)

def outer_function():
    real = torch.tensor([2, 3], dtype=torch.float32).cuda(0)
    imag = torch.tensor([4, 5], dtype=torch.float32).cuda(0)
    z = complex_function(real, imag)
    return torch.abs(z)

# Try to compile the outer_function
try:
    opt_outer_function = torch.compile(outer_function)
    print(opt_outer_function())
except Exception as e:
    print("Compilation of outer_function failed:", e)