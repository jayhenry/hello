import torch
import numpy as np

torch.cuda.set_device(0)
print(f"Using GPU: {torch.cuda.get_device_name(0)}")

# Returns the result of running `fn()` and the time it took for `fn()` to run,
# in seconds. We use CUDA events and synchronization for the most accurate
# measurements.
def timed(fn):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    result = fn()
    end.record()
    torch.cuda.synchronize()
    return result, start.elapsed_time(end) / 1000

# Generates random input and targets data for the model, where `b` is
# batch size.
def generate_data(b):
    return (
        torch.randn(b, 3, 128, 128).to(torch.float32).cuda(),
        torch.randint(1000, (b,)).cuda(),
    )

N_ITERS = 10

from torchvision.models import densenet121
def init_model():
    return densenet121().to(torch.float32).cuda()


def bench_eval():
    model = init_model()
    
    # Reset since we are using a different mode.
    import torch._dynamo
    torch._dynamo.reset()
    
    # The "reduce-overhead" mode uses CUDA graphs to further reduce the overhead of Python. 
    # You may might also notice that the second time we run our model with torch.compile is significantly slower than the other runs, 
    # although it is much faster than the first run. 
    # This is because the "reduce-overhead" mode runs a few warm-up iterations for CUDA graphs.
    model_opt = torch.compile(model, mode="reduce-overhead")
    
    print("warming up".center(50, "="))
    inp = generate_data(16)[0]
    with torch.no_grad():
        print("eager:", timed(lambda: model(inp))[1])
        print("compile:", timed(lambda: model_opt(inp))[1])
    
    print("benchmarking eval".center(50, "="))
    eager_times = []
    for i in range(N_ITERS):
        inp = generate_data(16)[0]
        with torch.no_grad():
            _, eager_time = timed(lambda: model(inp))
        eager_times.append(eager_time)
        print(f"eager eval time {i}: {eager_time}")
    
    print("~" * 10)
    
    compile_times = []
    for i in range(N_ITERS):
        inp = generate_data(16)[0]
        with torch.no_grad():
            _, compile_time = timed(lambda: model_opt(inp))
        compile_times.append(compile_time)
        print(f"compile eval time {i}: {compile_time}")
    print("~" * 10)
    
    eager_med = np.median(eager_times)
    compile_med = np.median(compile_times)
    speedup = eager_med / compile_med
    assert(speedup > 1)
    print(f"(eval) eager median: {eager_med}, compile median: {compile_med}, speedup: {speedup}x")
    print("~" * 10)


def train(mod, data, opt):
    opt.zero_grad(True)
    pred = mod(data[0])
    loss = torch.nn.CrossEntropyLoss()(pred, data[1])
    loss.backward()
    opt.step()
    
def bench_train():
    model = init_model()
    opt = torch.optim.Adam(model.parameters())
    
    eager_times = []
    for i in range(N_ITERS):
        inp = generate_data(16)
        _, eager_time = timed(lambda: train(model, inp, opt))
        eager_times.append(eager_time)
        print(f"eager train time {i}: {eager_time}")
    print("~" * 10)
    
    model = init_model()
    opt = torch.optim.Adam(model.parameters())
    train_opt = torch.compile(train, mode="reduce-overhead")
    
    compile_times = []
    for i in range(N_ITERS):
        inp = generate_data(16)
        _, compile_time = timed(lambda: train_opt(model, inp, opt))
        compile_times.append(compile_time)
        print(f"compile train time {i}: {compile_time}")
    print("~" * 10)
    
    eager_med = np.median(eager_times)
    compile_med = np.median(compile_times)
    speedup = eager_med / compile_med
    assert(speedup > 1)
    print(f"(train) eager median: {eager_med}, compile median: {compile_med}, speedup: {speedup}x")
    print("~" * 10)

if __name__ == "__main__":
    # bench_eval()
    bench_train()
