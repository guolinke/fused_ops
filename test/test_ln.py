import torch
import sys 
sys.path.append('..')
import fused_layernorm_cuda
import fused_layernorm_fast_cuda
import fused_layernorm_backward_gamma_beta_cuda
import time
import fused_ops.layernorm
import fused_ops.layernorm_fast

def test_speed(name, bsz, dim, seq_len, n, cls):
    torch.manual_seed(0)
    x = torch.rand((seq_len, bsz, dim)).cuda().half()
    grad = torch.rand((seq_len, bsz, dim)).cuda().half()
    ln = cls(dim).cuda().half().requires_grad_(True)
    x.requires_grad_(True)
    forward_time = 0
    backward_time = 0
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for i in range(n):
        x.grad = None
        ln.weight.grad = None
        ln.bias.grad = None
        start.record()
        y = ln(x)
        end.record()
        torch.cuda.synchronize()
        forward_time += start.elapsed_time(end)
        start.record()
        y.backward(grad)
        end.record()
        torch.cuda.synchronize()
        backward_time += start.elapsed_time(end)

    print(name, 'forward', forward_time / n, 'ms, backward', backward_time / n, 'ms')

test_speed('old', 16, 768, 512, 10000, fused_ops.layernorm.FusedLayerNorm)
test_speed('new', 16, 768, 512, 10000, fused_ops.layernorm_fast.FusedLayerNormFast)
