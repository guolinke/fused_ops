import torch
import torch.nn.functional as F
import sys 
sys.path.append('..')
import fused_bias_dropout_add_cuda
from fused_ops.bias_dropout_add import BiasDropoutAddFunction

bias_dropout_add_func = BiasDropoutAddFunction.apply

def test_err(bsz, dim, seq_len, p):
    torch.manual_seed(0)
    bias = torch.rand(dim).cuda().half()
    x = torch.rand((seq_len, bsz, dim)).cuda().half()
    residual = torch.rand((seq_len, bsz, dim)).cuda().half()
    grad = torch.rand((seq_len, bsz, dim)).cuda().half()
    y1, mask = fused_bias_dropout_add_cuda.forward(x, bias, residual, True, p, None)
    mask_ = torch.zeros((seq_len, bsz, dim), dtype=torch.uint8)
    mask_cpu = mask.cpu().view(seq_len, bsz, dim // 64)
    for i in range(seq_len):
        for j in range(bsz):
            for d in range(dim // 64):
                m = mask_cpu[i, j, d].item()
                for k in range(64):
                    mask_[i, j, d * 64 + k] = (m >> k) & 1;
    mask_ = mask_.cuda()
    y2 = (x + bias) * mask_ / (1 - p) + residual
    dy = (y1 - y2).abs()
    print('y err mean', dy.mean().item(), 'max', dy.max().item())
    g1 = fused_bias_dropout_add_cuda.backward(grad, mask, p)
    g2 = grad * mask_ / (1 - p)
    dg = (g1 - g2).abs()
    print('g err mean', dg.mean().item(), 'max', dg.max().item())

def test_speed(name, bsz, dim, seq_len, n, f):
    torch.manual_seed(0)
    bias = torch.rand(dim).cuda().half()
    x = torch.rand((seq_len, bsz, dim)).cuda().half()
    residual = torch.rand((seq_len, bsz, dim)).cuda().half()
    grad = torch.rand((seq_len, bsz, dim)).cuda().half()
    bias.requires_grad_(True)
    x.requires_grad_(True)
    residual.requires_grad_(True)
    forward_time = 0
    backward_time = 0
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for i in range(n):
        bias.grad = None
        x.grad = None
        residual.grad = None
        start.record()
        y = f(x, bias, residual)
        end.record()
        torch.cuda.synchronize()
        forward_time += start.elapsed_time(end)
        
        start.record()
        y.backward(grad)
        end.record()
        torch.cuda.synchronize()
        backward_time += start.elapsed_time(end)

    print(name, 'forward', forward_time / n, 'ms, backward', backward_time / n, 'ms')

test_err(16, 768, 512, 0.1)
test_speed('new', 16, 768, 512, 10000, lambda x, b, r: F.dropout(x + b, 0.1) + r)
test_speed('old', 16, 768, 512, 10000, lambda x, b, r: bias_dropout_add_func(x, b, r, True, 0.1))