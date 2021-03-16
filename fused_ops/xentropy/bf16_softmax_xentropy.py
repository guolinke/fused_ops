import torch
import bf16_fused_xentropy_cuda

class BF16SoftmaxCrossEntropyLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, labels, padding_idx=0, bf16_to_float=False):
        losses, max_log_sum_exp = bf16_fused_xentropy_cuda.forward(
            logits, labels, bf16_to_float)
        if padding_idx >= 0:
            losses.masked_fill_(labels==padding_idx, 0)

        ctx.save_for_backward(logits, max_log_sum_exp, labels,
            torch.LongTensor([padding_idx]))

        return losses

    @staticmethod
    def backward(ctx, grad_loss):
        logits, max_log_sum_exp, labels, padding_idx = ctx.saved_tensors

        if not grad_loss.is_contiguous():
            grad_loss = grad_loss.contiguous()
        if padding_idx >= 0:
            grad_loss.masked_fill_(labels==padding_idx.item(), 0)
        grad_logits = bf16_fused_xentropy_cuda.backward(
            grad_loss, logits, max_log_sum_exp,
            labels)

        return grad_logits, None, None, None, None
