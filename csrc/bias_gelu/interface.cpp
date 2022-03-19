#include <torch/extension.h>
#include <ATen/Generator.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <vector>

extern torch::Tensor (*bias_gelu_torch_forward_cuda)(const torch::Tensor &x, const torch::Tensor &bias);
extern torch::Tensor (*bias_gelu_fast_forward_cuda)(const torch::Tensor &x, const torch::Tensor &bias);
extern torch::Tensor (*bias_gelu_torch_backward_cuda)(const torch::Tensor &x, const torch::Tensor &bias, const torch::Tensor &grad);
extern torch::Tensor (*bias_gelu_fast_backward_cuda)(const torch::Tensor &x, const torch::Tensor &bias, const torch::Tensor &grad);

// C++ interface

#define CHECK_CUDA(x) AT_ASSERTM(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor bias_gelu_torch_forward(const torch::Tensor &x, const torch::Tensor &bias) {
    CHECK_INPUT(x);
    CHECK_INPUT(bias);
    AT_ASSERTM(bias.dim() == 1, "expected 1D tensor");
    AT_ASSERTM(bias.size(-1) == x.size(-1), "the dimension of bias and the last dimension of the input should be the same");
    AT_ASSERTM(x.scalar_type() == bias.scalar_type(), "the types mismatch");
    return bias_gelu_torch_forward_cuda(x, bias);
}

torch::Tensor bias_gelu_fast_forward(const torch::Tensor &x, const torch::Tensor &bias) {
    CHECK_INPUT(x);
    CHECK_INPUT(bias);
    AT_ASSERTM(bias.dim() == 1, "expected 1D tensor");
    AT_ASSERTM(bias.size(-1) == x.size(-1), "the dimension of bias and the last dimension of the input should be the same");
    AT_ASSERTM(x.scalar_type() == bias.scalar_type(), "the types mismatch");
    return bias_gelu_fast_forward_cuda(x, bias);
}

torch::Tensor bias_gelu_torch_backward(const torch::Tensor &x, const torch::Tensor &bias, const torch::Tensor &grad) {
    CHECK_INPUT(x);
    CHECK_INPUT(bias);
    AT_ASSERTM(bias.dim() == 1, "expected 1D tensor");
    AT_ASSERTM(bias.size(-1) == x.size(-1), "the dimension of bias and the last dimension of the input should be the same");
    AT_ASSERTM(x.scalar_type() == bias.scalar_type() && x.scalar_type() == grad.scalar_type(), "the types mismatch");
    return bias_gelu_torch_backward_cuda(x, bias, grad);
}

torch::Tensor bias_gelu_fast_backward(const torch::Tensor &x, const torch::Tensor &bias, const torch::Tensor &grad) {
    CHECK_INPUT(x);
    CHECK_INPUT(bias);
    AT_ASSERTM(bias.dim() == 1, "expected 1D tensor");
    AT_ASSERTM(bias.size(-1) == x.size(-1), "the dimension of bias and the last dimension of the input should be the same");
    AT_ASSERTM(x.scalar_type() == bias.scalar_type() && x.scalar_type() == grad.scalar_type(), "the types mismatch");
    return bias_gelu_fast_backward_cuda(x, bias, grad);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward_torch", &bias_gelu_torch_forward, "bias gelu torch -- Forward.");
    m.def("forward_fast", &bias_gelu_fast_forward, "bias gelu fast -- Forward.");
    m.def("backward_torch", &bias_gelu_torch_backward, "bias gelu torch -- Backward.");
    m.def("backward_fast", &bias_gelu_fast_backward, "bias gelu fast -- Backward.");
}
