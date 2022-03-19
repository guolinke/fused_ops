#include <torch/extension.h>
#include <ATen/Generator.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <vector>

std::vector<c10::optional<torch::Tensor>> bias_dropout_add_forward_cuda(const torch::Tensor &x,
    const torch::Tensor &bias, const torch::Tensor &residual, bool is_training, float dropout_prob,
    c10::optional<at::Generator> gen_);
torch::Tensor bias_dropout_add_backward_cuda(const torch::Tensor &grad, const torch::Tensor &mask,
    float dropout_prob);

// C++ interface

#define CHECK_CUDA(x) AT_ASSERTM(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::vector<c10::optional<torch::Tensor>> bias_dropout_add_forward(const torch::Tensor &x, const torch::Tensor &bias,
    const torch::Tensor &residual, bool is_training, float dropout_prob, c10::optional<at::Generator> gen_) {
    CHECK_INPUT(x);
    CHECK_INPUT(bias);
    CHECK_INPUT(residual);
    AT_ASSERTM(x.sizes() == residual.sizes(), "the shapes mismatch");
    AT_ASSERTM(bias.dim() == 1, "expected 1D tensor");
    AT_ASSERTM(bias.size(-1) == x.size(-1) && bias.size(-1) == residual.size(-1),
        "the dimension of bias and the last dimension of the input should be the same");
    AT_ASSERTM(x.scalar_type() == bias.scalar_type() && residual.scalar_type() == bias.scalar_type(),
        "the types mismatch");
    return bias_dropout_add_forward_cuda(x, bias, residual, is_training, dropout_prob, gen_);
}

torch::Tensor bias_dropout_add_backward(const torch::Tensor &grad, const torch::Tensor &mask, float dropout_prob) {
    CHECK_INPUT(grad);
    CHECK_INPUT(mask);
    return bias_dropout_add_backward_cuda(grad, mask, dropout_prob);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &bias_dropout_add_forward, "bias dropout add -- Forward.");
    m.def("backward", &bias_dropout_add_backward, "bias dropout add -- Backward.");
}
