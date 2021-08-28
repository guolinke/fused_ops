#include <torch/extension.h>
#include <ATen/Generator.h>
#include <ATen/CUDAGeneratorImpl.h>
#include <vector>

torch::Tensor add_by_index_cuda(const torch::Tensor &input, const torch::Tensor &indices, int64_t num_embeddings, torch::Tensor &output);

// C++ interface

#define CHECK_CUDA(x) AT_ASSERTM(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


torch::Tensor add_by_index(const torch::Tensor &input, const torch::Tensor &indices, int64_t num_embeddings, torch::Tensor &output) {
    CHECK_INPUT(input);
    CHECK_INPUT(indices);
    return add_by_index_cuda(input, indices, num_embeddings, output);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &add_by_index, "add_by_index forward");
}
