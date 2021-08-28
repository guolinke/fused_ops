#include <ATen/ATen.h>
#include <ATen/CUDAGeneratorImpl.h>
#include <ATen/cuda/detail/IndexUtils.cuh>
#include <ATen/cuda/detail/TensorInfo.cuh>
#include <c10/cuda/CUDAMathCompat.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <math.h>
#include "util.h"


template <typename scalar_t, typename accscalar_t>
__global__ void add_by_index_kernel(
    const scalar_t* __restrict__ input,
    const int64_t* __restrict__ indices,
    accscalar_t* __restrict__ output,
    int64_t size) {
    int64_t i = blockIdx.x;
    if (i < size) {
        atomicAdd(&output[indices[i]], static_cast<accscalar_t>(input[i]));
    }
}

torch::Tensor add_by_index_cuda(const torch::Tensor &input, const torch::Tensor &indices, int64_t num_embeddings, torch::Tensor &output) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    auto sizes = input.sizes();
    size_t bsz = sizes[0];
    auto type = input.scalar_type();
    const int ThreadsPerBlock = 256;
    if (type == at::ScalarType::BFloat16) {
        add_by_index_kernel<nv_bfloat16, float><<<bsz, ThreadsPerBlock, 0, stream>>>(
                (const nv_bfloat16 *)input.data_ptr(),
                (const int64_t *)indices.data_ptr(),
                (float *)output.data_ptr(),
                bsz);
    } else if (type == at::ScalarType::Half) {
        add_by_index_kernel<half, float><<<bsz, ThreadsPerBlock, 0, stream>>>(
                (const half *)input.data_ptr(),
                (const int64_t *)indices.data_ptr(),
                (float *)output.data_ptr(),
                bsz);

    } else if (type == at::ScalarType::Float) {
        add_by_index_kernel<float, float><<<bsz, ThreadsPerBlock, 0, stream>>>(
                (const float *)input.data_ptr(),
                (const int64_t *)indices.data_ptr(),
                (float *)output.data_ptr(),
                bsz);
    }
    return output;
}