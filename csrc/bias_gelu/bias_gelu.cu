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

template <typename acc_t>
__device__ acc_t torch_gelu(acc_t y) {
    return normcdff(y) * y;
}

template <typename acc_t>
__device__ acc_t fast_gelu(acc_t y) {
    return y * 0.5 * (1.0 + tanhf(0.79788456 * y * (1 + 0.044715 * y * y)));
}

template <typename acc_t>
__device__ acc_t torch_gelu_back(acc_t y, acc_t g) {
    constexpr acc_t kBeta = M_2_SQRTPI * M_SQRT1_2 * 0.5;
    const acc_t cdf = normcdff(y);
    const acc_t pdf = expf(-0.5f * y * y) * kBeta;
    return g * (cdf + y * pdf);
}

template <typename acc_t>
__device__ acc_t fast_gelu_back(acc_t y, acc_t g) {
    const acc_t tanh_out = tanhf(0.79788456 * y * (1 + 0.044715 * y * y));
    const acc_t ff = 0.5 * y * ((1 - tanh_out * tanh_out) * (0.79788456 + 0.1070322243 * y * y)) + 0.5 * (1 + tanh_out);
    return ff * g;
}

template <typename index_t, typename input_t, typename output_t, typename acc_t, acc_t (*gelu_func)(acc_t)>
__global__ void bias_gelu_forward(output_t *dst, const input_t *src, const input_t *bias, index_t bsz, int dim) {
    for (int j = threadIdx.x; j < dim; j += blockDim.x) {
        if (blockIdx.x < bsz) {
            const index_t idx = blockIdx.x * dim + j;
            const acc_t y = (acc_t)(src[idx] + bias[j]);
            dst[idx] = (output_t)gelu_func(y);
        }
    }
}

template <typename index_t, typename input_t, typename output_t, typename acc_t, acc_t (*gelu_func)(acc_t)>
__global__ void bias_gelu_forward_vec(output_t *dst, const input_t *src, const input_t *bias, index_t bsz, int dim) {
    using VecInType = VecType<input_t, 2>;
    using VecOutType = VecType<output_t, 2>;
    for (int j = threadIdx.x * 2; j < dim; j += blockDim.x * 2) {
        if (blockIdx.x < bsz) {
            const index_t idx = blockIdx.x * dim + j;
            const VecInType s = *(VecInType *)(src + idx);
            const VecInType b = *(VecInType *)(bias + j);
            const acc_t y1 = s.x + b.x;
            const acc_t y2 = s.y + b.y;
            VecOutType d;
            d.x = gelu_func(y1);
            d.y = gelu_func(y2);
            *(VecOutType *)(dst + idx) = d;
        }
    }
}

template <typename index_t, typename input_t, typename output_t, typename acc_t, acc_t (*gelu_back_func)(acc_t, acc_t)>
__global__ void bias_gelu_backward(output_t *dst, const input_t *src, const input_t *bias,
    const input_t *grad, index_t bsz, int dim) {
    for (int j = threadIdx.x; j < dim; j += blockDim.x) {
        if (blockIdx.x < bsz) {
            const index_t idx = blockIdx.x * dim + j;
            const acc_t y = (acc_t)(src[idx] + bias[j]);
            const acc_t g = grad[idx];
            dst[idx] = (output_t)gelu_back_func(y, g);
        }
    }
}

template <typename index_t, typename input_t, typename output_t, typename acc_t, acc_t (*gelu_back_func)(acc_t, acc_t)>
__global__ void bias_gelu_backward_vec(output_t *dst, const input_t *src, const input_t *bias,
    const input_t *grad, index_t bsz, int dim) {
    using VecInType = VecType<input_t, 2>;
    using VecOutType = VecType<output_t, 2>;
    for (int j = threadIdx.x * 2; j < dim; j += blockDim.x * 2) {
        if (blockIdx.x < bsz) {
            const index_t idx = blockIdx.x * dim + j;
            const VecInType s = *(VecInType *)(src + idx);
            const VecInType b = *(VecInType *)(bias + j);
            const VecInType g = *(VecInType *)(grad + idx);
            const acc_t y1 = s.x + b.x;
            const acc_t y2 = s.y + b.y;
            VecOutType d;
            d.x = gelu_back_func(y1, g.x);
            d.y = gelu_back_func(y2, g.y);
            *(VecOutType *)(dst + idx) = d;
        }
    }
}

template <float (*gelu_func)(float)>
torch::Tensor bias_gelu_forward_cuda(const torch::Tensor &x, const torch::Tensor &bias) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    auto sizes = x.sizes();
    size_t bsz = 1;
    for (size_t i = 0; i + 1 < sizes.size(); ++i) {
        bsz *= sizes[i];
    }
    int dim = sizes[sizes.size() - 1];
    auto dst_options = x.options().requires_grad(false);
    torch::Tensor results = torch::empty(sizes, dst_options);
    auto type = x.scalar_type();
    const int ThreadsPerBlock = 256;
    int ThreadsPerBlockVec = DIV_CELL(dim, 256) * 256 % 512 == 0 ? 256 : 128;
    if (type == at::ScalarType::BFloat16) {
        if (dim % 2 == 0) {
            bias_gelu_forward_vec<size_t, nv_bfloat16, nv_bfloat16, float, gelu_func><<<bsz, ThreadsPerBlockVec, 0, stream>>>(
                (nv_bfloat16 *)results.data_ptr(),
                (const nv_bfloat16 *)x.data_ptr(),
                (const nv_bfloat16 *)bias.data_ptr(),
                bsz,
                dim);
        } else {
            bias_gelu_forward<size_t, nv_bfloat16, nv_bfloat16, float, gelu_func><<<bsz, ThreadsPerBlock, 0, stream>>>(
                (nv_bfloat16 *)results.data_ptr(),
                (const nv_bfloat16 *)x.data_ptr(),
                (const nv_bfloat16 *)bias.data_ptr(),
                bsz,
                dim);
        }
    } else if (type == at::ScalarType::Half) {
        if (dim % 2 == 0) {
            bias_gelu_forward_vec<size_t, half, half, float, gelu_func><<<bsz, ThreadsPerBlockVec, 0, stream>>>(
                (half *)results.data_ptr(),
                (const half *)x.data_ptr(),
                (const half *)bias.data_ptr(),
                bsz,
                dim);
        } else {
            bias_gelu_forward<size_t, half, half, float, gelu_func><<<bsz, ThreadsPerBlock, 0, stream>>>(
                (half *)results.data_ptr(),
                (const half *)x.data_ptr(),
                (const half *)bias.data_ptr(),
                bsz,
                dim);
        }
    } else if (type == at::ScalarType::Float) {
        bias_gelu_forward<size_t, float, float, float, gelu_func><<<bsz, ThreadsPerBlock, 0, stream>>>(
            (float *)results.data_ptr(),
            (const float *)x.data_ptr(),
            (const float *)bias.data_ptr(),
            bsz,
            dim);
    }
    return results;
}

template <float (*gelu_back_func)(float, float)>
torch::Tensor bias_gelu_backward_cuda(const torch::Tensor &x, const torch::Tensor &bias, const torch::Tensor &grad) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    auto sizes = x.sizes();
    size_t bsz = 1;
    for (size_t i = 0; i + 1 < sizes.size(); ++i) {
        bsz *= sizes[i];
    }
    int dim = sizes[sizes.size() - 1];
    auto dst_options = x.options().requires_grad(false);
    torch::Tensor results = torch::empty(sizes, dst_options);
    auto type = x.scalar_type();
    const int ThreadsPerBlock = 256;
    int ThreadsPerBlockVec = DIV_CELL(dim, 256) * 256 % 512 == 0 ? 256 : 128;
    if (type == at::ScalarType::BFloat16) {
        if (dim % 2 == 0) {
            bias_gelu_backward_vec<size_t, nv_bfloat16, nv_bfloat16, float, gelu_back_func><<<bsz, ThreadsPerBlockVec, 0, stream>>>(
                (nv_bfloat16 *)results.data_ptr(),
                (const nv_bfloat16 *)x.data_ptr(),
                (const nv_bfloat16 *)bias.data_ptr(),
                (const nv_bfloat16 *)grad.data_ptr(),
                bsz,
                dim);
        } else {
            bias_gelu_backward<size_t, nv_bfloat16, nv_bfloat16, float, gelu_back_func><<<bsz, ThreadsPerBlock, 0, stream>>>(
                (nv_bfloat16 *)results.data_ptr(),
                (const nv_bfloat16 *)x.data_ptr(),
                (const nv_bfloat16 *)bias.data_ptr(),
                (const nv_bfloat16 *)grad.data_ptr(),
                bsz,
                dim);
            }
    } else if (type == at::ScalarType::Half) {
        if (dim % 2 == 0) {
            bias_gelu_backward_vec<size_t, half, half, float, gelu_back_func><<<bsz, ThreadsPerBlockVec, 0, stream>>>(
                (half *)results.data_ptr(),
                (const half *)x.data_ptr(),
                (const half *)bias.data_ptr(),
                (const half *)grad.data_ptr(),
                bsz,
                dim);
        } else {
            bias_gelu_backward<size_t, half, half, float, gelu_back_func><<<bsz, ThreadsPerBlock, 0, stream>>>(
                (half *)results.data_ptr(),
                (const half *)x.data_ptr(),
                (const half *)bias.data_ptr(),
                (const half *)grad.data_ptr(),
                bsz,
                dim);
        }
    } else if (type == at::ScalarType::Float) {
        bias_gelu_backward<size_t, float, float, float, gelu_back_func><<<bsz, ThreadsPerBlock, 0, stream>>>(
            (float *)results.data_ptr(),
            (const float *)x.data_ptr(),
            (const float *)bias.data_ptr(),
            (const float *)grad.data_ptr(),
            bsz,
            dim);
    }
    return results;
}

using ForwardFunc = torch::Tensor (*)(const torch::Tensor &, const torch::Tensor &);

ForwardFunc bias_gelu_torch_forward_cuda = bias_gelu_forward_cuda<torch_gelu>;
ForwardFunc bias_gelu_fast_forward_cuda = bias_gelu_forward_cuda<fast_gelu>;

using BackwardFunc = torch::Tensor (*)(const torch::Tensor &, const torch::Tensor &, const torch::Tensor &);

BackwardFunc bias_gelu_torch_backward_cuda = bias_gelu_backward_cuda<torch_gelu_back>;
BackwardFunc bias_gelu_fast_backward_cuda = bias_gelu_backward_cuda<fast_gelu_back>;