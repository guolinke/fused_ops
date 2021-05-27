#include <vector>
#include <ATen/ATen.h>
#include <ATen/CUDAGeneratorImpl.h>
#include <ATen/cuda/detail/IndexUtils.cuh>
#include <ATen/cuda/detail/TensorInfo.cuh>
#include <c10/cuda/CUDAMathCompat.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_profiler_api.h>
#include <curand_kernel.h>
#include "THC/THC.h"
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <math.h>
#include "../dropout/dropout.h"

#define CELL(a, b) (((a) + (b) - 1) / (b))
#if __cplusplus >= 201703L
    #define IF_CONSTEXPR constexpr
#else
    #define IF_CONSTEXPR
#endif

template <typename T>
__device__ __forceinline__ T from_uint8(uint8_t input) {
    return (T)input;
}

template <>
__device__ __forceinline__ __nv_bfloat16 from_uint8(uint8_t input) {
    return (__nv_bfloat16)(float)input;
}

template <typename index_t, typename input_t, typename output_t, bool is_training>
__global__ void bias_dropout_add_forward(output_t *dst, const input_t *x, const input_t *bias,
    const input_t *residual, const uint8_t *mask, index_t bsz, int dim, input_t pinv) {
    if IF_CONSTEXPR (is_training) {
        int mask_index = blockIdx.x * CELL(dim, 8);
        uint8_t mask_offset = threadIdx.x % 8;
        for (int j = threadIdx.x; j < dim; j += blockDim.x) {
            if (blockIdx.x < bsz) {
                index_t idx = blockIdx.x * dim + j;
                input_t y = x[idx] + bias[j];
                input_t m = from_uint8<input_t>(((mask[mask_index + j / 8] & (1 << mask_offset)) >> mask_offset));
                dst[idx] = y * m * pinv + residual[idx];
            }
        }
    } else {
        for (int j = threadIdx.x; j < dim; j += blockDim.x) {
            if (blockIdx.x < bsz) {
                index_t idx = blockIdx.x * dim + j;
                dst[idx] = x[idx] + bias[j] + residual[idx];
            }
        }
    }
}

template <typename T>
struct VecTypeImpl;

template <>
struct VecTypeImpl<half> {
    using type = half2;
};


template <>
struct VecTypeImpl<nv_bfloat16> {
    using type = nv_bfloat162;
};

template <>
struct VecTypeImpl<float> {
    using type = float2;
};

template <typename T>
using VecType = typename VecTypeImpl<T>::type;

template <typename index_t, typename input_t, typename output_t, bool is_training>
__global__ void bias_dropout_add_forward_vec(output_t *dst, const input_t *x, const input_t *bias,
    const input_t *residual, const uint8_t *mask, index_t bsz, int dim, input_t pinv) {
    using VecInType = VecType<input_t>;
    using VecOutType = VecType<output_t>;
    if IF_CONSTEXPR (is_training) {
        int mask_index = blockIdx.x * CELL(dim, 8);
        uint8_t mask_offset1 = (threadIdx.x * 2) % 8;
        uint8_t mask_offset2 = (threadIdx.x * 2 + 1) % 8;
        for (int j = threadIdx.x * 2; j < dim; j += blockDim.x * 2) {
            if (blockIdx.x < bsz) {
                index_t idx = blockIdx.x * dim + j;
                VecInType xi = *(VecInType *)(x + idx);
                VecInType b = *(VecInType *)(bias + j);
                VecInType r = *(VecInType *)(residual + idx);
                uint8_t m = mask[mask_index + j / 8];
                input_t m1 = from_uint8<input_t>(((m & (1 << mask_offset1)) >> mask_offset1));
                input_t m2 = from_uint8<input_t>(((m & (1 << mask_offset2)) >> mask_offset2));
                VecOutType d;
                d.x = (xi.x + b.x) * m1 * pinv + r.x;
                d.y = (xi.y + b.y) * m2 * pinv + r.y;
                *(VecOutType *)(dst + idx) = d;
            }
        }
    } else {
        for (int j = threadIdx.x * 2; j < dim; j += blockDim.x * 2) {
            if (blockIdx.x < bsz) {
                index_t idx = blockIdx.x * dim + j;
                VecInType xi = *(VecInType *)(x + idx);
                VecInType b = *(VecInType *)(bias + j);
                VecInType r = *(VecInType *)(residual + idx);
                VecOutType d;
                d.x = xi.x + b.x + r.x;
                d.y = xi.y + b.y + r.y;
                *(VecOutType *)(dst + idx) = d;
            }
        }
    }
}

template <typename index_t, typename input_t, typename output_t>
__global__ void bias_dropout_add_backward(output_t *dst, const input_t *grad, const uint8_t *mask, index_t bsz, int dim) {
    int mask_index = blockIdx.x * CELL(dim, 8);
    uint8_t mask_offset = threadIdx.x % 8;
    for (int j = threadIdx.x; j < dim; j += blockDim.x) {
        if (blockIdx.x < bsz) {
            index_t idx = blockIdx.x * dim + j;
            uint8_t m = (mask[mask_index + j / 8] & (1 << mask_offset)) >> mask_offset;
            dst[idx] = grad[idx] * from_uint8<input_t>(m);
        }
    }
}

template <typename index_t, typename input_t, typename output_t>
__global__ void bias_dropout_add_backward_vec(output_t *dst, const input_t *grad, const uint8_t *mask, index_t bsz, int dim) {
    using VecInType = VecType<input_t>;
    using VecOutType = VecType<output_t>;
    int mask_index = blockIdx.x * CELL(dim, 8);
    uint8_t mask_offset1 = (threadIdx.x * 2) % 8;
    uint8_t mask_offset2 = (threadIdx.x * 2 + 1) % 8;
    for (int j = threadIdx.x * 2; j < dim; j += blockDim.x * 2) {
        if (blockIdx.x < bsz) {
            index_t idx = blockIdx.x * dim + j;
            uint8_t m = mask[mask_index + j / 8];
            VecInType g = *(VecInType *)(grad + idx);
            VecOutType d;
            d.x = g.x * from_uint8<input_t>(((m & (1 << mask_offset1)) >> mask_offset1));
            d.y = g.y * from_uint8<input_t>(((m & (1 << mask_offset2)) >> mask_offset2));
            *(VecOutType *)(dst + idx) = d;
        }
    }
}

std::vector<c10::optional<torch::Tensor>> bias_dropout_add_forward_cuda(const torch::Tensor &x, const torch::Tensor &bias,
    const torch::Tensor &residual, bool is_training, float dropout_prob, c10::optional<at::Generator> gen_) {
    using MaskType = uint64_t;
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
    int ThreadsPerBlockVec = CELL(dim, 256) * 256 % 512 == 0 ? 256 : 128;
    if (is_training && dropout_prob != 0.0) {
        auto mask_options = dst_options.dtype(torch::kInt64);
        torch::Tensor mask = torch::empty(bsz * CELL(dim, sizeof(MaskType) * 8), mask_options);
        auto gen = at::get_generator_or_default<at::CUDAGeneratorImpl>(gen_, at::cuda::detail::getDefaultCUDAGenerator());
        std::pair<uint64_t, uint64_t> rng_engine_inputs;
        {
            // See Note [Acquire lock when using random generators]
            std::lock_guard<std::mutex> lock(gen->mutex_);
            rng_engine_inputs = gen->philox_engine_inputs(sizeof(MaskType) * 8);
        }
        uint64_t seed = std::get<0>(rng_engine_inputs);
        uint64_t offset = std::get<1>(rng_engine_inputs);
        generate_dropout_mask<MaskType>((MaskType *)mask.data_ptr(), bsz, dim, 1.0 - dropout_prob, seed, offset);
        if (type == at::ScalarType::BFloat16) {
            if (dim % 2 == 0) {
                bias_dropout_add_forward_vec<size_t, nv_bfloat16, nv_bfloat16, true><<<bsz, ThreadsPerBlockVec, 0, stream>>>(
                    (nv_bfloat16 *)results.data_ptr(),
                    (const nv_bfloat16 *)x.data_ptr(),
                    (const nv_bfloat16 *)bias.data_ptr(),
                    (const nv_bfloat16 *)residual.data_ptr(),
                    (const uint8_t *)mask.data_ptr(),
                    bsz,
                    dim,
                    1.0 / (1.0 - dropout_prob));
            } else {
                bias_dropout_add_forward<size_t, nv_bfloat16, nv_bfloat16, true><<<bsz, ThreadsPerBlock, 0, stream>>>(
                    (nv_bfloat16 *)results.data_ptr(),
                    (const nv_bfloat16 *)x.data_ptr(),
                    (const nv_bfloat16 *)bias.data_ptr(),
                    (const nv_bfloat16 *)residual.data_ptr(),
                    (const uint8_t *)mask.data_ptr(),
                    bsz,
                    dim,
                    1.0 / (1.0 - dropout_prob));
            }
        } else if (type == at::ScalarType::Half) {
            if (dim % 2 == 0) {
                bias_dropout_add_forward_vec<size_t, half, half, true><<<bsz, ThreadsPerBlockVec, 0, stream>>>(
                    (half *)results.data_ptr(),
                    (const half *)x.data_ptr(),
                    (const half *)bias.data_ptr(),
                    (const half *)residual.data_ptr(),
                    (const uint8_t *)mask.data_ptr(),
                    bsz,
                    dim,
                    1.0 / (1.0 - dropout_prob));
            } else {
                bias_dropout_add_forward<size_t, half, half, true><<<bsz, ThreadsPerBlock, 0, stream>>>(
                    (half *)results.data_ptr(),
                    (const half *)x.data_ptr(),
                    (const half *)bias.data_ptr(),
                    (const half *)residual.data_ptr(),
                    (const uint8_t *)mask.data_ptr(),
                    bsz,
                    dim,
                    1.0 / (1.0 - dropout_prob));
            }
        } else if (type == at::ScalarType::Float) {
            bias_dropout_add_forward<size_t, float, float, true><<<bsz, ThreadsPerBlock, 0, stream>>>(
                (float *)results.data_ptr(),
                (const float *)x.data_ptr(),
                (const float *)bias.data_ptr(),
                (const float *)residual.data_ptr(),
                (const uint8_t *)mask.data_ptr(),
                bsz,
                dim,
                1.0 / (1.0 - dropout_prob));
        }
        return {results, mask};
    } else {
        if (type == at::ScalarType::BFloat16) {
            if (dim % 2 == 0) {
                bias_dropout_add_forward_vec<size_t, nv_bfloat16, nv_bfloat16, false><<<bsz, ThreadsPerBlockVec, 0, stream>>>(
                    (nv_bfloat16 *)results.data_ptr(),
                    (const nv_bfloat16 *)x.data_ptr(),
                    (const nv_bfloat16 *)bias.data_ptr(),
                    (const nv_bfloat16 *)residual.data_ptr(),
                    nullptr,
                    bsz,
                    dim,
                    0.0);
            } else {
                bias_dropout_add_forward<size_t, nv_bfloat16, nv_bfloat16, false><<<bsz, ThreadsPerBlock, 0, stream>>>(
                    (nv_bfloat16 *)results.data_ptr(),
                    (const nv_bfloat16 *)x.data_ptr(),
                    (const nv_bfloat16 *)bias.data_ptr(),
                    (const nv_bfloat16 *)residual.data_ptr(),
                    nullptr,
                    bsz,
                    dim,
                    0.0);
            }
        } else if (type == at::ScalarType::Half) {
            if (dim % 2 == 0) {
                bias_dropout_add_forward_vec<size_t, half, half, false><<<bsz, ThreadsPerBlockVec, 0, stream>>>(
                    (half *)results.data_ptr(),
                    (const half *)x.data_ptr(),
                    (const half *)bias.data_ptr(),
                    (const half *)residual.data_ptr(),
                    nullptr,
                    bsz,
                    dim,
                    0.0);
            } else {
                bias_dropout_add_forward<size_t, half, half, false><<<bsz, ThreadsPerBlock, 0, stream>>>(
                    (half *)results.data_ptr(),
                    (const half *)x.data_ptr(),
                    (const half *)bias.data_ptr(),
                    (const half *)residual.data_ptr(),
                    nullptr,
                    bsz,
                    dim,
                    0.0);
            }
        } else if (type == at::ScalarType::Float) {
            bias_dropout_add_forward<size_t, float, float, false><<<bsz, ThreadsPerBlock, 0, stream>>>(
                (float *)results.data_ptr(),
                (const float *)x.data_ptr(),
                (const float *)bias.data_ptr(),
                (const float *)residual.data_ptr(),
                nullptr,
                bsz,
                dim,
                0.0);
        }
        return {results, c10::optional<torch::Tensor>()};
    }
}

torch::Tensor bias_dropout_add_backward_cuda(const torch::Tensor &grad, const torch::Tensor &mask) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    auto sizes = grad.sizes();
    size_t bsz = 1;
    for (size_t i = 0; i + 1 < sizes.size(); ++i) {
        bsz *= sizes[i];
    }
    int dim = sizes[sizes.size() - 1];
    auto dst_options = grad.options().requires_grad(false);
    torch::Tensor results = torch::empty(sizes, dst_options);
    auto type = grad.scalar_type();
    const int ThreadsPerBlock = 256;
    int ThreadsPerBlockVec = CELL(dim, 256) * 256 % 512 == 0 ? 256 : 128;
    if (type == at::ScalarType::BFloat16) {
        if (dim % 2 == 0) {
            bias_dropout_add_backward_vec<size_t, nv_bfloat16, nv_bfloat16><<<bsz, ThreadsPerBlockVec, 0, stream>>>(
                (nv_bfloat16 *)results.data_ptr(),
                (const nv_bfloat16 *)grad.data_ptr(),
                (const uint8_t *)mask.data_ptr(),
                bsz,
                dim);
        } else {
            bias_dropout_add_backward<size_t, nv_bfloat16, nv_bfloat16><<<bsz, ThreadsPerBlock, 0, stream>>>(
                (nv_bfloat16 *)results.data_ptr(),
                (const nv_bfloat16 *)grad.data_ptr(),
                (const uint8_t *)mask.data_ptr(),
                bsz,
                dim);
        }
    } else if (type == at::ScalarType::Half) {
        if (dim % 2 == 0) {
            bias_dropout_add_backward_vec<size_t, half, half><<<bsz, ThreadsPerBlockVec, 0, stream>>>(
                (half *)results.data_ptr(),
                (const half *)grad.data_ptr(),
                (const uint8_t *)mask.data_ptr(),
                bsz,
                dim);
        } else {
            bias_dropout_add_backward<size_t, half, half><<<bsz, ThreadsPerBlock, 0, stream>>>(
                (half *)results.data_ptr(),
                (const half *)grad.data_ptr(),
                (const uint8_t *)mask.data_ptr(),
                bsz,
                dim);
        }
    } else if (type == at::ScalarType::Float) {
        bias_dropout_add_backward<size_t, float, float><<<bsz, ThreadsPerBlock, 0, stream>>>(
            (float *)results.data_ptr(),
            (const float *)grad.data_ptr(),
            (const uint8_t *)mask.data_ptr(),
            bsz,
            dim);
    }
    return results;
}