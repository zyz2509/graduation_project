#include <torch/extension.h>
#include "ball_query_gpu.h"

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x "必须是CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x "必须是连续的 tensor")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

at::Tensor ball_query_wrapper_cpp(int b, int n, int m, float radius, int nsample,
                                  at::Tensor new_xyz_tensor, at::Tensor xyz_tensor, at::Tensor idx_tensor) {
    CHECK_INPUT(new_xyz_tensor);
    CHECK_INPUT(xyz_tensor);

    const float *new_xyz = new_xyz_tensor.data_ptr<float>();
    const float *xyz = xyz_tensor.data_ptr<float>();
    int *idx = idx_tensor.data_ptr<int>();

    ball_query_kernel_launcher_cuda(b, n, m, radius, nsample, new_xyz, xyz, idx);

    return idx_tensor;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("ball_query_wrapper_cpp", &ball_query_wrapper_cpp, "Ball Query 包装器（CUDA）");
}
