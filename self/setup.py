from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

if __name__ == '__main__':
    setup(
        name='example',
        ext_modules=[
            CUDAExtension(
                name="cpp_CUDA_code.pointnet_cuda",
                sources=[
                    "cpp_CUDA_code/pointnet_api.cpp",
                    "cpp_CUDA_code/ball_query.cpp",
                    "cpp_CUDA_code/ball_query_gpu.cu",
                ]
            ),
        ],
        cmdclass={
            'build_ext': BuildExtension.with_options(use_ninja=False),
        },
    )
