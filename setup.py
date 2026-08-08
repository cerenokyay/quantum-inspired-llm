import os
import sys
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

compile_args = ['-O3', '-std=c++17']
link_args = []

if sys.platform == 'darwin':
    print("🍏 Apple Silicon (M-Series) Algılandı. Apple Accelerate Framework bağlanıyor...")
    compile_args += ['-D__APPLE_MPS__']
    link_args += ['-framework', 'Accelerate']

setup(
    name='qillm_cpp',
    version='2.0.0',
    ext_modules=[
        CppExtension(
            name='qillm_cpp',
            sources=['csrc/tensor_network.cpp'],
            extra_compile_args=compile_args,
            extra_link_args=link_args
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)