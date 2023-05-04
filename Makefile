all: cpu cunvcc llvmcu opencl

CXX=clang++
CUCC=nvcc
CXX_FLAGS=-O3 -Wall -Wextra -std=c++17
SYCL_FLAGS=-fsycl -fsycl-targets=nvptx64-nvidia-cuda --cuda-path=/opt/cuda
LLVMCU_FLAGS=--cuda-gpu-arch=sm_86 -L/home/pmatos/installs/cuda-11.8/lib64 -lcudart -lrt -ldl -pthread
CU_FLAGS=-O3 -Xcompiler -Wall

.PHONY: all clean run

cpu: src/cpu.cpp
	$(CXX) $(CXX_FLAGS) -o $@ $?

cunvcc: src/cuda.cu
	PATH=/opt/cuda/bin:$(PATH) $(CUCC) $(CU_FLAGS) -o $@ $?

llvmcu: src/cuda.cu
	PATH=/home/pmatos/installs/cuda-11.8/bin:/home/pmatos/dev/llvm-project/build/bin$(PATH) $(CXX) $(LLVMCU_FLAGS) $(CXX_FLAGS) -o $@ $?

opencl: src/opencl.cpp
	$(CXX) $(CXX_FLAGS) -o $@ $? -lOpenCL

sycl: src/sycl.cpp
	PATH=/opt/intel/oneapi/compiler/2023.0.0/linux/bin/:/opt/intel/oneapi/compiler/2023.0.0/linux/bin-llvm/:$(PATH) $(CXX) $(SYCL_FLAGS) $(CXX_FLAGS) -o $@ $?

vulkan: src/vulkan.cpp src/shaders/vulkan.h src/shaders/matrixmul.spv
	$(CXX) $(CXX_FLAGS) -o $@ $< -I/home/pmatos/installs/vulkan/1.3.243.0/x86_64/include -L/home/pmatos/installs/vulkan/1.3.243.0/x86_64/lib -lvulkan

src/shaders/matrixmul.spv: src/shaders/matrixmul.comp src/shaders/vulkan.h
	glslc -fshader-stage=compute -g --target-env=vulkan1.2 $< -o $@

run: all
	./cpu
	./cunvcc
	LD_LIBRARY_PATH=/home/pmatos/installs/cuda-11.8/lib64 ./llvmcu
	./opencl
	./sycl
	./vulkan

clean:
	rm -f cpu cunvcc llvmcu	opencl sycl vulkan src/matrixmul.spv