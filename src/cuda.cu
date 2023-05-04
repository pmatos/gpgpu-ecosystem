#include <chrono>
#include <iostream>

#include <cuda_runtime.h>

const int N = 1024; // matrix size

__global__ void multiply_matrices(float *A, float *B, float *C) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  if (i < N && j < N) {
    float sum = 0;
    for (int k = 0; k < N; ++k) {
      sum += A[i * N + k] * B[k * N + j];
    }
    C[i * N + j] = sum;
  }
}

int main() {
  float *A, *B, *C;

  // allocate shared memory between device and host
  cudaMallocManaged(&A, N * N * sizeof(float));
  cudaMallocManaged(&B, N * N * sizeof(float));
  cudaMallocManaged(&C, N * N * sizeof(float));

  // initialize matrices
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      A[i * N + j] = 1;
      B[i * N + j] = 2;
    }
  }

  // perform matrix multiplication on device
  // cudaDeviceProp prop;
  // cudaGetDeviceProperties(&prop, 0);
  // int max_threads_per_block = prop.maxThreadsPerBlock; // FIXME unused
  dim3 threadsPerBlock(16, 16);
  dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                 (N + threadsPerBlock.y - 1) / threadsPerBlock.y);
  auto start = std::chrono::high_resolution_clock::now();
  multiply_matrices<<<numBlocks, threadsPerBlock>>>(A, B, C);
  cudaDeviceSynchronize();
  auto end = std::chrono::high_resolution_clock::now();

  // free memory on device
  cudaFree(A);
  cudaFree(B);
  cudaFree(C);

  auto duration =
      std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  std::cout << "Time taken by function: " << duration.count() << " microseconds"
            << std::endl;

  return 0;
}