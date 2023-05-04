#include <chrono>
#include <cstddef>
#include <iostream>

#include <sycl/sycl.hpp>

const int N = 1024; // matrix size

void multiply_matrices(sycl::queue &queue, sycl::buffer<float, 2> &A,
                       sycl::buffer<float, 2> &B, sycl::buffer<float, 2> &C) {
  queue
      .submit([&](sycl::handler &cgh) {
        auto a_accessor = A.get_access<sycl::access::mode::read>(cgh);
        auto b_accessor = B.get_access<sycl::access::mode::read>(cgh);
        auto c_accessor = C.get_access<sycl::access::mode::write>(cgh);

        cgh.parallel_for<class MatrixMultiply>(
            sycl::range<2>(N, N), [=](sycl::id<2> idx) {
              size_t i = idx[0];
              size_t j = idx[1];

              float sum = 0;
              for (std::size_t k = 0; k < N; ++k) {
                sum += a_accessor[{i, k}] * b_accessor[{k, j}];
              }
              c_accessor[{i, j}] = sum;
            });
      })
      .wait();
}

int main() {
  float A[N][N], B[N][N], C[N][N];
  sycl::buffer<float, 2> dev_A({N, N}), dev_B({N, N}), dev_C({N, N});

  // initialize matrices
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      A[i][j] = 1;
      B[i][j] = 2;
    }
  }

  // create a SYCL queue
  sycl::queue queue;

  // copy data from host to device
  queue
      .submit([&](sycl::handler &cgh) {
        auto a_accessor = dev_A.get_access<sycl::access::mode::write>(cgh);
        auto b_accessor = dev_B.get_access<sycl::access::mode::write>(cgh);
        cgh.copy(A, a_accessor);
        cgh.copy(B, b_accessor);
      })
      .wait();

  // perform matrix multiplication on device
  auto start = std::chrono::high_resolution_clock::now();
  multiply_matrices(queue, dev_A, dev_B, dev_C);
  auto end = std::chrono::high_resolution_clock::now();

  // copy data from device to host
  queue
      .submit([&](sycl::handler &cgh) {
        auto c_accessor = dev_C.get_access<sycl::access::mode::read>(cgh);
        cgh.copy(c_accessor, C);
      })
      .wait();

  auto duration =
      std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  std::cout << "Time taken by function: " << duration.count() << " microseconds"
            << std::endl;

  return 0;
}