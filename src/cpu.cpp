#include <chrono>
#include <iostream>
#include <vector>

const int N = 1024; // matrix size

void multiply_matrices(float **A, float **B, float **C) {
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      for (int k = 0; k < N; ++k) {
        C[i][j] += A[i][k] * B[k][j];
      }
    }
  }
}

int main() {
  float **A, **B, **C;

  // allocate memory
  A = (float **)malloc(N * sizeof(float *));
  B = (float **)malloc(N * sizeof(float *));
  C = (float **)malloc(N * sizeof(float *));
  for (int i = 0; i < N; ++i) {
    A[i] = (float *)malloc(N * sizeof(float));
    B[i] = (float *)malloc(N * sizeof(float));
    C[i] = (float *)malloc(N * sizeof(float));
  }

  // initialize matrices
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      A[i][j] = 1;
      B[i][j] = 2;
    }
  }

  // Zero out C
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      C[i][j] = 0;
    }
  }

  // measure time
  auto start = std::chrono::high_resolution_clock::now();
  multiply_matrices(A, B, C);
  auto end = std::chrono::high_resolution_clock::now();

  auto duration =
      std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  std::cout << "Time taken by function: " << duration.count() << " microseconds"
            << std::endl;

  // free memory
  for (int i = 0; i < N; ++i) {
    free(A[i]);
    free(B[i]);
    free(C[i]);
  }
  free(A);
  free(B);
  free(C);

  return 0;
}
