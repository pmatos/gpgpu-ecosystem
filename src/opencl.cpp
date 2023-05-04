#include <CL/cl.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <iostream>

#define N 1024
#define STR(x) #x
#define EXP(x) STR(x)

void explainOpenCLError(cl_int err) {
  switch (err) {
  case CL_SUCCESS:
    printf("Success!\n");
    break;
  case CL_DEVICE_NOT_FOUND:
    printf("Error: Device not found.\n");
    break;
  case CL_DEVICE_NOT_AVAILABLE:
    printf("Error: Device not available\n");
    break;
  case CL_COMPILER_NOT_AVAILABLE:
    printf("Error: Compiler not available\n");
    break;
  case CL_MEM_OBJECT_ALLOCATION_FAILURE:
    printf("Error: Memory object allocation failure\n");
    break;
  case CL_OUT_OF_RESOURCES:
    printf("Error: Out of resources\n");
    break;
  case CL_OUT_OF_HOST_MEMORY:
    printf("Error: Out of host memory\n");
    break;
  case CL_PROFILING_INFO_NOT_AVAILABLE:
    printf("Error: Profiling information not available\n");
    break;
  case CL_MEM_COPY_OVERLAP:
    printf("Error: Memory copy overlap\n");
    break;
  case CL_IMAGE_FORMAT_MISMATCH:
    printf("Error: Image format mismatch\n");
    break;
  case CL_IMAGE_FORMAT_NOT_SUPPORTED:
    printf("Error: Image format not supported\n");
    break;
  case CL_BUILD_PROGRAM_FAILURE:
    printf("Error: Program build failure\n");
    break;
  case CL_MAP_FAILURE:
    printf("Error: Map failure\n");
    break;
  case CL_INVALID_VALUE:
    printf("Error: Invalid value\n");
    break;
  case CL_INVALID_DEVICE_TYPE:
    printf("Error: Invalid device type\n");
    break;
  case CL_INVALID_PLATFORM:
    printf("Error: Invalid platform\n");
    break;
  case CL_INVALID_DEVICE:
    printf("Error: Invalid device\n");
    break;
  case CL_INVALID_CONTEXT:
    printf("Error: Invalid context\n");
    break;
  case CL_INVALID_QUEUE_PROPERTIES:
    printf("Error: Invalid queue properties\n");
    break;
  case CL_INVALID_COMMAND_QUEUE:
    printf("Error: Invalid command queue\n");
    break;
  case CL_INVALID_HOST_PTR:
    printf("Error: Invalid host pointer\n");
    break;
  case CL_INVALID_MEM_OBJECT:
    printf("Error: Invalid memory object\n");
    break;
  case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:
    printf("Error: Invalid image format descriptor\n");
    break;
  case CL_INVALID_IMAGE_SIZE:
    printf("Error: Invalid image size\n");
    break;
  case CL_INVALID_SAMPLER:
    printf("Error: Invalid sampler\n");
    break;
  case CL_INVALID_BINARY:
    printf("Error: Invalid binary\n");
    break;
  case CL_INVALID_BUILD_OPTIONS:
    printf("Error: Invalid build options\n");
    break;
  case CL_INVALID_PROGRAM:
    printf("Error: Invalid program\n");
    break;
  case CL_INVALID_PROGRAM_EXECUTABLE:
    printf("Error: Invalid program executable\n");
    break;
  case CL_INVALID_KERNEL_NAME:
    printf("Error: Invalid kernel name\n");
    break;
  case CL_INVALID_KERNEL_DEFINITION:
    printf("Error: Invalid kernel definition\n");
    break;
  case CL_INVALID_KERNEL:
    printf("Error: Invalid kernel\n");
    break;
  case CL_INVALID_ARG_INDEX:
    printf("Error: Invalid argument index\n");
    break;
  case CL_INVALID_ARG_VALUE:
    printf("Error: Invalid argument value\n");
    break;
  case CL_INVALID_ARG_SIZE:
    printf("Error: Invalid argument size\n");
    break;
  case CL_INVALID_KERNEL_ARGS:
    printf("Error: Invalid kernel arguments\n");
    break;
  case CL_INVALID_WORK_DIMENSION:
    printf("Error: Invalid work dimension\n");
    break;
  case CL_INVALID_WORK_GROUP_SIZE:
    printf("Error: Invalid work group size\n");
    break;
  case CL_INVALID_WORK_ITEM_SIZE:
    printf("Error: Invalid work item size\n");
    break;
  case CL_INVALID_GLOBAL_OFFSET:
    printf("Error: Invalid global offset\n");
    break;
  case CL_INVALID_EVENT_WAIT_LIST:
    printf("Error: Invalid event wait list\n");
    break;
  case CL_INVALID_EVENT:
    printf("Error: Invalid event\n");
    break;
  case CL_INVALID_OPERATION:
    printf("Error: Invalid operation\n");
    break;
  case CL_INVALID_GL_OBJECT:
    printf("Error: Invalid OpenGL object\n");
    break;
  case CL_INVALID_BUFFER_SIZE:
    printf("Error: Invalid buffer size\n");
    break;
  case CL_INVALID_MIP_LEVEL:
    printf("Error: Invalid mip-map level\n");
    break;
  default:
    printf("Error: Unknown\n");
  }
}

int main() {
  float *A = (float *)malloc(N * N * sizeof(float));
  float *B = (float *)malloc(N * N * sizeof(float));
  float *C = (float *)malloc(N * N * sizeof(float));

  for (int i = 0; i < N * N; i++) {
    A[i] = 1.0;
    B[i] = 2.0;
    C[i] = 0.0;
  }

  cl_int err;

  // Create OpenCL platform
  cl_platform_id platform;
  err = clGetPlatformIDs(1, &platform, NULL);

  if (err != CL_SUCCESS) {
    printf("Error: Failed to find an OpenCL platform!\n");
    return 1;
  }

  // Create OpenCL device
  cl_device_id device;
  err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);

  if (err != CL_SUCCESS) {
    printf("Error: Failed to find an OpenCL device!\n");
    return 1;
  }

  // Create OpenCL context
  cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);

  if (err != CL_SUCCESS) {
    printf("Error: Failed to create an OpenCL context!\n");
    return 1;
  }

  // Create OpenCL command queue
  cl_command_queue queue =
      clCreateCommandQueueWithProperties(context, device, 0, &err);

  if (err != CL_SUCCESS) {
    printf("Error: Failed to create an OpenCL command queue!\n");
    return 1;
  }

  // Create OpenCL memory buffers for A, B, and C matrices
  cl_mem bufferA = clCreateBuffer(context, CL_MEM_READ_ONLY,
                                  N * N * sizeof(float), NULL, &err);
  cl_mem bufferB = clCreateBuffer(context, CL_MEM_READ_ONLY,
                                  N * N * sizeof(float), NULL, &err);
  cl_mem bufferC = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                  N * N * sizeof(float), NULL, &err);

  if (err != CL_SUCCESS) {
    printf("Error: Failed to create OpenCL memory buffers!\n");
    return 1;
  }

  // Copy matrices A and B to the device
  err = clEnqueueWriteBuffer(queue, bufferA, CL_TRUE, 0, N * N * sizeof(float),
                             A, 0, NULL, NULL);
  err |= clEnqueueWriteBuffer(queue, bufferB, CL_TRUE, 0, N * N * sizeof(float),
                              B, 0, NULL, NULL);

  if (err != CL_SUCCESS) {
    printf("Error: Failed to copy matrices A and B to the device!\n");
    return 1;
  }

  // Build OpenCL kernel program
  const char *source = R"CLC(
        __kernel void matrixMul(__global float* A, __global float* B, __global float* C) {
            int i = get_global_id(0);
            int j = get_global_id(1);
            if (i < N && j < N) {
                float sum = 0.0;
                for (int k = 0; k < N; k++) {
                    sum += A[j * N + k] * B[k * N + i];
                }
                C[j * N + i] = sum;
            }
        }
    )CLC";

  auto start = std::chrono::high_resolution_clock::now();
  cl_program program =
      clCreateProgramWithSource(context, 1, &source, NULL, &err);

  if (err != CL_SUCCESS) {
    printf("Error: Failed to create OpenCL program with source!\n");
    return 1;
  }

  err = clBuildProgram(program, 1, &device, "-DN=" EXP(N), NULL, NULL);

  if (err != CL_SUCCESS) {
    printf("Error: Failed to build OpenCL program!\n");
    printf("Error code: %d\n", err);
    explainOpenCLError(err);
    // get the build log
    char buffer[4096];
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(buffer),
                          buffer, NULL);
    printf("Build log: %s\n", buffer);
    return 1;
  }

  // Create OpenCL kernel
  cl_kernel kernel = clCreateKernel(program, "matrixMul", &err);

  if (err != CL_SUCCESS) {
    printf("Error: Failed to create OpenCL kernel!\n");
    return 1;
  }

  // Set OpenCL kernel arguments
  err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufferA);
  err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufferB);
  err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufferC);

  if (err != CL_SUCCESS) {
    printf("Error: Failed to set OpenCL kernel arguments!\n");
    return 1;
  }

  // Execute the OpenCL kernel
  size_t globalWorkSize[2] = {N, N};
  size_t localWorkSize[2] = {16, 16};

  err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, globalWorkSize,
                               localWorkSize, 0, NULL, NULL);

  if (err != CL_SUCCESS) {
    printf("Error: Failed to execute OpenCL kernel!\n");
    return 1;
  }

  // Copy the result C matrix from the device
  err = clEnqueueReadBuffer(queue, bufferC, CL_TRUE, 0, sizeof(float) * N * N,
                            C, 0, NULL, NULL);

  if (err != CL_SUCCESS) {
    printf("Error: Failed to copy the result C matrix from the device!\n");
    return 1;
  }

  auto end = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  std::cout << "Time taken by function: " << duration.count() << " microseconds"
            << std::endl;

  // Clean up OpenCL resources
  clReleaseKernel(kernel);
  clReleaseProgram(program);
  clReleaseCommandQueue(queue);
  clReleaseMemObject(bufferA);
  clReleaseMemObject(bufferB);
  clReleaseMemObject(bufferC);
  clReleaseContext(context);

  // Clean up host resources
  free(A);
  free(B);
  free(C);

  return 0;
}
