# GPGPU Ecosystem

This is a guide to the GPGPU mostly surrounding LLVM that I have created as I am learning the area.

Feel free to suggest improvements, re-use and correct mistakes.

# The Algorithm

As an example algorithm, we'll use Matrix Multiplication. This is a simple algorithm that is easy to implement and is a good example of a parallelizable algorithm.
The alternative I initially thought about was Gaussian Blur but it's harder to test and implement due to the loading/saving of PNG. Even with the use of libpng, it's just more code we don't need to deal with.

# Program Guide

    1. `src/cpu.cpp` - CPU implementation of the algorithm