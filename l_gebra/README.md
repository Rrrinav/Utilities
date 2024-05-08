# Linear Algebra Library

This is a C++ library for performing basic linear algebra operations including matrix and vector arithmetic, transformations, and decompositions. It provides a `Matrix` class for matrices and a `Vec` class for vectors.

## Features

- Matrix and vector arithmetic operations
- Matrix transformations: transpose, inverse, power, determinant, trace, etc.
- Vector operations: dot product, cross product, magnitude, normalization, etc.
- QR decomposition
- Gaussian elimination for solving linear systems
- Eigenvalue and eigenvector calculations
- Utility functions to create identity, zero, ones, and random matrices/vectors
- Interpolation functions (lerp) for matrices and vectors

## Usage

### Matrix Class

```cpp
#include "Matrix.h"

// Create a 3x3 matrix initialized with zeros
utl::Matrix<float> mat(3, 3);

// Access elements using () operator
mat(0, 0) = 1.0f;
mat(1, 1) = 2.0f;
mat(2, 2) = 3.0f;

// Perform operations
utl::Matrix<float> transposed = mat.transpose();
double det = mat.determinant();
// More operations...

// Solve linear system
utl::Matrix<float> b(3, {{1.0}, {2.0}, {3.0}});
utl::Matrix<float> solution = mat.solve_linear_system(b);
utl::Matrix<float> id = Matrix<float>::identity_matrix(10);
```

## Dependencies

C++ standard library

## Installation

Simply include the "l_gebra.hpp" header files in your project and put #define "L_GEBRA_IMPLEMENTATION" macro before it.

## Inspirations
[qbLinAlg](https://github.com/QuantitativeBytes/qbLinAlg)
[la](https://github.com/tsoding/la)
[stb](https://github.com/nothings/stb)

## License

This library is released under the [MIT License](https://opensource.org/license/mit).
