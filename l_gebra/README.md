# L_Gebra  

**A simple header-only linear algebra library**

L_Gebra is a lightweight linear algebra library implemented in C++. It provides a simple interface for working with vectors of any size and performing basic arithmetic operations on them.

## Features

- Header-only: No need to compile or link, just include the header file in your project.
- Support for vectors of any size, with element-wise arithmetic operations.
- Scalar multiplication and addition.
- Easy to use and integrate into your projects.

## Usage

To use L_Gebra in your project, simply include the `lgebra.hpp` header file after macro L_GEBRA_IMPLEMENTATION. 

You will also have to use namespace **utl**.

Example usage:

```cpp
#define L_GEBRA_IMPLEMENTATION
#include "lgebra.hpp"
#include <iostream>

int main() {
  // Create a vector with initial values
  utl::vec<double> v = {1.0, 2.0, 3.0};

  // Multiply by scalar
  v = v.multiply(2.0);
  std::cout << "After multiplying by 2: ";
  v.print(); // Output: [ 2 4 6 ]

  // Add scalar
  v = v.add(1.0);
  std::cout << "After adding 1: ";
  v.print(); // Output: [ 3 5 7 ]

  // Multiply by 0.5 and add 1
  v = v.multiply(0.5).add(1.0);
  std::cout << "After multiplying by 0.5 and adding 1: ";
  v.print(); // Output: [ 2 3.5 5 ]

  return 0;
}
```
## License

This library is licensed under the MIT License. See the LICENSE file for details

## Warning

This isn't fully functional version, it is yet in devlopment.