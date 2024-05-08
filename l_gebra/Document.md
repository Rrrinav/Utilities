# Linear Algebra Library Documentation

This document provides detailed documentation for the `Matrix` and `Vec` classes in the linear algebra library.

## Matrix Class

### Constructors

- `Matrix()`: Default constructor.
- `Matrix(size_t rows, size_t cols)`: Constructor to create a matrix of size rows x cols.
- `Matrix(std::initializer_list<std::initializer_list<T>> initList)`: Constructor to create a matrix from an initializer list of initializer lists.
- `Matrix(std::initializer_list<T> init_list)`: Constructor to create a row matrix of size rows x 1 and initialize with initList.
- `Matrix(std::vector<T> &&other)`: Move constructor.
- `Matrix(size_t rows, size_t cols, T val)`: Constructor to create a matrix of size rows x cols and initialize all elements with val.
- `Matrix(size_t rows, std::initializer_list<T> initList)`: Constructor to create a column matrix of size rows x 1 and initialize with initList.

### Operators

- `operator()`: Access the element at the specified position in the matrix.
- `operator==`: Equality comparison.
- `operator+`, `-`, `*`, `/`: Arithmetic operations with matrices and scalars.

### Matrix Operations

- `transpose()`: Transpose the matrix.
- `determinant()`: Calculate the determinant of the matrix.
- `cofactor(size_t row, size_t col)`: Calculate the cofactor of a specified element.
- `minor(size_t row, size_t col)`: Get the minor matrix by deleting a specified row and column.
- `inverse()`: Get the inverse matrix.
- `power(int n)`: Raise the matrix to a power.
- `rank()`: Get the rank of the matrix.
- `trace()`: Get the trace of the matrix.
- `norm()`: Calculate the Frobenius norm of the matrix.

### Row Operations

- `swap_rows(size_t r1, size_t r2)`: Swap two rows.
- `multiply_row(size_t r, float scalar)`: Multiply a row by a scalar.
- `add_scaled_row(size_t r1, size_t r2, float scalar)`: Add a scaled row to another.

### Complex Operations

- `eigenvalues()`: Get eigenvalues using QR algorithm.
- `eigenvalues_power_iteration()`: Get dominant eigenvalue using power iteration.
- `eigenvectors()`: Get eigenvectors using QR algorithm.
- `qr_decomposition()`: Get QR decomposition.
- `augment(const Matrix<T> &other)`: Augment two matrices.
- `gauss_elimination()`: Convert the matrix to echelon form.
- `has_solution(const Matrix<T> &reduced_matrix)`: Determine if a matrix has a solution.
- `solve_linear_system(const Matrix<T> &b)`: Solve a linear system of equations.

## Vec Class

### Constructors

- `Vec()`: Default constructor.
- `Vec(std::initializer_list<T> init_list)`: Constructor with initializing list.
- `Vec(const Matrix<T> &matrix)`: Constructor with a column matrix.
- `Vec(const Vec<Y, _size> &other)`: Copy constructor.
- `Vec(Vec<Y, _size> &&other)`: Move constructor.

### Vector Functions

- `print()`: Print the vector.

### Operators

- `operator[]`: Access elements of the vector.
- `operator*`, `+`, `-`, `/`: Arithmetic operations with vectors and scalars.
- `operator*(const Matrix<Y> &m)`: Multiply a vector with a matrix.

### Vector Operations

- `dot(const Vec<Y, n_x> &x)`: Calculate the dot product.
- `cross(const Vec<Y, n_x> &x)`: Calculate the cross product.
- `squared_magnitude()`: Calculate the squared magnitude.
- `magnitude()`: Calculate the magnitude.
- `power(float x)`: Raise elements to a power.
- `normalize()`: Normalize the vector.
- `get_normalized_vector()`: Get the normalized vector.
- `angle(const Vec<Y, n_x> &x)`: Calculate the angle between vectors.
- `distance(const Vec<Y, n_x> &x)`: Calculate the distance between vectors.
- `projection_onto(const Vec<Y, n_x> &x)`: Calculate the projection onto another vector.
- `rotate(float angle, char8_t axis)`: Rotate the vector by an angle.

### Utility Functions

- `zero_vector()`, `ones_vector()`, `random_vector(T min, T max)`: Get vectors with specified parameters.
- `min(const Vec<T, _size> &v1, const Vec<T, _size> &v2)`, `max(const Vec<T, _size> &v1, const Vec<T, _size> &v2)`: Get the minimum or maximum of two vectors.
- `lerp(const Vec<T, _size> &v1, const Vec<T, _size> &v2, float t)`: Linear interpolation between two vectors.
