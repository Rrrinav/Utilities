/***
 ██╗              ██████╗ ███████╗██████╗ ██████╗  █████╗
 ██║             ██╔════╝ ██╔════╝██╔══██╗██╔══██╗██╔══██╗
 ██║             ██║  ███╗█████╗  ██████╔╝██████╔╝███████║
 ██║             ██║   ██║██╔══╝  ██╔══██╗██╔══██╗██╔══██║
 ███████╗███████╗╚██████╔╝███████╗██████╔╝██║  ██║██║  ██║
 ╚══════╝╚══════╝ ╚═════╝ ╚══════╝╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝
 */

/*
 * MIT License
 * Copyright (c) 2024 Rinav
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

/*
----------------------------------HOW TO USE?-----------------------------------------
| You just need to include this header file in your project and use the vec          |
| class with namespace utl. You also need macro L_GEBRA_IMPLEMENTATION before        |
| "#include "l_gebra"" in one of your source files to include the                    |
| implementation. Basically :- #define L_GEBRA_IMPLEMENTATION #include "l_gebra.hpp" |
--------------------------------------------------------------------------------------
*/

// TODO:  min, max, lerp, ceil, clamp, Solving Linear Systems

#pragma once

#include <cwchar>
#include <uchar.h>

#include <cmath>
#include <cstddef>
#include <initializer_list>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

#define IL inline
#define S static
#define V virtual

namespace utl
{

    enum class sol_type
    {
        Unique,
        Infinite,
        NoSolution
    };

    // Matrix class
    template <typename T>
    class Matrix
    {
    private:
        Matrix(size_t rows, size_t cols, std::initializer_list<T> initList) : _rows(rows), _cols(cols), data(initList)
        {
        }

    protected:
        size_t _rows;
        size_t _cols;
        std::vector<T> data;

    public:
        IL Matrix() : _rows(0), _cols(0) {}

        IL Matrix(size_t rows, size_t cols) : _rows(rows), _cols(cols), data(rows * cols, 0) {}

        IL Matrix(std::initializer_list<std::initializer_list<T>> initList)
            : _rows(initList.size()), _cols(initList.size() > 0 ? initList.begin() -> size() : 0)
        {
            size_t size = 0;
            size_t rowSize = _cols;  // Size of the first row
            for (const auto &row : initList)
            {
                size += row.size();
                if (row.size() != rowSize)
                {
                    throw std::invalid_argument("Initializer list rows have different sizes");
                }
            }
            data.reserve(size);
            for (const auto &row : initList)
            {
                for (const auto &val : row)
                {
                    data.push_back(val);
                }
            }
        }

        IL Matrix(std::initializer_list<T> init_list) : _rows(1), _cols(init_list.size()), data(init_list) {}

        IL Matrix(std::vector<T> &&other) noexcept
            : _rows(data.size()), _cols(data.empty() ? 0 : data.size() / _rows), data(std::move(other))
        {
        }
        Matrix(size_t rows, std::initializer_list<T> initList) : Matrix(rows, 1, initList) {}
        IL ~Matrix() = default;

        IL size_t size() const { return data.size(); }

        IL size_t rows() const { return _rows; }

        IL size_t cols() const { return _cols; }

        IL T &operator()(size_t row, size_t col)
        {
            if (row >= _rows || col >= _cols)
            {
                throw std::out_of_range("Index out of range");
            }
            return data[row * _cols + col];
        }

        IL const T &operator()(size_t row, size_t col) const
        {
            if (row >= rows() || col >= cols())
            {
                throw std::out_of_range("Index out of range");
            }
            return data[row * _cols + col];
        }
        IL void print();
        template <typename Y>
        IL Matrix<T> operator+(const Matrix<Y> &other) const;
        template <typename Y>
        IL Matrix<T> operator-(const Matrix<Y> &other) const;
        IL Matrix<T> operator*(const T &scalar) const;
        template <typename Y>
        IL Matrix<T> operator/(const Matrix<Y> &other) const;
        template <typename Y>
        IL Matrix<T> operator*(const Matrix<Y> &other) const;
        IL void sin();
        IL void cos();
        IL Matrix<T> transpose() const;
        double determinant() const;
        double cofactor(size_t row, size_t col) const;
        Matrix<T> minor(size_t row, size_t col) const;
        Matrix<T> inverse(const Matrix<T> &m);
        Matrix<T> power(int n);
        Matrix<T> row_reduce();
        size_t rank() const;
        T trace() const;
        IL T norm() const;
        IL void swap_rows(size_t r1, size_t r2);
        IL void multiply_row(size_t r, float scalar);
        IL void add_scaled_row(size_t r1, size_t r2, float scalar);
        S std::vector<T> eigenvalues(const Matrix<T> &m);
        S std::vector<std::vector<T>> eigenvectors(const Matrix<T> &m);
        std::pair<Matrix<T>, Matrix<T>> lu_decomposition();
        std::pair<Matrix<T>, Matrix<T>> qr_decomposition();
        Matrix<T> augment(const Matrix<T> &other) const;
        Matrix<T> gauss_elimination() const;
        Matrix<T> row_echelon()
        {
            if (_cols < _rows)
            {
                throw std::invalid_argument("Matrix must have more columns than rows");
            }

            int c_row, c_col;
            int max_count = 100;
        }
        S sol_type has_solution(const Matrix<T> &reduced_matrix);
        Matrix<T> solve_linear_system(const Matrix<T> &b) const;
        S IL Matrix<T> identity_matrix(size_t n);
        S IL Matrix<T> zero_matrix(size_t n);
        S IL Matrix<T> ones_matrix(size_t n);
        S IL Matrix<T> random_matrix(size_t rows, size_t cols, T min, T max);
    };

    // Vec class
    template <typename T, size_t _size>
    class Vec : public Matrix<T>
    {
    public:
        using Matrix<T>::Matrix;

        IL Vec() : Matrix<T>(_size, 1) {}

        IL Vec(std::initializer_list<T> init_list) : Matrix<T>(init_list.size(), init_list) {}

        Vec(const Matrix<T> &matrix) : Matrix<T>(matrix)
        {
            if (matrix.rows() != _size || matrix.cols() != 1)

                throw std::invalid_argument("Invalid matrix dimensions for Vec construction");
        }
        template <typename Y>
        IL Vec(const Vec<Y, _size> &other) : Matrix<T>(other)
        {
        }

        template <typename Y>
        IL Vec(Vec<Y, _size> &&other) noexcept : Matrix<T>(std::move(other))
        {
        }

        IL void print();
        IL T x() const
        {
            if (_size >= 1) return (*this)[0];
        }
        IL T y() const
        {
            if (_size >= 2) return (*this)[1];
        }
        IL T z() const
        {
            if (_size >= 3) return (*this)[2];
        }

        IL size_t size() const { return _size; }
        IL T &operator[](size_t i);
        IL const T &operator[](size_t i) const;
        IL Vec operator*(const T x) const;
        template <typename Y, size_t n_x>
        IL Vec operator*(const Vec<Y, n_x> &x) const;
        template <typename Y, size_t n_x>
        IL Vec operator/(const Vec<Y, n_x> &x) const;
        IL Vec operator+(const T x) const;
        template <typename Y, size_t n_x>
        IL Vec operator+(const Vec<Y, n_x> &x) const;
        template <typename Y, size_t n_x>
        IL Vec operator-(const Vec<Y, n_x> &x) const;
        template <typename Y>
        Vec<T, _size> operator*(const Matrix<Y> &m) const;
        IL void sin();
        IL void cos();
        template <typename Y, size_t n_x>
        IL double dot(const Vec<Y, n_x> &x) const;
        template <typename Y, size_t n_x>
        IL Vec cross(const Vec<Y, n_x> &x) const;
        IL double squared_magnitude() const;
        IL double magnitude() const;
        IL void power(float x);
        IL float normalize();
        IL Vec<float, _size> get_normalized_vector();
        template <typename Y, size_t n_x>
        IL float angle(const Vec<Y, n_x> &x) const;
        template <typename Y, size_t n_x>
        IL double distance(const Vec<Y, n_x> &x) const;
        template <typename Y, size_t n_x>
        IL double projection_onto(const Vec<Y, n_x> &x) const;
        IL void rotate(float angle, char8_t axis);
        IL Vec<T, _size> zero_vector();
        IL Vec<T, _size> ones_vector();
        IL Vec<T, _size> random_vector(T min, T max);
    };
}  // namespace utl

#ifdef L_GEBRA_IMPLEMENTATION

namespace utl
{

    template <typename T>
    void Matrix<T>::print()
    {
        // Find the maximum width for each column
        std::vector<size_t> col_widths(_cols, 0);
        for (size_t j = 0; j < _cols; ++j)
        {
            for (size_t i = 0; i < _rows; ++i)
            {
                size_t width = std::to_string((*this)(i, j)).size();
                if (width > col_widths[j]) col_widths[j] = width;
            }
        }

        // Print the matrix
        for (size_t i = 0; i < _rows; ++i)
        {
            std::cout << "[ ";
            for (size_t j = 0; j < _cols; ++j)
            {
                std::cout << std::setw(col_widths[j]) << (*this)(i, j) << " ";
            }
            std::cout << "]\n";
        }
    }

    template <typename T>
    template <typename Y>
    IL Matrix<T> Matrix<T>::operator+(const Matrix<Y> &other) const
    {
        if (_rows != other.rows() || _cols != other.cols())
        {
            throw std::invalid_argument("Matrix dimensions must match for addition");
        }

        Matrix<T> result(_rows, _cols);
        for (size_t i = 0; i < _rows; ++i)
        {
            for (size_t j = 0; j < _cols; ++j)
            {
                result(i, j) = (*this)(i, j) + static_cast<T>(other(i, j));
            }
        }
        return result;
    }

    template <typename T>
    template <typename Y>
    IL Matrix<T> Matrix<T>::operator-(const Matrix<Y> &other) const
    {
        if (_rows != other.rows() || _cols != other.cols())
        {
            throw std::invalid_argument("Matrix dimensions must match for subtraction");
        }

        Matrix<T> result(_rows, _cols);
        for (size_t i = 0; i < _rows; ++i)
        {
            for (size_t j = 0; j < _cols; ++j)
            {
                result(i, j) = (*this)(i, j) - static_cast<T>(other(i, j));
            }
        }
        return result;
    }

    template <typename T>
    IL Matrix<T> Matrix<T>::operator*(const T &scalar) const
    {
        Matrix<T> result(_rows, _cols);
        for (size_t i = 0; i < _rows; ++i)
        {
            for (size_t j = 0; j < _cols; ++j)
            {
                result(i, j) = (*this)(i, j) * scalar;
            }
        }
        return result;
    }

    template <typename T>
    template <typename Y>
    IL Matrix<T> Matrix<T>::operator*(const Matrix<Y> &other) const
    {
        if (_cols != other.rows())
        {
            throw std::invalid_argument("Matrix dimensions do not match for multiplication");
        }

        Matrix<T> result(_rows, other.cols());
        for (size_t i = 0; i < _rows; ++i)
        {
            for (size_t j = 0; j < other.cols(); ++j)
            {
                T sum = static_cast<T>(0);
                for (size_t k = 0; k < _cols; ++k)
                {
                    sum += (*this)(i, k) * static_cast<T>(other(k, j));
                }
                result(i, j) = sum;
            }
        }
        return result;
    }

    template <typename T>
    template <typename Y>
    IL Matrix<T> Matrix<T>::operator/(const Matrix<Y> &other) const
    {
        if (_rows != other.rows() || _cols != other.cols())
        {
            throw std::invalid_argument("Matrix dimensions must match for element-wise division");
        }

        Matrix<T> result(_rows, _cols);
        for (size_t i = 0; i < _rows; ++i)
        {
            for (size_t j = 0; j < _cols; ++j)
            {
                if (other(i, j) == 0)
                {
                    throw std::runtime_error("Division by zero");
                }
                result(i, j) = (*this)(i, j) / (T)other(i, j);
            }
        }
        return result;
    }

    template <typename T>
    IL void Matrix<T>::sin()
    {
        for (auto &val : data)
        {
            val = std::sin(val);
        }
    }

    template <typename T>
    IL void Matrix<T>::cos()
    {
        for (auto &val : data)
        {
            val = std::cos(val);
        }
    }

    template <typename T>
    IL Matrix<T> Matrix<T>::transpose() const
    {
        Matrix<T> result(_cols, _rows);
        for (size_t i = 0; i < _rows; ++i)
        {
            for (size_t j = 0; j < _cols; ++j)
            {
                result(j, i) = (*this)(i, j);
            }
        }
        return result;
    }

    template <typename T>
    double Matrix<T>::determinant() const
    {
        if (_rows != _cols)
        {
            throw std::invalid_argument("Matrix must be square to compute determinant");
        }

        if (_rows == 1)
        {
            return data[0];
        }
        else if (_rows == 2)
        {
            return (*this)(0, 0) * (*this)(1, 1) - (*this)(0, 1) * (*this)(1, 0);
        }
        else
        {
            double det = 0;
            int sign = 1;
            for (size_t i = 0; i < _cols; ++i)
            {
                det += sign * (*this)(0, i) * minor(0, i).determinant();
                sign = -sign;
            }
            return det;
        }
    }

    template <typename T>
    Matrix<T> Matrix<T>::minor(size_t row, size_t col) const
    {
        if (_rows <= 1 || _cols <= 1)
        {
            throw std::invalid_argument("Matrix must have dimensions greater than 1 to compute minor");
        }

        Matrix<T> result(_rows - 1, _cols - 1);
        size_t m = 0, n = 0;
        for (size_t i = 0; i < _rows; ++i)
        {
            if (i == row) continue;
            n = 0;
            for (size_t j = 0; j < _cols; ++j)
            {
                if (j == col) continue;
                result(m, n++) = (*this)(i, j);
            }
            if (i != row) ++m;
        }
        return result;
    }

    template <typename T>
    double Matrix<T>::cofactor(size_t row, size_t col) const
    {
        double factor = minor(row, col).determinant();
        return ((row + col) % 2 == 0) ? factor : -factor;
    }

    template <typename T>
    Matrix<T> Matrix<T>::inverse(const Matrix<T> &m)
    {
        if (m._rows != m._cols)
        {
            throw std::invalid_argument("Matrix must be square to compute inverse");
        }

        const double det = m.determinant();
        if (std::abs(det) < std::numeric_limits<double>::epsilon())
        {
            throw std::runtime_error("Matrix is singular and has no inverse");
        }

        // Check for singularity before creating the result matrix
        if (m._rows == 1)
        {
            // 1x1 matrix
            return Matrix<T>(1, 1, 1.0 / m(0, 0));
        }
        else if (m._rows == 2)
        {
            // 2x2 matrix
            const T a = m(0, 0), b = m(0, 1), c = m(1, 0), d = m(1, 1);
            const T inv_det = static_cast<T>(1.0 / det);
            return Matrix<T>(2, 2, d * inv_det, -b * inv_det, -c * inv_det, a * inv_det);
        }

        Matrix<T> result(m._rows, m._cols);
        for (size_t i = 0; i < m._rows; ++i)
        {
            for (size_t j = 0; j < m._cols; ++j)
            {
                result(i, j) = m.cofactor(i, j) / det;
            }
        }

        return result.transpose();
    }

    template <typename T>
    Matrix<T> Matrix<T>::power(int n)
    {
        if (_rows != _cols)
        {
            throw std::invalid_argument("Matrix must be square to compute power");
        }

        if (n < 0)
        {
            return inverse(*this).power(-n);
        }

        Matrix<T> result(_rows, _cols);
        for (size_t i = 0; i < _rows; ++i)
        {
            result(i, i) = 1;
        }

        Matrix<T> x = *this;
        while (n)
        {
            if (n & 1)
            {
                result = result * x;
            }
            x = x * x;
            n >>= 1;
        }
        return result;
    }

    template <typename T>
    Matrix<T> Matrix<T>::row_reduce()
    {
        Matrix<T> result = *this;
        size_t lead = 0;
        for (size_t r = 0; r < result._rows; ++r)
        {
            if (lead >= result._cols)
            {
                break;
            }
            size_t i = r;
            while (result(i, lead) == 0)
            {
                if (++i >= result._rows)
                {
                    i = r;
                    ++lead;
                    if (lead == result._cols)
                    {
                        return result;
                    }
                }
            }
            std::swap(result.data[result._cols * r], result.data[result._cols * i]);
            if (result(r, lead) != 0)
            {
                T lv = result(r, lead);
                for (size_t j = 0; j < result._cols; ++j)
                {
                    result(r, j) /= lv;
                }
                for (size_t i = 0; i < result._rows; ++i)
                {
                    if (i != r)
                    {
                        T lv = result(i, lead);
                        for (size_t j = 0; j < result._cols; ++j)
                        {
                            result(i, j) -= lv * result(r, j);
                        }
                    }
                }
                ++lead;
            }
        }
        return result;
    }
    template <typename T>
    size_t Matrix<T>::rank() const
    {
        Matrix<T> reduced = this->gauss_elimination().first;
        size_t rank = 0;
        for (size_t i = 0; i < _rows; ++i)
        {
            bool is_zero_row = true;
            for (size_t j = 0; j < _cols; ++j)
            {
                if (reduced(i, j) != 0)
                {
                    is_zero_row = false;
                    break;
                }
            }
            if (!is_zero_row) ++rank;
        }
        return rank;
    }
    template <typename T>
    T Matrix<T>::trace() const
    {
        if (_rows != _cols)
        {
            throw std::invalid_argument("Matrix must be square to compute trace");
        }

        T result = 0;
        for (size_t i = 0; i < _rows; ++i)
        {
            result += (*this)(i, i);
        }
        return result;
    }
    template <typename T>
    T Matrix<T>::norm() const
    {
        T result = 0;
        for (size_t i = 0; i < _rows; ++i)
        {
            for (size_t j = 0; j < _cols; ++j)
            {
                result += (*this)(i, j) * (*this)(i, j);
            }
        }
        return std::sqrt(result);
    }
    template <typename T>
    void Matrix<T>::swap_rows(size_t r1, size_t r2)
    {
        if (r1 >= _rows || r2 >= _rows)
        {
            throw std::out_of_range("Row index out of range");
        }

        for (size_t j = 0; j < _cols; ++j)
        {
            std::swap((*this)(r1, j), (*this)(r2, j));
        }
    }

    // Multiply a row by a scalar value
    template <typename T>
    void Matrix<T>::multiply_row(size_t r, float scalar)
    {
        if (r >= _rows)
        {
            throw std::out_of_range("Row index out of range");
        }

        for (size_t j = 0; j < _cols; ++j)
        {
            (*this)(r, j) *= scalar;
        }
    }

    // Add a scalar multiple of one row to another row
    template <typename T>
    void Matrix<T>::add_scaled_row(size_t r1, size_t r2, float scalar)
    {
        if (r1 >= _rows || r2 >= _rows)
        {
            throw std::out_of_range("Row index out of range");
        }

        for (size_t j = 0; j < _cols; ++j)
        {
            (*this)(r1, j) += scalar * (*this)(r2, j);
        }
    }
    template <typename T>
    std::vector<T> Matrix<T>::eigenvalues(const Matrix<T> &m)
    {
        if (m._rows != m._cols)
        {
            throw std::invalid_argument("Matrix must be square to compute eigenvalues");
        }

        size_t n = m._rows;
        std::vector<T> eigenvals(n);

        // Helper function to compute characteristic polynomial
        auto charpoly = [&](const T &x)
        {
            std::vector<std::vector<T>> B(n, std::vector<T>(n));
            for (size_t i = 0; i < n; ++i)
            {
                for (size_t j = 0; j < n; ++j)
                {
                    B[i][j] = m(i, j);
                }
                B[i][i] -= x;
            }
            T det = 1;
            for (size_t i = 0; i < n; ++i)
            {
                T sum = 0;
                for (size_t j = 0; j < n; ++j)
                {
                    sum += B[i][j];
                }
                det *= sum;
                if (det == 0)
                {
                    break;
                }
                for (size_t j = i + 1; j < n; ++j)
                {
                    for (size_t k = i + 1; k < n; ++k)
                    {
                        B[j][k] -= B[j][i] * B[i][k] / B[i][i];
                    }
                }
            }
            return det;
        };

        // Find eigenvalues using Newton's method
        for (size_t i = 0; i < n; ++i)
        {
            T x = 1;
            double tol = 1e-12;  // Desired tolerance for convergence
            for (size_t j = 0; j < 100; ++j)
            {
                T f = charpoly(x);
                if (std::abs(f) < tol)
                {
                    eigenvals[i] = x;
                    break;
                }
                T fp = 0;
                T dx = 1e-6;
                for (size_t k = 0; k < 5; ++k)
                {
                    fp += (charpoly(x + dx) - f) / dx;
                    dx /= 2;
                }
                x -= f / fp;
            }
        }

        return eigenvals;
    }

    template <typename T>
    std::vector<std::vector<T>> Matrix<T>::eigenvectors(const Matrix<T> &m)
    {
        if (m._rows != m._cols)
        {
            throw std::invalid_argument("Matrix must be square to compute eigenvectors");
        }

        size_t n = m._rows;
        std::vector<std::vector<T>> eigvecs(n, std::vector<T>(n));
        std::vector<T> eigenvals = eigenvalues(m);

        // Compute eigenvectors
        for (size_t i = 0; i < n; ++i)
        {
            T lambda = eigenvals[i];
            std::vector<T> b(n, 0);
            b[0] = 1;
            for (size_t j = 0; j < n; ++j)
            {
                T sum = 0;
                for (size_t k = 0; k < n; ++k)
                {
                    sum += m(j, k) * b[k];
                }
                b[j] = sum - lambda * b[j];
            }
            T norm = 0;
            for (size_t j = 0; j < n; ++j)
            {
                norm += b[j] * b[j];
            }
            norm = std::sqrt(norm);
            for (size_t j = 0; j < n; ++j)
            {
                eigvecs[i][j] = b[j] / norm;
            }
        }

        return eigvecs;
    }

    template <typename T>
    std::pair<Matrix<T>, Matrix<T>> Matrix<T>::lu_decomposition()
    {
        if (_rows != _cols)
        {
            throw std::invalid_argument("LU decomposition requires a square matrix");
        }

        Matrix<T> L(_rows, _cols);
        Matrix<T> U(*this);

        for (size_t i = 0; i < _rows; ++i)
        {
            // Partial pivoting
            T maxVal = std::abs(U(i, i));
            size_t maxRow = i;
            for (size_t k = i + 1; k < _rows; ++k)
            {
                if (std::abs(U(k, i)) > maxVal)
                {
                    maxVal = std::abs(U(k, i));
                    maxRow = k;
                }
            }

            if (std::abs(maxVal) < std::numeric_limits<double>::epsilon())
            {
                throw std::runtime_error("Matrix is singular or nearly singular");
            }

            // Swap maximum row with current row
            for (size_t k = i; k < _rows; ++k)
            {
                std::swap(U(maxRow, k), U(i, k));
                std::swap(L(maxRow, k), L(i, k));
            }

            L(i, i) = 1;  // Diagonal of L is 1
            for (size_t j = i + 1; j < _rows; ++j)
            {
                T factor = U(j, i) / U(i, i);
                L(j, i) = factor;
                for (size_t k = i; k < _cols; ++k)
                {
                    U(j, k) -= factor * U(i, k);
                }
            }
        }

        return std::make_pair(L, U);
    }

    template <typename T>
    std::pair<Matrix<T>, Matrix<T>> Matrix<T>::qr_decomposition()
    {
        if (_rows != _cols)
        {
            throw std::invalid_argument("QR decomposition requires a square matrix");
        }

        Matrix<T> Q(_rows, _cols);
        Matrix<T> R(_rows, _cols, 0);  // Initialize to zero

        for (size_t k = 0; k < _cols; ++k)
        {
            // Compute the kth column of Q
            for (size_t i = 0; i < _rows; ++i)
            {
                Q(i, k) = (*this)(i, k);
            }

            for (size_t i = 0; i < k; ++i)
            {
                T dotProduct = (*this).dot(Q);

                for (size_t j = 0; j < _rows; ++j)
                {
                    Q(j, k) -= dotProduct * Q(j, i);
                }
            }

            // Normalize the kth column of Q
            T norm = 0;
            for (size_t j = 0; j < _rows; ++j)
            {
                norm += Q(j, k) * Q(j, k);
            }
            norm = std::sqrt(norm);

            if (std::abs(norm) < std::numeric_limits<double>::epsilon())
            {
                throw std::runtime_error("Matrix is singular or nearly singular");
            }

            for (size_t j = 0; j < _rows; ++j)
            {
                Q(j, k) /= norm;
            }

            // Compute the kth row of R
            for (size_t i = k; i < _cols; ++i)
            {
                for (size_t j = 0; j < _rows; ++j)
                {
                    R(k, i) += Q(j, k) * (*this)(j, i);
                }
            }
        }

        return std::make_pair(Q, R);
    }

    template <typename T>
    Matrix<T> Matrix<T>::augment(const Matrix<T> &other) const
    {
        if (_rows != other._rows)
        {
            throw std::invalid_argument("Matrix dimensions must match for augmentation");
        }

        Matrix<T> augmented(_rows, _cols + other._cols);
        for (size_t i = 0; i < _rows; ++i)
        {
            for (size_t j = 0; j < _cols; ++j)
            {
                augmented(i, j) = (*this)(i, j);
            }
            for (size_t j = 0; j < other._cols; ++j)
            {
                augmented(i, _cols + j) = other(i, j);
            }
        }
        return augmented;
    }

    template <typename T>
    Matrix<T> Matrix<T>::gauss_elimination() const
    {
        Matrix<float> result = (*this);

        int lead = 0;
        int n_rows = result.rows();

        while (lead < n_rows)
        {
            float divisor, multiplier;
            for (size_t r = 0; r < n_rows; ++r)
            {
                divisor = 1 / (float)(result(lead, lead));

                multiplier = (float)result(r, lead) / (float)result(lead, lead) * (-1);
                if (r == lead)
                    result.multiply_row(r, divisor);
                else
                {
                    result.add_scaled_row(r, lead, multiplier);
                }
            }
            lead++;
        }
        return result;
    }

    template <typename T>
    sol_type Matrix<T>::has_solution(const Matrix<T> &reduced_matrix)
    {
        size_t non_zero_rows = 0;
        for (size_t i = 0; i < reduced_matrix.rows(); ++i)
        {
            bool is_zero_row = true;
            for (size_t j = 0; j < reduced_matrix.cols(); ++j)
            {
                if (reduced_matrix(i, j) != 0)
                {
                    is_zero_row = false;
                    break;
                }
            }
            if (!is_zero_row) ++non_zero_rows;
        }

        if (non_zero_rows == reduced_matrix.cols())
        {
            return sol_type::Unique;
        }
        else if (non_zero_rows < reduced_matrix.cols())
        {
            return sol_type::Infinite;
        }
        else
        {
            return sol_type::NoSolution;
        }
    }

    template <typename T>
    Matrix<T> Matrix<T>::solve_linear_system(const Matrix<T> &b) const
    {
        if (_rows != _cols || _rows != b.rows())
        {
            throw std::invalid_argument("Invalid matrix and vector dimensions");
        }

        Matrix<T> augmented = augment(b);
        Matrix<T> reduced_matrix = augmented.gauss_elimination();
        sol_type solution_type = has_solution(*this);

        if (solution_type == sol_type::NoSolution)
        {
            throw std::runtime_error("No solution exists for the linear system");
        }

        if (solution_type == sol_type::Infinite)
        {
            throw std::runtime_error("Infinite solutions exist for the linear system");
        }

        Matrix<T> x(_cols, 1);
        if (solution_type == sol_type::Unique)
        {
            for (int i = _rows - 1; i >= 0; --i)
            {
                T sum = reduced_matrix(i, _cols);
                for (size_t j = i + 1; j < _cols; ++j)
                {
                    sum -= reduced_matrix(i, j) * x(j, 0);
                }

                x(i, 0) = sum / reduced_matrix(i, i);
            }
        }
        return x;
    }
    template <typename T>
    IL Matrix<T> Matrix<T>::identity_matrix(size_t n)
    {
        Matrix<T> result(n, n);
        for (size_t i = 0; i < n; ++i)
        {
            result(i, i) = static_cast<T>(1);
        }
        return result;
    }

    template <typename T>
    IL Matrix<T> Matrix<T>::zero_matrix(size_t n)
    {
        Matrix<T> result(n, n);
        for (size_t i = 0; i < n; ++i)
        {
            for (size_t j = 0; j < n; ++j) result(i, j) = static_cast<T>(0);
        }
        return result;
    }

    template <typename T>
    IL Matrix<T> Matrix<T>::ones_matrix(size_t n)
    {
        Matrix<T> result(n, n);
        for (size_t i = 0; i < n; ++i)
        {
            for (size_t j = 0; j < n; ++j) result(i, j) = static_cast<T>(1);
        }
        return result;
    }

    template <typename T>
    IL Matrix<T> Matrix<T>::random_matrix(size_t rows, size_t cols, T min, T max)
    {
        Matrix<T> result(rows, cols);
        for (size_t i = 0; i < rows; ++i)
        {
            for (size_t j = 0; j < cols; ++j)
            {
                result(i, j) = min + static_cast<T>(rand()) / (static_cast<T>(RAND_MAX / (max - min)));
            }
        }
        return result;
    }

    // Definitions of Vec member functions
    template <typename T, size_t _size>
    IL void Vec<T, _size>::print()
    {
        std::cout << "[ ";
        for (size_t i = 0; i < _size; ++i)
        {
            std::cout << (*this)[i] << " ";
        }
        std::cout << "]\n";
    }

    template <typename T, size_t _size>
    IL T &Vec<T, _size>::operator[](size_t i)
    {
        if (i >= _size)
        {
            throw std::out_of_range("Index out of range");
        }
        return this->operator()(i, 0);
    }

    template <typename T, size_t _size>
    IL const T &Vec<T, _size>::operator[](size_t i) const
    {
        if (i >= _size)
        {
            throw std::out_of_range("Index out of range");
        }
        return this->operator()(i, 0);
    }

    template <typename T, size_t _size>
    IL Vec<T, _size> Vec<T, _size>::operator*(const T x) const
    {
        Vec<T, _size> result;
        for (size_t i = 0; i < _size; ++i)
        {
            result[i] = (*this)[i] * x;
        }
        return result;
    }

    template <typename T, size_t _size>
    template <typename Y, size_t n_x>
    IL Vec<T, _size> Vec<T, _size>::operator*(const Vec<Y, n_x> &x) const
    {
        if (_size != n_x)
        {
            throw std::invalid_argument("Vector sizes do not match for multiplication");
        }
        Vec<T, _size> result;
        for (size_t i = 0; i < _size; ++i)
        {
            result[i] = (*this)[i] * x[i];
        }
        return result;
    }

    template <typename T, size_t _size>
    template <typename Y, size_t n_x>
    IL Vec<T, _size> Vec<T, _size>::operator/(const Vec<Y, n_x> &x) const
    {
        if (_size != n_x)
        {
            throw std::invalid_argument("Vector sizes do not match for division");
        }
        Vec<T, _size> result;
        for (size_t i = 0; i < _size; ++i)
        {
            if (x[i] == 0)
            {
                throw std::invalid_argument("Division by zero");
            }
            result[i] = (*this)[i] / x[i];
        }
        return result;
    }

    template <typename T, size_t _size>
    IL Vec<T, _size> Vec<T, _size>::operator+(const T x) const
    {
        Vec<T, _size> result;
        for (size_t i = 0; i < _size; ++i)
        {
            result[i] = (*this)[i] + x;
        }
        return result;
    }

    template <typename T, size_t _size>
    template <typename Y, size_t n_x>
    IL Vec<T, _size> Vec<T, _size>::operator+(const Vec<Y, n_x> &x) const
    {
        if (_size != n_x)
        {
            throw std::invalid_argument("Vector sizes do not match for addition");
        }
        Vec<T, _size> result;
        for (size_t i = 0; i < _size; ++i)
        {
            result[i] = (*this)[i] + x[i];
        }
        return result;
    }

    template <typename T, size_t _size>
    template <typename Y, size_t n_x>
    IL Vec<T, _size> Vec<T, _size>::operator-(const Vec<Y, n_x> &x) const
    {
        if (_size != n_x)
        {
            throw std::invalid_argument("Vector sizes do not match for subtraction");
        }
        Vec<T, _size> result;
        for (size_t i = 0; i < _size; ++i)
        {
            result[i] = (*this)[i] - x[i];
        }
        return result;
    }

    template <typename T, size_t _size>
    template <typename Y>
    Vec<T, _size> Vec<T, _size>::operator*(const Matrix<Y> &m) const
    {
        if (m.cols() != _size)
        {
            throw std::invalid_argument("Vector and matrix dimensions do not match for multiplication");
        }

        Vec<T, m.rows()> result;
        for (size_t i = 0; i < m.rows(); ++i)
        {
            T sum = static_cast<T>(0);
            for (size_t j = 0; j < _size; ++j)
            {
                sum += (*this)[j] * m(i, j);
            }
            result[i] = sum;
        }
        return result;
    }

    template <typename T, size_t _size>
    IL void Vec<T, _size>::sin()
    {
        for (size_t i = 0; i < _size; ++i)
        {
            (*this)[i] = std::sin((*this)[i]);
        }
    }

    template <typename T, size_t _size>
    IL void Vec<T, _size>::cos()
    {
        for (size_t i = 0; i < _size; ++i)
        {
            (*this)[i] = std::cos((*this)[i]);
        }
    }

    template <typename T, size_t _size>
    template <typename Y, size_t n_x>
    IL double Vec<T, _size>::dot(const Vec<Y, n_x> &x) const
    {
        if (_size != n_x)
        {
            throw std::invalid_argument("Vector sizes do not match for dot product");
        }
        double result = 0;
        for (size_t i = 0; i < _size; ++i)
        {
            result += (*this)[i] * x[i];
        }
        return result;
    }

    template <typename T, size_t _size>
    template <typename Y, size_t n_x>
    IL Vec<T, _size> Vec<T, _size>::cross(const Vec<Y, n_x> &x) const
    {
        if (_size != 3 || n_x != 3)
        {
            throw std::invalid_argument("Cross product is defined only for 3-dimensional vectors");
        }
        Vec<T, _size> result;
        result[0] = (*this)[1] * x[2] - (*this)[2] * x[1];
        result[1] = (*this)[2] * x[0] - (*this)[0] * x[2];
        result[2] = (*this)[0] * x[1] - (*this)[1] * x[0];
        return result;
    }

    template <typename T, size_t _size>
    IL double Vec<T, _size>::squared_magnitude() const
    {
        double sum = 0;
        for (size_t i = 0; i < _size; ++i)
        {
            sum += (*this)[i] * (*this)[i];
        }
        return sum;
    }

    template <typename T, size_t _size>
    IL double Vec<T, _size>::magnitude() const
    {
        double sum = 0;
        for (size_t i = 0; i < _size; ++i)
        {
            sum += (*this)[i] * (*this)[i];
        }
        return sqrt(sum);
    }

    template <typename T, size_t _size>
    IL void Vec<T, _size>::power(float x)
    {
        if (!std::isfinite(x)) throw std::invalid_argument("Power must be a finite number");

        for (size_t i = 0; i < _size; ++i)
        {
            (*this)[i] = std::pow((*this)[i], x);
        }
    }

    template <typename T, size_t _size>
    IL float Vec<T, _size>::normalize()
    {
        double mag = magnitude();
        if (mag == 0)
        {
            throw std::runtime_error("Cannot normalize a zero vector");
        }
        for (size_t i = 0; i < _size; ++i)
        {
            (*this)[i] /= mag;
        }
        return static_cast<float>(mag);
    }

    template <typename T, size_t _size>
    IL Vec<float, _size> Vec<T, _size>::get_normalized_vector()
    {
        Vec<float, _size> result;
        double mag = magnitude();
        if (mag == 0)
        {
            throw std::runtime_error("Cannot normalize a zero vector");
        }
        for (size_t i = 0; i < _size; ++i)
        {
            result[i] = static_cast<float>((*this)[i] / mag);
        }
        return result;
    }

    template <typename T, size_t _size>
    template <typename Y, size_t n_x>
    IL float Vec<T, _size>::angle(const Vec<Y, n_x> &x) const
    {
        double dot_product = dot(x);
        double this_magnitude = magnitude();
        double x_magnitude = x.magnitude();

        if (this_magnitude == 0 || x_magnitude == 0)
        {
            throw std::runtime_error("Cannot compute angle for zero vector");
        }

        double cos_theta = dot_product / (this_magnitude * x_magnitude);
        return acos(static_cast<float>(cos_theta));
    }

    template <typename T, size_t _size>
    template <typename Y, size_t n_x>
    IL double Vec<T, _size>::distance(const Vec<Y, n_x> &x) const
    {
        if (_size != n_x)
        {
            throw std::invalid_argument("Vector sizes do not match for distance calculation");
        }
        double sum = 0;
        for (size_t i = 0; i < _size; ++i)
        {
            sum += ((*this)[i] - x[i]) * ((*this)[i] - x[i]);
        }
        return sqrt(sum);
    }

    template <typename T, size_t _size>
    template <typename Y, size_t n_x>
    IL double Vec<T, _size>::projection_onto(const Vec<Y, n_x> &x) const
    {
        if (x.magnitude() == 0)
        {
            throw std::runtime_error("Cannot project onto a zero vector");
        }
        return dot(x) / (x.magnitude() * x.magnitude());
    }

    template <typename T, size_t _size>
    IL void Vec<T, _size>::rotate(float angle, char8_t axis)
    {
        if (_size == 2)
        {
            if (axis != 'z' && axis != 'Z')
            {
                throw std::invalid_argument("For vectors of size 2, only rotation around the Z-axis is supported");
            }

            Vec<T, _size> result;

            double cosA = std::cos(angle);
            double sinA = std::sin(angle);

            double x = (*this)[0];
            double y = (*this)[1];
            result[0] = x * cosA - y * sinA;
            result[1] = x * sinA + y * cosA;

            *this = result;
        }
        else if (_size == 3)
        {
            Vec<T, _size> result;

            double cosA = std::cos(angle);
            double sinA = std::sin(angle);

            switch (axis)
            {
                case 'x':
                case 'X':
                    result[0] = (*this)[0];
                    result[1] = (*this)[1] * cosA - (*this)[2] * sinA;
                    result[2] = (*this)[1] * sinA + (*this)[2] * cosA;
                    break;

                case 'y':
                case 'Y':
                    result[0] = (*this)[0] * cosA + (*this)[2] * sinA;
                    result[1] = (*this)[1];
                    result[2] = -(*this)[0] * sinA + (*this)[2] * cosA;
                    break;

                case 'z':
                case 'Z':
                    result[0] = (*this)[0] * cosA - (*this)[1] * sinA;
                    result[1] = (*this)[0] * sinA + (*this)[1] * cosA;
                    result[2] = (*this)[2];
                    break;

                default:
                    throw std::invalid_argument("Invalid rotation axis");
            }
            *this = result;
        }
        else
        {
            throw std::invalid_argument("Rotation only supported for vectors of size 2 or 3");
        }
    }

    template <typename T, size_t _size>
    IL Vec<T, _size> Vec<T, _size>::zero_vector()
    {
        for (size_t i = 0; i < _size; ++i)
        {
            (*this)[i] = 0;
        }
    }

    template <typename T, size_t _size>
    IL Vec<T, _size> Vec<T, _size>::ones_vector()
    {
        for (size_t i = 0; i < _size; ++i)
        {
            (*this)[i] = 1;
        }
    }

    template <typename T, size_t _size>
    IL Vec<T, _size> Vec<T, _size>::random_vector(T min, T max)
    {
        for (size_t i = 0; i < _size; ++i)
        {
            (*this)[i] = min + static_cast<T>(rand()) / (static_cast<T>(RAND_MAX / (max - min)));
        }
    }

}  // namespace utl

#endif  // L_GEBRA_IMPLEMENTATION

