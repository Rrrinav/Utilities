//   ##:::::::::::::::::'######:::'########:'########::'########:::::'###::::
//   ##::::::::::::::::'##... ##:: ##.....:: ##.... ##: ##.... ##:::'## ##:::
//   ##:::::::::::::::: ##:::..::: ##::::::: ##:::: ##: ##:::: ##::'##:. ##::
//   ##:::::::::::::::: ##::'####: ######::: ########:: ########::'##:::. ##:
//   ##:::::::::::::::: ##::: ##:: ##...:::: ##.... ##: ##.. ##::: #########:
//   ##:::::::::::::::: ##::: ##:: ##::::::: ##:::: ##: ##::. ##:: ##.... ##:
//   ########:'#######:. ######::: ########: ########:: ##:::. ##: ##:::: ##:
//  ........::.......:::......::::........::........:::..:::::..::..:::::..::
// L_Gebra - A simple header only linear algebra library

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

#pragma once

#include <cassert>
#include <cmath>
#include <cstddef>
#include <initializer_list>
#include <iostream>
#include <stdexcept>
#include <type_traits>
#include <vector>

#define IL inline
#define ST static
#define V virtual

namespace utl
{

    // Matrix class
    template <typename T>
    class Matrix
    {
    protected:
        std::vector<T> data;
        size_t _rows;
        size_t _cols;

    public:
        Matrix() : _rows(0), _cols(0) {}

        Matrix(size_t rows, size_t cols) : data(rows * cols, 0), _rows(rows), _cols(cols) {}

        Matrix(std::initializer_list<std::initializer_list<T>> initList)
            : _rows(initList.size()), _cols(initList.size() > 0 ? initList.begin()->size() : 0)
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

        Matrix(std::initializer_list<T> initList) : _rows(initList.size()), _cols(1), data(initList) {}

        Matrix(std::vector<T> &&other) noexcept
            : data(std::move(other)), _rows(data.size()), _cols(data.empty() ? 0 : data.size() / _rows)
        {
        }

        ~Matrix() = default;

        V size_t size() const { return data.size(); }

        size_t rows() const { return _rows; }

        size_t cols() const { return _cols; }

        T &operator()(size_t row, size_t col)
        {
            if (row >= rows() || col >= cols())
            {
                throw std::out_of_range("Index out of range");
            }
            return data[row * cols() + col];
        }

        const T &operator()(size_t row, size_t col) const
        {
            if (row >= rows() || col >= cols())
            {
                throw std::out_of_range("Index out of range");
            }
            return data[row * cols() + col];
        }
        template <typename Y>
        Matrix<T> operator+(const Matrix<Y> &other) const;
        template <typename Y>
        Matrix<T> operator-(const Matrix<Y> &other) const;
        Matrix<T> operator*(const T &scalar) const;
        template <typename Y>
        Matrix<T> operator*(const Matrix<Y> &other) const;
    };

    // Vec class
    template <typename T>
    class Vec : public Matrix<T>
    {
        size_t _size;

    public:
        /* using Matrix<T>::Matrix; */

        Vec() : _size(0) {}

        Vec(int size) : _size(size), Matrix<T>(size, 1)
        {
            if (size <= 0)
            {
                throw std::invalid_argument("Vector size must be positve");
            }
        }

        Vec(std::initializer_list<T> initList) : _size(initList.size()), Matrix<T>(initList) {}

        Vec(const Vec<T> &other) : _size(other.size()), Matrix<T>(other) {}

        Vec(Vec<T> &&other) noexcept : _size(other._size), Matrix<T>(std::move(other)) { other._size = 0; }

        // Constructors for compatibility with Vec2 and Vec3
        Vec(T x, T y) : _size(2), Matrix<T>({x, y}) {}
        Vec(T x, T y, T z) : _size(3), Matrix<T>({x, y, z}) {}

        size_t size() const { return _size; }
        V IL void print();
        V IL T &operator[](size_t i);
        V IL const T &operator[](size_t i) const;

        V IL Vec operator*(const T x) const;
        template <typename Y>
        IL Vec operator*(const Vec<Y> &x) const;
        template <typename Y>
        IL Vec operator/(const Vec<Y> &x) const;
        V IL Vec operator+(const T x) const;
        template <typename Y>
        IL Vec operator+(const Vec<Y> &x) const;
        template <typename Y>
        IL Vec operator-(const Vec<Y> &x) const;

        template <typename Y>
        IL double dot(const Vec<Y> &x) const;
        template <typename Y>
        IL Vec cross(const Vec<Y> &x) const;
        V IL double magnitude() const;
        V IL float normalize();
        V IL Vec<float> get_normalized_vector();
        template <typename Y>
        IL float angle(const Vec<Y> &x) const;
        template <typename Y>
        IL double distance(const Vec<Y> &x) const;
        template <typename Y>
        IL double projection(const Vec<Y> &x) const;
    };

    // Vec2 class
    template <typename T>
    class Vec2 : public Vec<T>
    {
    public:
        using Vec<T>::Vec;
    };

    // Vec3 class
    template <typename T>
    class Vec3 : public Vec<T>
    {
    public:
        using Vec<T>::Vec;
    };
}  // namespace utl
#ifdef L_GEBRA_IMPLEMENTATION

namespace utl
{

    template <typename T>
    template <typename Y>
    Matrix<T> Matrix<T>::operator+(const Matrix<Y> &other) const
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
    Matrix<T> Matrix<T>::operator-(const Matrix<Y> &other) const
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
    Matrix<T> Matrix<T>::operator*(const T &scalar) const
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
    Matrix<T> Matrix<T>::operator*(const Matrix<Y> &other) const
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
    IL void Vec<T>::print()
    {
        std::cout << "[ ";
        for (size_t i = 0; i < this->size(); ++i)
        {
            std::cout << (*this)[i] << " ";
        }
        std::cout << "]\n";
    }

    template <typename T>
    IL T &Vec<T>::operator[](size_t i)
    {
        if (i >= this->size())
        {
            throw std::out_of_range("Index out of range");
        }
        return this->operator()(i, 0);
    }

    template <typename T>
    IL const T &Vec<T>::operator[](size_t i) const
    {
        if (i >= this->size())
        {
            throw std::out_of_range("Index out of range");
        }
        return this->operator()(i, 0);
    }

    template <typename T>
    IL Vec<T> Vec<T>::operator*(const T x) const
    {
        Vec<T> result(this->size());
        for (size_t i = 0; i < this->size(); ++i)
        {
            result[i] = (*this)[i] * x;
        }
        return result;
    }

    template <typename T>
    template <typename Y>
    Vec<T> Vec<T>::operator*(const Vec<Y> &x) const
    {
        if (_size != x.size())
        {
            throw std::invalid_argument("Vector sizes do not match for multiplication");
        }
        Vec<T> result(_size);
        for (size_t i = 0; i < _size; ++i)
        {
            result[i] = (*this)[i] * static_cast<T>(x[i]);
        }

        return result;
    }
    template <typename T>
    template <typename Y>
    IL Vec<T> Vec<T>::operator/(const Vec<Y> &x) const
    {
        if (this->size() != x.size())
        {
            throw std::invalid_argument("Vector sizes do not match for division");
        }
        Vec<T> result(this->size());
        for (size_t i = 0; i < this->size(); ++i)
        {
            if (x[i] == 0)
            {
                throw std::invalid_argument("Division by zero");
            }
            result[i] = (*this)[i] / static_cast<T>(x[i]);
        }
        return result;
    }

    template <typename T>
    IL Vec<T> Vec<T>::operator+(const T x) const
    {
        Vec<T> result(this->size());
        for (size_t i = 0; i < this->size(); ++i)
        {
            result[i] = (*this)[i] + x;
        }
        return result;
    }

    template <typename T>
    template <typename Y>
    IL Vec<T> Vec<T>::operator+(const Vec<Y> &x) const
    {
        if (this->size() != x.size())
        {
            throw std::invalid_argument("Vector sizes do not match for addition");
        }
        Vec<T> result(this->size());
        for (size_t i = 0; i < this->size(); ++i)
        {
            result[i] = (*this)[i] + static_cast<T>(x[i]);
        }
        return result;
    }

    template <typename T>
    template <typename Y>
    IL Vec<T> Vec<T>::operator-(const Vec<Y> &x) const
    {
        if (this->size() != x.size())
        {
            throw std::invalid_argument("Vector sizes do not match for subtraction");
        }
        Vec<T> result(this->size());
        for (size_t i = 0; i < this->size(); ++i)
        {
            result[i] = (*this)[i] - static_cast<T>(x[i]);
        }
        return result;
    }

    template <typename T>
    template <typename Y>
    IL double Vec<T>::dot(const Vec<Y> &x) const
    {
        if (this->size() != x.size())
        {
            throw std::invalid_argument("Vector sizes do not match for dot product");
        }
        double result;
        for (size_t i = 0; i < this->size(); ++i)
        {
            result += (double)(*this)[i] * (double)(x[i]);
        }
        return result;
    }

    template <typename T>
    template <typename Y>
    IL Vec<T> Vec<T>::cross(const Vec<Y> &x) const
    {
        if (this->size() != 3 || x.size() != 3)
        {
            throw std::invalid_argument("Cross product is only defined for 3D Vectors");
        }
        Vec<T> result(3);
        result[0] = (*this)[1] * x[2] - (*this)[2] * x[1];
        result[1] = (*this)[2] * x[0] - (*this)[0] * x[2];
        result[2] = (*this)[0] * x[1] - (*this)[1] * x[0];
        return result;
    }

    template <typename T>
    IL double Vec<T>::magnitude() const
    {
        T result = 0;
        for (size_t i = 0; i < this->size(); ++i)
        {
            result += (*this)[i] * (*this)[i];
        }
        return std::sqrt(result);
    }

    template <typename T>
    IL float Vec<T>::normalize()
    {
        double mag = magnitude();
        if (mag == 0)
        {
            throw std::invalid_argument("Cannot normalize a zero vector");
        }
        for (size_t i = 0; i < _size; ++i)
        {
            (*this)[i] /= mag;
        }
        return mag;
    }

    template <typename T>
    IL Vec<float> Vec<T>::get_normalized_vector()
    {
        double mag = magnitude();
        if (mag == 0)
        {
            throw std::invalid_argument("Cannot normalize a zero vector");
        }
        Vec<float> result(_size);

        for (size_t i = 0; i < _size; ++i)
        {
            result[i] = (float)(*this)[i] / mag;
        }
        return result;
    }

    template <typename T>
    template <typename Y>
    IL float Vec<T>::angle(const Vec<Y> &x) const
    {
        if (this->size() != x.size())
        {
            throw std::invalid_argument("Vector sizes do not match for angle calculation");
        }
        double thisMag = magnitude();
        double xMag = x.magnitude();
        if (thisMag == 0 || xMag == 0)
        {
            throw std::invalid_argument("Cannot calculate angle for zero vector(s)");
        }
        float d = dot(x);
        return static_cast<float>(std::acos(static_cast<float>(dot(x) / (thisMag * xMag))));
    }

    template <typename T>
    template <typename Y>
    IL double Vec<T>::distance(const Vec<Y> &x) const
    {
        if (this->size() != x.size())
        {
            throw std::invalid_argument("Vector sizes do not match for distance calculation");
        }
        double result = 0;
        for (size_t i = 0; i < this->size(); ++i)
        {
            result += ((*this)[i] - (double)(x[i])) * ((*this)[i] - (double)(x[i]));
        }
        return std::sqrt(result);
    }

    template <typename T>
    template <typename Y>
    IL double Vec<T>::projection(const Vec<Y> &x) const
    {
        if (this->size() != x.size())
        {
            throw std::invalid_argument("Vector sizes do not match for projection");
        }
        double xMag = x.magnitude();
        double dot = this->dot(x);
        if (xMag == 0)
        {
            throw std::invalid_argument("Cannot project onto a zero vector");
        }
        return (this->dot(x) / (xMag * xMag));
    }

}  // namespace utl

#endif  // L_GEBRA_IMPLEMENTATION
