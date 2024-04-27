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

// HOW TO USE?
//  You just need to include this header file in your project and use the vec
//  class with namespace utl. You also need macro L_GEBRA_IMPLEMENTATION before
//  "#include "l_gebra"" in one of your source files to include the
//  implementation. Basically :- #define L_GEBRA_IMPLEMENTATION #include
//  "l_gebra.hpp"

#pragma once

#include <cstddef>
#include <initializer_list>
#include <iostream>
#include <stdexcept>
#include <type_traits>
#include <vector>

#define IL inline
#define ST static

namespace utl
{
  /*
  * TODO: Implement the following methods:
          dot product, cross product, magnitude, normalize, angle between two
          vectors, distance between two vectors, projection of one vector onto
  another, reflection of one vector over another, vector rotation, vector scaling
  reject, copy, is equal, empty, reflect, swizzle, clamp, min, max, sum, average,
  median, mode, variance, standard deviation,
   */

  template <typename T>
  class vec
  {
  private:
    std::vector<T> data;

  public:
    // Constructors
    vec() = default;

    // Constructor with size
    vec(int n) : data(n) {}

    // Constructor with initializer list
    vec(std::initializer_list<T> list) : data(list) {}

    // Constructor with a scalar value to initialize all elements
    explicit vec(int n, const T &value) : data(n, value) {}

    // Constructor with iterators
    template <typename InputIt>
    vec(InputIt first, InputIt last) : data(first, last) {}

    // Constructor with an existing vector
    vec(const std::vector<T> &other) : data(other) {}

    // Constructor with an existing vector, move version
    vec(std::vector<T> &&other) noexcept : data(std::move(other)) {}

    // FUNCTION DECLARATION

    // Print the vector
    IL void print();

    // Return size of the vector
    IL size_t size() const;

    // Operator to access elements
    IL T &operator[](size_t i);
    IL const T &operator[](size_t i) const;

    // ARITHMETIC OPERATIONS

    // Multiply vector by a scalar
    IL vec multiply(const T x);

    // Multiply vector by another vector
    template <typename Y>
    IL vec multiply(const vec<Y> &x);

    // Add a scalar to vector, just use negative scalar to subtract, if its a
    // variable, multiply by -1.
    IL vec add(const T x);

    // Add two vectors
    template <typename Y>
    IL vec add(const vec<Y> &x);

    // Subtract a vector from vector
    template <typename Y>
    IL vec subtract(const vec<Y> &x);
  };

  template <typename T>
  ST IL T lerp(const T &start, const T &end, double t);

} // namespace utl

// IMPLEMENTATIONS HERE

#ifdef L_GEBRA_IMPLEMENTATION

namespace utl
{

  template <typename T>
  IL size_t vec<T>::size() const { return data.size(); }

  template <typename T>
  IL void vec<T>::print()
  {
    std::cout << "[ ";
    for (size_t i = 0; i < size(); ++i)
    {
      std::cout << data[i] << " ";
    }
    std::cout << "]\n";
  }

  template <typename T>
  IL T &vec<T>::operator[](size_t i)
  {
    if (i < 0 || i >= size())
    {
      throw std::out_of_range("Index out of range");
    }
    return data[i];
  }

  template <typename T>
  IL const T &vec<T>::operator[](size_t i) const
  {
    if (i < 0 || i >= size())
    {
      throw std::out_of_range("Index out of range");
    }
    return data[i];
  }

  template <typename T>
  IL vec<T> vec<T>::multiply(const T x)
  {
    vec<T> result(size());
    for (size_t i = 0; i < size(); ++i)
    {
      result[i] = data[i] * x;
    }
    return result;
  }

  template <typename T>
  template <typename Y>
  IL vec<T> vec<T>::multiply(const vec<Y> &x)
  {
    if (size() != x.size())
    {
      throw std::invalid_argument("Vector sizes do not match for multiplication");
    }
    vec<T> result(size());
    for (size_t i = 0; i < size(); ++i)
    {
      result[i] = data[i] * static_cast<T>(x[i]);
    }
    return result;
  }

  template <typename T>
  IL vec<T> vec<T>::add(const T x)
  {
    vec<T> result(size());
    for (size_t i = 0; i < size(); ++i)
    {
      result[i] = data[i] + x;
    }
    return result;
  }

  template <typename T>
  template <typename Y>
  IL vec<T> vec<T>::add(const vec<Y> &x)
  {
    if (size() != x.size())
    {
      throw std::invalid_argument("Vector sizes do not match for addition");
    }
    vec<T> result(size());
    for (size_t i = 0; i < size(); ++i)
    {
      result[i] = data[i] + static_cast<T>(x[i]);
    }
    return result;
  }

  template <typename T>
  ST T lerp(const T &start, const T &end, double t)
  {
    static_assert(std::is_arithmetic<T>::value, "T must be an arithmetic type");
    return start * (1.0 - t) + end * t;
  }

  template <typename T>
  template <typename Y>
  IL vec<T> vec<T>::subtract(const vec<Y> &x)
  {
    if (size() != x.size())
    {
      throw std::invalid_argument("Vector sizes do not match for subtraction");
    }
    vec<T> result(size());
    for (size_t i = 0; i < size(); ++i)
    {
      result[i] = data[i] - static_cast<T>(x[i]);
    }
    return result;
  }

} // namespace utl

#endif // L_GEBRA_IMPLEMENTATION
