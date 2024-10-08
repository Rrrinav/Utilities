/*
 * MIT License
 * Copyright (c) 2024 Rinav <rinavhansa4@gmail.com>
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
--------------------------------| HOW TO USE? |---------------------------------------
| You just need to include this header file in your project and use the vec          |
| class with namespace utl. You also need macro R_ENA_IMPLEMENTATION before        |
| "#include "r_ena.hpp" in one of your source files to include the                    |
| implementation. Basically :-                                                       |
| #define R_ENA_IMPLEMENTATION                                                       |
| #include "r_ena.hpp"                                                               |
--------------------------------------------------------------------------------------
*/

//===============================================================================
//  [ DECLARATIONS ] : Includes interface declarations!
//===============================================================================

#include <cstddef>
#include <cstring>

namespace utl
{
  /*
   * @brief Arena allocator
   * @details Arena allocator is a simple memory allocator that allocates memory from a fixed-size buffer.
   * It is useful when you need to allocate a lot of small objects that are short-lived.
   * The arena allocator is fast because it doesn't need to manage the memory allocation and deallocation.
   * @note The arena allocator is not suitable for long-lived objects because it doesn't free memory until the arena is destroyed.
   */
  class R_ena
  {
    std::size_t size;    // > Size of the arena
    std::size_t offset;  // > Offset to the next allocation
    char *_arena;        // > Pointer to the beginning of arena

  public:
    /*
     * @brief Constructor
     * @param size Size of the arena
     * @details Constructor that initializes the arena with a fixed size.
     * The size is the total size of the arena in bytes.
     */
    R_ena(std::size_t size = 1024) : size(size), offset(0) { _arena = new char[size]; }

    /*
    * @brief Destructor
    * @details Destructor that frees the memory allocated for the arena.
    */
    ~R_ena() { delete[] _arena; }

    /*
     * @brief Get the size of arena
     * @return (size_t) size of the arena
     */
    size_t get_size() const { return size; }

    /*
     * @brief Allocate memory
     * @param bytes Number of bytes to allocate
     * @return (void) Pointer to the allocated memory
     */
    void *allocate_raw(std::size_t bytes);

    /*
    * @brief Reset the arena
    * @details Reset the offset to 0 and clear the memory
    */
    void reset();

    /*
     * @brief Deallocate memory
     * @details Deallocate memory by setting the offset to 0 & size to 0 and deleting the arena
     */
    void deallocate_whole();

    /*
     * @brief Allocate object of type T in Arena
     * @tparam T Type of object to allocate
     * @tparam Args Types of arguments to pass to the constructor
     * @param args Arguments to pass to the constructor
     * @return T* Pointer to the allocated object
     * @details Allocate an object of type T in the arena and construct it using placement new.
     * */
    template <typename T, typename... Targs>
    T *create_object(Targs &&...Fargs);

    /*
     * @brief Manually destroy an object by calling its destructor
     * @param T* ptr Pointer to the object to destroy
     * @details Call the destructor of an object without deallocating its memory.
     *          This is only needed if you want to manage object lifetimes explicitly.
    */
    template <typename T>
    void destroy_object(T *ptr);

    /*
     * @brief allocate memory
     * @return (size_t) remaining space in the arena
     */
    size_t remaining_space();

    /*
     * @brief Allocate memory
     * @return (bool) wether a pointer is in the arena or not
     */
    bool contains(void *ptr);

    /*
     * @brief allocate memory
     * @return (size_t) used space in the arena
     */
    size_t used_space();

    /*
     * @brief Increase size of arena
     * @param (size_t) new_size New size of the arena
     * @return (bool) wether the arena was resized or not, generally false if new_size is less than or equal to the current size.
     * @details Adds more memory to the arena if the new size is greater than the current size.
     */
    bool resize(std::size_t new_size);

    void print_state() const;
  };
}  // namespace utl

#ifdef R_ENA_IMPLEMENTATION

#include <iostream>

void *utl::R_ena::allocate_raw(std::size_t bytes)
{
  // Align the sized based on architecture using [ sizeof(void *) ]
  std::size_t alignedSize = (bytes + sizeof(void *) - 1) & ~(sizeof(void *) - 1);

  // Check if we have enough memory
  if (offset + alignedSize > size)
  {
    std::cerr << "[ ERROR ]: Not enough memory, returning a nullptr\n";
    return nullptr;  // Not enough memory
  }
  // Return the pointer to the memory and increment the offset
  void *ptr = _arena + offset;
  offset += alignedSize;
  return ptr;
}

void utl::R_ena::reset()
{
  offset = 0;
  memset(_arena, 0, size);
}

void utl::R_ena::deallocate_whole()
{
  offset = 0;
  delete[] _arena;
}

template <typename T, typename... Targs>
T *utl::R_ena::create_object(Targs &&...Fargs)
{
  void *memory = allocate_raw(sizeof(T));
  if (!memory)
  {
    std::cerr << "[ ERROR ]: Some problem allocating memory, returning a nullptr\n";
    return nullptr;
  }
  return new (memory) T(std::forward<Targs>(Fargs)...);
}

template <typename T>
void utl::R_ena::destroy_object(T *ptr)
{
  if (ptr)
    ptr->~T();  // Explicitly call destructor
}

size_t utl::R_ena::remaining_space() { return size - offset; }

bool utl::R_ena::contains(void *ptr)
{
  char *char_ptr = static_cast<char *>(ptr);  // Cast to char* for comparison
  return char_ptr >= _arena && char_ptr < _arena + size;
}

size_t utl::R_ena::used_space() { return offset; }

bool utl::R_ena::resize(std::size_t new_size)
{
  if (new_size <= size)
    return false;
  char *new_arena = new char[new_size];
  std::memcpy(new_arena, _arena, size);
  delete[] _arena;
  _arena = new_arena;
  size = new_size;
  return true;
}

void utl::R_ena::print_state() const
{
  std::cout << "[ STATE ]: Arena Size: " << size << ", Used Memory: " << offset << ", Available Memory: " << get_available_memory() << '\n';
}

#endif  // R_ENA_IMPLEMENTATION
