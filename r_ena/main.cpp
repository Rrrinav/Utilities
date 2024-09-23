#include <cstddef>
#include <cstring>
#include <iostream>

class R_ena
{
  std::size_t size;
  std::size_t offset;
  char *_arena;

public:
  R_ena(std::size_t size = 1024) : size(size), offset(0) { _arena = new char[size]; }
  ~R_ena() { delete[] _arena; }
  size_t get_size() { return size; }
  void *allocate_raw(std::size_t bytes)
  {
    std::size_t alignedSize = (bytes + sizeof(void *) - 1) & ~(sizeof(void *) - 1);
    if (offset + alignedSize > size)
    {
      std::cerr << "[ ERROR ]: Not enough memory.\n";
      return nullptr;
    }
    void *ptr = _arena + offset;
    offset += alignedSize;
    return ptr;
  }

  template <typename T, typename... Targs>
  T *create_object(Targs &&...args)
  {
    void *memory = allocate_raw(sizeof(T));
    if (!memory)
      return nullptr;
    return new (memory) T(std::forward<Targs>(args)...);
  }

  void reset()
  {
    offset = 0;
    std::memset(_arena, 0, size);
  }
  void clear() { offset = 0; }

  std::size_t remaining_space() const { return size - offset; }
  std::size_t used_space() const { return offset; }

  bool contains(void *ptr) const
  {
    char *char_ptr = static_cast<char *>(ptr);  // Fix by casting
    return char_ptr >= _arena && char_ptr < _arena + size;
  }

  void resize(std::size_t new_size)
  {
    if (new_size <= size)
      return;
    char *new_arena = new char[new_size];
    std::memcpy(new_arena, _arena, size);
    delete[] _arena;
    _arena = new_arena;
    size = new_size;
  }
};

class MyClass
{
public:
  int value;

  MyClass(int val) : value(val) { std::cerr << "MyClass constructed with value: " << value << std::endl; }

  ~MyClass() { std::cerr << "MyClass destructed with value: " << value << std::endl; }
};

int main()
{
  R_ena arena(1024);  // Create an arena with 1024 bytes

  // Allocate object of MyClass
  MyClass *myObject = arena.create_object<MyClass>(42);

  // Allocate primitive type
  int *intObject = static_cast<int *>(arena.allocate_raw(sizeof(int)));
  *intObject = 100;

  // Check if both objects are in the arena
  if (arena.contains(myObject))
    std::cout << "myObject is part of the arena." << std::endl;

  if (arena.contains(intObject))
    std::cout << "intObject is part of the arena." << std::endl;

  std::cout << "Size of arena is: " << arena.get_size() << std::endl;

  std::cout << "Remaining space is: " << arena.remaining_space() << '\n';

  arena.resize(1050);
  std::cout << "Size of arena after reisze is: " << arena.get_size() << std::endl;
  std::cout << "Remaining space after resize is: " << arena.remaining_space() << '\n';
  return 0;
}
