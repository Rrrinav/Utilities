#include <cstddef>
#include <cstring>
#include <iostream>

// Arena allocator
namespace utl
{
  class R_ena
  {
    std::size_t size;    // > Size of the arena
    std::size_t offset;  // > Offset to the next allocation
    char *_arena;        // > Pointer to the beginning of arena

  public:
    R_ena(std::size_t size = 1024) : size(size), offset(0) { _arena = new char[size]; }

    ~R_ena() { delete[] _arena; }

    void *allocate_raw(std::size_t bytes)
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
    void reset()
    {
      offset = 0;
      memset(_arena, 0, size);
    }

    void deallocate_whole()
    {
      offset = 0;
      delete[] _arena;
    }

    template <typename T, typename... Targs>
    T *create_object(Targs &&...Fargs)
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
    void destroy_object(T *ptr)
    {
      if (ptr)
        ptr->~T();  // Explicitly call destructor
    }
  };
}  // namespace utl

class MyClass
{
public:
  int value;

  MyClass(int val) : value(val) { std::cerr << "MyClass constructed with value: " << value << std::endl; }

  ~MyClass() { std::cerr << "MyClass destructed with value: " << value << std::endl; }
};

int main()
{
  utl::R_ena arena(1024);  // Create an arena with 1024 bytes

  // Allocate memory for an instance of MyClass
  MyClass *myObject = arena.create_object<MyClass>(42);
  if (myObject)
  {
    // Use the object
    std::cout << "MyClass value: " << myObject->value << std::endl;
  }
  else
  {
    std::cout << "Failed to allocate memory for MyClass." << std::endl;
  }

  // Reset the arena for reuse
  arena.reset();

  return 0;
}
