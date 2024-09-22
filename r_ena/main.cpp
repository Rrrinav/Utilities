#include <cstddef>
#include <iostream>

namespace utl
{
  class r_ena
  {
    std::size_t size;
    std::size_t offset;
    char *_arena;

  public:
    r_ena(std::size_t size = 1024) : size(size), offset(0) { _arena = new char[size]; }

    ~r_ena() { delete[] _arena; }

    void *allocate(std::size_t bytes)
    {
      std::size_t alignedSize = (bytes + sizeof(void *) - 1) & ~(sizeof(void *) - 1);

      if (offset + alignedSize > size)
        return nullptr;  // Not enough memory

      void *ptr = _arena + offset;
      offset += alignedSize;
      return ptr;
    }

    void reset()
    {
      offset = 0;  // Reset the offset for reuse
    }
  };
}  // namespace utl

class MyClass
{
public:
  int value;

  MyClass(int val) : value(val) { std::cout << "MyClass constructed with value: " << value << std::endl; }

  ~MyClass() { std::cout << "MyClass destructed with value: " << value << std::endl; }
};

int main()
{
  utl::r_ena arena(1024);  // Create an arena with 1024 bytes

  // Allocate memory for an instance of MyClass
  MyClass *myObject = static_cast<MyClass *>(arena.allocate(sizeof(MyClass)));
  if (myObject)
  {
    // Construct the object using placement new
    new (myObject) MyClass(42);

    // Use the object
    std::cout << "MyClass value: " << myObject->value << std::endl;

    // Manually call the destructor
    myObject->~MyClass();
  }
  else
  {
    std::cout << "Failed to allocate memory for MyClass." << std::endl;
  }

  // Reset the arena for reuse
  arena.reset();

  return 0;
}
