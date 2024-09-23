#define R_ENA_IMPLEMENTATION
#include <iostream>

#include "r_ena.hpp"

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
