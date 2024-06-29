#include <chrono>
#include <cstddef>
#include <cstring>
#include <iostream>
#include <memory>
#include <thread>

#define L_GEBRA_IMPLEMENTATION
#include "./l_gebra.hpp"  // Assuming this is your external library header

// Buffer class
struct Buffer
{
    char *data;
    size_t width;
    size_t height;

    Buffer();
    Buffer(size_t width, size_t height);
    Buffer(size_t width, size_t height, char fill);
    Buffer(const Buffer &other);
    Buffer &operator=(const Buffer &other);
    ~Buffer();

    void set(utl::Vec<int, 2> point, char ch);
    char &operator()(size_t x, size_t y);
    const char &operator()(size_t x, size_t y) const;
};

// Renderer class
class Renderer
{
    std::shared_ptr<Buffer> _buffer;

public:
    Renderer();
    Renderer(size_t width, size_t height);
    Renderer(std::shared_ptr<Buffer> buffer);

    const Buffer &get_buffer() const;
    size_t get_width() const;
    size_t get_height() const;

    bool draw_point(utl::Vec<int, 2> point, char c);
    bool draw_line(utl::Vec<int, 2> start, utl::Vec<int, 2> end, char c);
    bool draw_triangle(utl::Vec<int, 2> a, utl::Vec<int, 2> b, utl::Vec<int, 2> c, char ch);
    bool draw_circle(utl::Vec<int, 2> center, int radius);
    void draw();
    void supersample_line(utl::Vec<int, 2> start, utl::Vec<int, 2> end, char c);
    bool default_char_line(utl::Vec<int, 2> start, utl::Vec<int, 2> end);
    bool anti_aliased_line(utl::Vec<int, 2> start, utl::Vec<int, 2> end, char c1, char c2);
    static std::shared_ptr<Buffer> create_buffer(size_t width, size_t height);
    void empty();
    void clear_screen();
    void reset_screen();
    void fill_buffer(char c);
    inline void sleep(int milliseconds);
};

// Buffer definitions
Buffer::Buffer() : data(nullptr), width(0), height(0) {}

Buffer::Buffer(size_t width, size_t height) : data(new char[width * height]), width(width), height(height)
{
    std::memset(data, ' ', width * height);  // Initialize buffer with spaces
}

Buffer::Buffer(size_t width, size_t height, char fill) : data(new char[width * height]), width(width), height(height)
{
    std::memset(data, fill, width * height);  // Initialize buffer with specified fill character
}

Buffer::Buffer(const Buffer &other) : data(new char[other.width * other.height]), width(other.width), height(other.height)
{
    std::memcpy(data, other.data, width * height);
}

Buffer &Buffer::operator=(const Buffer &other)
{
    if (this != &other)
    {
        delete[] data;
        width = other.width;
        height = other.height;
        data = new char[width * height];
        std::memcpy(data, other.data, width * height);
    }
    return *this;
}

void Buffer::set(utl::Vec<int, 2> point, char ch)
{
    size_t x = point.x();
    size_t y = point.y();
    if (x >= 0 && x < width && y >= 0 && y < height)
    {
        data[y * width + x] = ch;
    }
}

Buffer::~Buffer()
{
    // delete[] data;
}

char &Buffer::operator()(size_t x, size_t y) { return data[y * width + x]; }

const char &Buffer::operator()(size_t x, size_t y) const { return data[y * width + x]; }

// Renderer definitions
Renderer::Renderer() : _buffer(std::make_shared<Buffer>()) {}

Renderer::Renderer(size_t width, size_t height) : _buffer(std::make_shared<Buffer>(width, height)) {}

Renderer::Renderer(std::shared_ptr<Buffer> buffer) : _buffer(buffer) {}

const Buffer &Renderer::get_buffer() const { return *_buffer; }

size_t Renderer::get_width() const { return _buffer->width; }

size_t Renderer::get_height() const { return _buffer->height; }

bool Renderer::draw_point(utl::Vec<int, 2> point, char c)
{
    size_t x = point.x();
    size_t y = point.y();
    if (x >= 0 && x < static_cast<int>(_buffer->width) && y >= 0 && y < static_cast<int>(_buffer->height))
    {
        (*_buffer)(x, y) = c;
        return true;
    }
    return false;
}

bool Renderer::draw_line(utl::Vec<int, 2> start, utl::Vec<int, 2> end, char c)
{
    int x1 = start[0], y1 = start[1];
    int x2 = end[0], y2 = end[1];

    // Calculate the difference between the points
    int dx = std::abs(x2 - x1);
    int dy = std::abs(y2 - y1);

    // Determine the direction of the line
    int sx = (x1 < x2) ? 1 : -1;
    int sy = (y1 < y2) ? 1 : -1;

    int err = dx - dy;
    int e2;

    while (true)
    {
        // Plot the point
        if (x1 >= 0 && x1 < static_cast<int>(_buffer->width) && y1 >= 0 && y1 < static_cast<int>(_buffer->height))
        {
            (*_buffer)(x1, y1) = c;
        }

        // Check if the end point is reached
        if (x1 == x2 && y1 == y2) break;

        e2 = 2 * err;

        if (e2 > -dy)
        {
            err -= dy;
            x1 += sx;
        }

        if (e2 < dx)
        {
            err += dx;
            y1 += sy;
        }
    }

    return true;
}

bool Renderer::draw_triangle(utl::Vec<int, 2> a, utl::Vec<int, 2> b, utl::Vec<int, 2> c, char ch)
{
    supersample_line(a, b, ch);
    supersample_line(b, c, ch);
    supersample_line(c, a, ch);
    return true;
}

bool Renderer::draw_circle(utl::Vec<int, 2> center, int radius)
{
    // Terminal pixel aspect ratio (height / width)
    return true;
}

void Renderer::draw()
{
    for (size_t y = 0; y < _buffer->height; y++)
    {
        for (size_t x = 0; x < _buffer->width; x++)
        {
            std::cout << (*_buffer)(x, y);
        }
        std::cout << '\n';
    }
}

void Renderer::supersample_line(utl::Vec<int, 2> start, utl::Vec<int, 2> end, char c)
{
    Buffer high_res_buffer(_buffer->width * 2, _buffer->height * 2, ' ');
    Renderer high_res_renderer(std::make_shared<Buffer>(high_res_buffer));

    high_res_renderer.draw_line(start * 2, end * 2, c);

    for (size_t y = 0; y < _buffer->height; ++y)
    {
        for (size_t x = 0; x < _buffer->width; ++x)
        {
            char mergedChar = high_res_renderer.get_buffer()(x * 2, y * 2);
            (*_buffer)(x, y) = mergedChar;  // Simplified merge, can average characters
        }
    }
}

bool Renderer::default_char_line(utl::Vec<int, 2> start, utl::Vec<int, 2> end)
{
    int x1 = start[0], y1 = start[1];
    int x2 = end[0], y2 = end[1];

    if (x1 == x2)
    {
        // Vertical line
        for (int y = std::min(y1, y2); y <= std::max(y1, y2); ++y)
        {
            (*_buffer)(x1, y) = '|';
        }
    }
    else if (y1 == y2)
    {
        // Horizontal line
        for (int x = std::min(x1, x2); x <= std::max(x1, x2); ++x)
        {
            (*_buffer)(x, y1) = '-';
        }
    }
    else
    {
        // Diagonal line
        int dx = std::abs(x2 - x1);
        int dy = std::abs(y2 - y1);
        char lineChar = (dx > dy) ? '\\' : '/';  // Choose character based on the slope
        draw_line(start, end, lineChar);
    }
    return true;
}

bool Renderer::anti_aliased_line(utl::Vec<int, 2> start, utl::Vec<int, 2> end, char c1, char c2)
{
    int x1 = start[0], y1 = start[1];
    int x2 = end[0], y2 = end[1];

    bool steep = std::abs(y2 - y1) > std::abs(x2 - x1);
    if (steep)
    {
        std::swap(x1, y1);
        std::swap(x2, y2);
    }
    if (x1 > x2)
    {
        std::swap(x1, x2);
        std::swap(y1, y2);
    }

    int dx = x2 - x1;
    int dy = std::abs(y2 - y1);
    int err = dx / 2;
    int ystep = (y1 < y2) ? 1 : -1;
    int y = y1;

    for (int x = x1; x <= x2; x++)
    {
        if (steep)
        {
            (*_buffer)(y, x) = c1;  // Primary character
            if (y + 1 < static_cast<int>(_buffer->height))
            {
                (*_buffer)(y + 1, x) = c2;  // Secondary character for anti-aliasing
            }
        }
        else
        {
            (*_buffer)(x, y) = c1;  // Primary character
            if (x + 1 < static_cast<int>(_buffer->width))
            {
                (*_buffer)(x, y + 1) = c2;  // Secondary character for anti-aliasing
            }
        }
        err -= dy;
        if (err < 0)
        {
            y += ystep;
            err += dx;
        }
    }
    return true;
}

std::shared_ptr<Buffer> Renderer::create_buffer(size_t width, size_t height) { return std::make_shared<Buffer>(width, height); }

void Renderer::empty() { std::memset(_buffer->data, ' ', _buffer->width * _buffer->height); }

void Renderer::clear_screen() { std::cout << "\033[2J\033[1;1H"; }

void Renderer::reset_screen()
{
    clear_screen();
    std::cout << "\033[0;0H";
}

void Renderer::fill_buffer(char c) { std::memset(_buffer->data, c, _buffer->width * _buffer->height); }

inline void Renderer::sleep(int milliseconds) { std::this_thread::sleep_for(std::chrono::milliseconds(milliseconds)); }

int main()
{
    auto b = Renderer::create_buffer(40, 20);
    Renderer r(b);

    // Example usage: drawing a line
    r.draw_circle({20, 10}, 8);

    r.draw();
    return 0;
}
