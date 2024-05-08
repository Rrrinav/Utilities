#include <chrono>
#include <climits>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#define L_GEBRA_IMPLEMENTATION
#include <unistd.h>

#include "l_gebra.hpp"

#define ANSI_CLEAR_SCREEN "\033[2J\033[1;1H"  // Clear screen and move cursor to top-left corner
#define ANSI_HIDE_CURSOR "\033[?25l"          // Hide cursor
#define ANSI_SHOW_CURSOR "\033[?25h"          // Show cursor

void clearScreenAndHideCursor()
{
    std::cout << ANSI_CLEAR_SCREEN;  // Clear screen
    std::cout << ANSI_HIDE_CURSOR;   // Hide cursor
}

void showCursor()
{
    std::cout << ANSI_SHOW_CURSOR;  // Show cursor
}

enum Pixel
{
    BACK = 0,
    FORE = 1,
    COUNT = 2,
};

#define WIDTH 160
#define HEIGHT 80
#define RAD 8
static_assert(HEIGHT % 2 == 0, "Height must be divisible by two");
static Pixel pixels[HEIGHT * WIDTH];
#define FPS 30

void fill(Pixel p)
{
    Pixel *ptr = pixels;
    size_t size = WIDTH * HEIGHT;
    while (size-- > 0) *ptr++ = p;
}

void circle(utl::Vec<float, 2> center, int radius)
{
    utl::Vec<float, 2> begin = center + (-1 * radius);
    utl::Vec<float, 2> end = center + radius;

    for (int y = begin.y(); y < end.y(); ++y)
    {
        for (int x = begin.x(); x < end.x(); ++x)
        {
            if (center.distance(utl::Vec<int, 2>{x, y}) <= radius)
            {
                if (0 <= x && x < WIDTH && 0 <= y && y < HEIGHT)
                {
                    pixels[y * WIDTH + x] = Pixel::FORE;
                }
            }
        }
    }
}

void show()
{
    static char row[WIDTH];
    static char table[COUNT][Pixel::COUNT] = {{' ', '_'}, {'^', 'C'}};

    for (int y = 0; y < HEIGHT / 2; ++y)
    {
        for (int x = 0; x < WIDTH; ++x)
        {
            Pixel t = pixels[2 * y * WIDTH + x];
            Pixel b = pixels[(2 * y + 1) * WIDTH + x];
            row[x] = table[t][b];
        }
        fwrite(&row, WIDTH, 1, stdout);
        fputc('\n', stdout);
    }
}
#define GRAVITY 80.0f
#define DT (1.0f / FPS)

int main(void)
{
    clearScreenAndHideCursor();
    utl::Vec<float, 2> pos = {RAD, RAD};
    utl::Vec<float, 2> vel = {40.0f, 0.0f};
    utl::Vec<float, 2> gravity = {0.0f, GRAVITY};

    while (1)
    {
        system("clear");  // Clear screen

        vel = vel + (gravity * DT);
        pos = pos + (vel * DT);

        if (pos.y() + RAD > HEIGHT) {
            pos[1] = HEIGHT - RAD;
            vel[1] = vel[1] * (-0.7f);
        }
        if (pos.x() > WIDTH + RAD + 6)
        {
            pos = {RAD, RAD};
            vel = {40.0f, 0.0f};
        }
        fill(Pixel::BACK);
        circle(pos, RAD);
        show();
        usleep(1000000 / FPS);
    }

    showCursor();  // Show cursor after the program execution
    return 0;
}
