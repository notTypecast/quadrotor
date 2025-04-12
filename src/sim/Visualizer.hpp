#ifndef PQ_VISUALIZER_CPP
#define PQ_VISUALIZER_CPP
#include <Eigen/Core>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <sys/ioctl.h>
#include <unistd.h>

#include "src/sim/PlanarQuadrotor.hpp"
#define SCALE     10.0
#define CELLBYTES 4

namespace pq
{
namespace sim
{
class Visualizer
{
  public:
    Visualizer(bool use_scale = false, bool increase_scale = false)
      : _use_scale(use_scale)
      , _increase_scale(increase_scale)
      , _scale(SCALE)

    {
        struct winsize w;
        ioctl(STDOUT_FILENO, TIOCGWINSZ, &w);
        _rows   = w.ws_row;
        _cols   = w.ws_col;
        _zero_x = _cols / 2;
        _zero_y = _rows / 2;
        _grid   = std::shared_ptr<char[]>(new char[_rows * _cols * CELLBYTES]);
    }

    void set_message(std::string msg)
    {
        if (msg.size() > _cols)
        {
            msg = msg.substr(0, _cols);
        }
        _message = msg;
    }

    void show(PlanarQuadrotor p, std::pair<int, int> target)
    {
        Eigen::Vector<double, 6> x = p.get_state();
        _clear_screen();

        int x_pos =
          (int)round((target.first / (_use_scale ? _scale : 1)) + _zero_x) %
          _cols;
        int y_pos =
          (int)round(-(target.second / (_use_scale ? _scale : 1)) + _zero_y) %
          _rows;
        _grid[(y_pos * _cols + x_pos) * CELLBYTES] = 'T';

        if (_increase_scale)
        {
            _scale += 0.2;
        }

        x_pos =
          (int)round((x[0] / (_use_scale ? _scale : 1)) + _zero_x) % _cols;
        y_pos =
          (int)round(-(x[1] / (_use_scale ? _scale : 1)) + _zero_y) % _rows;
        x_pos += (x_pos < 0 ? _cols : 0);
        y_pos += (y_pos < 0 ? _rows : 0);
        if (_message.size() && y_pos == _rows - 1)
        {
            --y_pos;
        }
        char *repr = _get_repr(x[2] / (_use_scale ? _scale : 1));
        for (int i = 0; i < CELLBYTES; ++i)
        {
            _grid[(y_pos * _cols + x_pos) * CELLBYTES + i] = repr[i];
        }
        if (_message.size())
        {
            for (int i = 0; i < _message.size(); ++i)
            {
                _grid[(_rows - 1) * _cols * CELLBYTES + i * CELLBYTES] =
                  _message[i];
            }
        }
        _write_to_screen();
    }

  private:
    int                     _zero_x, _zero_y;
    int                     _rows, _cols;
    std::shared_ptr<char[]> _grid;
    bool                    _use_scale, _increase_scale;
    double                  _scale;
    std::string             _message;

    std::map<std::pair<double, double>, std::string> _angle_map = {
        {    { 0.0, 22.5 }, "↑" },
        {   { 22.5, 67.5 }, "↖" },
        {  { 67.5, 112.5 }, "←" },
        { { 112.5, 157.5 }, "↙" },
        { { 157.5, 202.5 }, "↓" },
        { { 202.5, 247.5 }, "↘" },
        { { 247.5, 292.5 }, "→" },
        { { 292.5, 337.5 }, "↗" }
    };

    char *_get_repr(double angle)
    {
        angle = fmod(angle, 2 * M_PI);
        if (angle < 0)
        {
            angle += 2 * M_PI;
        }
        angle = angle * 180 / M_PI;

        for (const auto &pair : _angle_map)
        {
            if (angle >= pair.first.first && angle < pair.first.second)
            {
                return (char *)pair.second.data();
            }
        }

        return (char *)"↑";
    }

    void _clear_screen()
    {
        for (int i = 0; i < _rows * _cols * CELLBYTES; i += 4)
        {
            _grid[i] = ' ';
            for (int j = 1; j < 3; ++j)
            {
                _grid[i + j] = 0;
            }
        }
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-result"
        system("clear");
#pragma GCC diagnostic pop
    }

    void _write_to_screen()
    {
        ulong   total_to_write = _rows * _cols * CELLBYTES * sizeof(char);
        ssize_t written        = 0;
        ssize_t count;
        do {
            count =
              write(STDOUT_FILENO, &_grid[written], total_to_write - written);
            if (count != -1)
            {
                written += count;
            }
        } while (written != total_to_write);
        fflush(stdout);
    }
};
}
}

#endif