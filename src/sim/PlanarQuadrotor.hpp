#ifndef PQ_PLANAR_QUADROTOR_CPP
#define PQ_PLANAR_QUADROTOR_CPP
#include <Eigen/Core>

#include "src/params.hpp"

namespace pq
{
namespace sim
{
class PlanarQuadrotor
{
  public:
    PlanarQuadrotor(double mass, double moment_of_inertia, double length)
      : _m(mass)
      , _I(moment_of_inertia)
      , _l(length)
      , _x(Eigen::Vector<double, 6>::Zero())
    {
    }

    void update(Eigen::Vector2d controls, double dt)
    {
        _last_ddq[0] = -(controls[0] + controls[1]) * sin(_x[2]) / _m;
        _last_ddq[1] = (controls[0] + controls[1]) * cos(_x[2]) / _m -
                       pq::Value::Constant::g;
        _last_ddq[2] = (controls[1] - controls[0]) * _l / (2 * _I);

        _x.segment(0, 3) += _x.segment(3, 3) * dt + 0.5 * _last_ddq * dt * dt;
        _x.segment(3, 3) += _last_ddq * dt;
        _total           += dt;
    }

    Eigen::Vector<double, 6> get_state()
    {
        return _x;
    }

    double get_sim_time()
    {
        return _total;
    }

    Eigen::Vector3d get_last_ddq()
    {
        return _last_ddq;
    }

  private:
    double                   _m, _I, _l;
    Eigen::Vector<double, 6> _x;
    double                   _total = 0.0;
    Eigen::Vector3d          _last_ddq;
};
}
}

#endif