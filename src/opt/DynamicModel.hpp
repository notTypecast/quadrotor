#ifndef PQ_DYNAMIC_MODEL_HPP
#define PQ_DYNAMIC_MODEL_HPP
#include <Eigen/Core>
#include <Eigen/Geometry>

#include "src/params.hpp"

namespace pq
{
Eigen::Vector3d dynamic_model_predict(const Eigen::Vector<double, 6> &state,
                                      const Eigen::Vector2d          &controls,
                                      const Eigen::Vector4d &model_params)
{
    // params: m [0], I [1], l [2], g [3]
    Eigen::Vector3d ddq;
    ddq[0] = -(controls[0] + controls[1]) * sin(state[2]) / model_params[0];
    ddq[1] = (controls[0] + controls[1]) * cos(state[2]) / model_params[0] -
             model_params[3];
    ddq[2] =
      (controls[1] - controls[0]) * model_params[2] / (2 * model_params[1]);
    return ddq;
}
}

namespace quadrotor
{
Eigen::Vector<double, 6> dynamic_model_predict(
  const Eigen::Vector<double, 13> &state, const Eigen::Vector4d &controls,
  const Eigen::Vector<double, 23> &model_params)
{
    // params: m [0], I [1-9], I_inv [10-18], l [19], Kf [20], Kt [21], g [22]
    Eigen::Matrix3d I     = Eigen::Matrix3d(model_params.segment(1, 9).data());
    Eigen::Matrix3d I_inv = Eigen::Matrix3d(model_params.segment(10, 9).data());

    Eigen::Matrix3d R =
      Eigen::Quaterniond(state[3], state[4], state[5], state[6])
        .toRotationMatrix();

    Eigen::Vector3d thrust = R.transpose() * model_params[0] *
                               Eigen::Vector3d(0, 0, -model_params[22]) +
                             (Eigen::Matrix<double, 3, 4>() << 0,
                              0,
                              0,
                              0,
                              0,
                              0,
                              0,
                              0,
                              model_params[20],
                              model_params[20],
                              model_params[20],
                              model_params[20])
                                 .finished() *
                               controls;
    double          lKf = model_params[19] * model_params[20];
    Eigen::Vector3d torque =
      (Eigen::Vector3d() << lKf * (controls[1] - controls[3]),
       lKf * (controls[2] - controls[0]),
       model_params[21] *
         (controls[0] - controls[1] + controls[2] - controls[3]))
        .finished();

    Eigen::Vector3d          vel     = state.segment(7, 3);
    Eigen::Vector3d          ang_vel = state.segment(10, 3);
    Eigen::Vector<double, 6> ddq;

    ddq.segment(0, 3) = thrust / model_params[0] - ang_vel.cross(vel);
    ddq.segment(3, 3) = I_inv * (torque - ang_vel.cross(I * ang_vel));

    return ddq;
}
}

#endif
