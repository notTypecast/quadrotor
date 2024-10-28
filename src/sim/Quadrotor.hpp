#ifndef PQ_QUADROTOR_HPP
#define PQ_QUADROTOR_HPP

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "src/params.hpp"

namespace quadrotor
{
    namespace sim
    {
        Eigen::Matrix4d get_quat_L(Eigen::Quaterniond &q)
        {
            Eigen::Matrix4d L;
            L << q.w(), -q.x(), -q.y(), -q.z(),
                q.x(), q.w(), -q.z(), q.y(),
                q.y(), q.z(), q.w(), -q.x(),
                q.z(), -q.y(), q.x(), q.w();
            return L;
        }

        class Quadrotor
        {
        public:
            Quadrotor(double mass, Eigen::Matrix3d inertia, double length,
                      double Kf = quadrotor::Value::Constant::Kf, double Kt = quadrotor::Value::Constant::Kt) : _m(mass),
                                                                                                                _I(inertia),
                                                                                                                _I_inv(_I.inverse()),
                                                                                                                _l(length),
                                                                                                                _Kf(Kf),
                                                                                                                _Kt(Kt),
                                                                                                                _position(Eigen::Vector3d::Zero()),
                                                                                                                _velocity(Eigen::Vector3d::Zero()),
                                                                                                                _orientation(Eigen::Quaterniond::Identity()),
                                                                                                                _angular_velocity(Eigen::Vector3d::Zero()),
                                                                                                                _Kf_matrix((Eigen::Matrix<double, 3, 4>() << 0, 0, 0, 0,
                                                                                                                            0, 0, 0, 0,
                                                                                                                            _Kf, _Kf, _Kf, _Kf)
                                                                                                                               .finished()),
                                                                                                                _lKf(_l * _Kf)

            {
            }

            void update(Eigen::Vector4d &controls, double dt)
            {
                Eigen::Vector3d thrust, torque;

                Eigen::Matrix3d R = _orientation.toRotationMatrix();

                thrust = R.transpose() * _m * _gravity + _Kf_matrix * controls;
                torque = (Eigen::Vector3d() << _lKf * (controls[1] - controls[3]),
                          _lKf * (controls[2] - controls[0]),
                          _Kt * (controls[0] - controls[1] + controls[2] - controls[3]))
                             .finished();

                _last_ddq.segment(0, 3) = thrust / _m - _angular_velocity.cross(_velocity);
                _last_ddq.segment(3, 3) = _I_inv * (torque - _angular_velocity.cross(_I * _angular_velocity));

                // Semi-implicit Euler integration
                _velocity += _last_ddq.segment(0, 3) * dt;
                _angular_velocity += _last_ddq.segment(3, 3) * dt;

                Eigen::Vector3d velocity_w = R * _velocity;
                Eigen::Vector4d quat_d = 0.5 * get_quat_L(_orientation) * quadrotor::Value::Constant::H_mat * _angular_velocity;

                _position += velocity_w * dt;
                _orientation.w() += quat_d[0] * dt;
                _orientation.vec() += quat_d.segment(1, 3) * dt;
                _orientation.normalize();
            }

            Eigen::Vector3d get_position()
            {
                return _position;
            }

            Eigen::Quaterniond get_orientation()
            {
                return _orientation;
            }

            Eigen::Vector3d get_velocity()
            {
                return _velocity;
            }

            Eigen::Vector3d get_angular_velocity()
            {
                return _angular_velocity;
            }

            Eigen::Vector<double, 13> get_state()
            {
                Eigen::Vector<double, 13> state;
                state << _position, _orientation.w(), _orientation.vec(), _velocity, _angular_velocity;
                return state;
            }

            Eigen::Vector<double, 6> get_last_ddq()
            {
                return _last_ddq;
            }

        protected:
            double _m, _l;
            double _Kf, _Kt;
            Eigen::Matrix3d _I, _I_inv;
            Eigen::Vector3d _position, _velocity;
            Eigen::Quaterniond _orientation;
            Eigen::Vector3d _angular_velocity;
            Eigen::Vector<double, 6> _last_ddq;

            const Eigen::Vector3d _gravity = Eigen::Vector3d(0, 0, -quadrotor::Value::Constant::g);
            Eigen::Matrix<double, 3, 4> _Kf_matrix;
            double _lKf;
        };
    }
}

#endif