#ifndef PQ_NUM_OPTIMIZER_HPP
#define PQ_NUM_OPTIMIZER_HPP

#include <Eigen/Geometry>
#include <iostream>

#include <casadi/casadi.hpp>

#include "src/params.hpp"
#include "src/opt/Optimizer.hpp"
#include "src/opt/num-opt/SymNN.hpp"

using namespace casadi;

namespace pq
{
    namespace num_opt
    {
        struct Params
        {
            int horizon = pq::Value::Param::NumOpt::horizon;
            double dt = pq::Value::Param::NumOpt::dt;
            double m = pq::Value::Constant::mass;
            double I = pq::Value::Constant::inertia;
            double l = pq::Value::Constant::length;
            double g = pq::Value::Constant::g;
            Eigen::Vector<double, 6> init;
            Eigen::Vector<double, 6> target;
        };

        class NumOptimizer : public pq::Optimizer
        {
        public:
            NumOptimizer(Params params) : _H(params.horizon),
                                          _dt(params.dt),
                                          _m(params.m),
                                          _I(params.I),
                                          _l(params.l),
                                          _g(params.g),
                                          _prev_result(DM::zeros(2, _H))
            {
                _model_params = Eigen::VectorXd::Zero(4);
                _model_params << params.m, params.I, params.l, params.g;

                _setup(params.init, params.target);
            }

            Eigen::MatrixXd solve()
            {
                OptiSol sol = _opti.solve();

                _prev_result = sol.value(_F);

                Eigen::MatrixXd forces(2, _H);

                for (int i = 0; i < _H; ++i)
                {
                    forces(0, i) = static_cast<double>(_prev_result(0, i));
                    forces(1, i) = static_cast<double>(_prev_result(1, i));
                }

                return forces;
            }

            virtual void reinit() {}

            virtual Eigen::VectorXd next(const Eigen::VectorXd &init, const Eigen::VectorXd &target)
            {
                /*
                if (_offset == _H)
                {
                    _setup(init, target);
                    _offset = -1;
                }
                if (_offset == -1)
                {
                    _offset = 0;
                    _forces = solve();
                }

                return _forces.col(_offset++);
                */
                _setup(init, target);
                return solve().col(0);
            }

            virtual Eigen::VectorXd model_params()
            {
                return _model_params;
            }

        protected:
            int _H;
            double _dt;
            double _m, _I, _l;
            double _g;
            Opti _opti;
            MX _x, _u, _a;
            MX _F;

            int _offset = -1;

            Eigen::VectorXd _model_params;

            DM _prev_result;

            void _setup(const Eigen::VectorXd &init, const Eigen::VectorXd &target)
            {
                _opti = Opti();
                _x = _opti.variable(3, _H + 1);
                _u = _opti.variable(3, _H + 1);
                _a = _opti.variable(3, _H);
                _F = _opti.variable(2, _H);

                DM init_x_dm = DM::vertcat({init[0], init[1], init[2]});
                DM target_x_dm = DM::vertcat({target[0], target[1], target[2]});

                DM init_u_dm = DM::vertcat({init[3], init[4], init[5]});
                DM target_u_dm = DM::vertcat({target[3], target[4], target[5]});

                _opti.minimize(4 * sumsqr(_x - target_x_dm) + sumsqr(_u));

                _opti.subject_to(0 <= _F <= pq::Value::Param::NumOpt::F_max);

                _opti.subject_to(_x(Slice(), 0) == init_x_dm);
                // _opti.subject_to(_x(Slice(), _H) == target_dm);
                _opti.subject_to(_u(Slice(), 0) == init_u_dm);
                // _opti.subject_to(_u(Slice(), _H) == 0);

                if (pq::Value::Param::NumOpt::use_learned)
                {
                    for (int i = 0; i < _H; ++i)
                    {
                        MX state = vertcat(_x(Slice(), i), _u(Slice(), i), _F(Slice(), i));
                        MX l = pq::Value::Param::SymNN::learned_model->forward(state);

                        _opti.subject_to(_a(0, i) == -(_F(0, i) + _F(1, i)) * sin(_x(2, i)) / _m + l(0));
                        _opti.subject_to(_a(1, i) == (_F(0, i) + _F(1, i)) * cos(_x(2, i)) / _m - _g + l(1));
                        _opti.subject_to(_a(2, i) == (_F(1, i) - _F(0, i)) * _l / (2 * _I) + l(2));
                    }
                }
                else
                {
                    for (int i = 0; i < _H; ++i)
                    {
                        _opti.subject_to(_a(0, i) == -(_F(0, i) + _F(1, i)) * sin(_x(2, i)) / _m);
                        _opti.subject_to(_a(1, i) == (_F(0, i) + _F(1, i)) * cos(_x(2, i)) / _m - _g);
                        _opti.subject_to(_a(2, i) == (_F(1, i) - _F(0, i)) * _l / (2 * _I));
                    }
                }

                for (int i = 1; i < _H + 1; ++i)
                {
                    _opti.subject_to(_x(Slice(), i) == _x(Slice(), i - 1) + _u(Slice(), i - 1) * _dt + 0.5 * _a(Slice(), i - 1) * _dt * _dt);
                    _opti.subject_to(_u(Slice(), i) == _u(Slice(), i - 1) + _a(Slice(), i - 1) * _dt);
                }

                for (int i = 0; i < pq::Value::Param::NumOpt::prev_steps_init; ++i)
                {
                    _opti.set_initial(_F(Slice(), i), _prev_result(Slice(), i + 1));
                }

                Dict opts;
                opts["ipopt.print_level"] = 0;
                opts["print_time"] = false;
                opts["ipopt.tol"] = 1e-3;

                _opti.solver("ipopt", opts);
            }
        };
    }
}

namespace quadrotor
{
    namespace num_opt
    {
        struct Params
        {
            int horizon = quadrotor::Value::Param::NumOpt::horizon;
            double dt = quadrotor::Value::Param::NumOpt::dt;
            double m = quadrotor::Value::Constant::mass;
            double l = quadrotor::Value::Constant::length;
            Eigen::Matrix3d I = quadrotor::Value::Constant::I;
            double g = quadrotor::Value::Constant::g;
            double Kf = quadrotor::Value::Constant::Kf;
            double Kt = quadrotor::Value::Constant::Kt;

            Eigen::Vector<double, 13> init;
            Eigen::Vector<double, 13> target;
        };

        class NumOptimizer : public pq::Optimizer
        {
        public:
            NumOptimizer(Params params) : _H(params.horizon),
                                          _dt(params.dt),
                                          _m(params.m),
                                          _l(params.l),
                                          _I(params.I),
                                          _I_inv(_I.inverse()),
                                          _g(params.g),
                                          _Kf(params.Kf),
                                          _Kt(params.Kt),
                                          _prev_result(DM::zeros(4, _H))
            {
                _model_params = Eigen::Vector<double, 23>();
                _model_params[0] = params.m;
                _model_params.segment(1, 9) = Eigen::Map<Eigen::VectorXd>(params.I.data(), 9);
                _model_params.segment(10, 9) = Eigen::Map<Eigen::VectorXd>(_I_inv.data(), 9);
                _model_params[19] = params.l;
                _model_params[20] = params.Kf;
                _model_params[21] = params.Kt;
                _model_params[22] = params.g;

                _setup(params.init, params.target);
            }

            Eigen::MatrixXd solve()
            {
                OptiSol sol = _opti.solve();
                _prev_result = sol.value(_c);
                _prev = true;

                Eigen::MatrixXd C(4, _H);

                for (int i = 0; i < _H; ++i)
                {
                    C(0, i) = static_cast<double>(_prev_result(0, i));
                    C(1, i) = static_cast<double>(_prev_result(1, i));
                    C(2, i) = static_cast<double>(_prev_result(2, i));
                    C(3, i) = static_cast<double>(_prev_result(3, i));
                }
                /*
                std::cout << _opti.debug().value(_x) << std::endl;
                std::cout << _opti.debug().value(_u) << std::endl;
                std::cout << _opti.debug().value(_a) << std::endl;
                std::cout << _opti.debug().value(_F) << std::endl;
                std::cout << _opti.debug().value(_c) << std::endl;
                */
                return C;
            }

            virtual void reinit() {
                _prev = false;
            }

            virtual Eigen::VectorXd next(const Eigen::VectorXd &init, const Eigen::VectorXd &target)
            {
                _setup(init, target);
                
                Eigen::VectorXd res = solve().col(0);
                return res;
            }

            virtual Eigen::VectorXd model_params()
            {
                return _model_params;
            }

        protected:
            int _H;
            double _dt;
            double _m, _l;
            Eigen::Matrix3d _I;
            Eigen::Matrix3d _I_inv;
            double _g;
            double _Kf, _Kt;

            Opti _opti;
            MX _x, _u, _a, _F;
            MX _c;

            Eigen::VectorXd _model_params;

            bool _prev = false;
            DM _prev_result;

            void _setup(const Eigen::VectorXd &init, const Eigen::VectorXd &target)
            {
                _opti = Opti();
                _x = _opti.variable(7, _H + 1);
                _u = _opti.variable(6, _H + 1);
                _a = _opti.variable(6, _H);
                _F = _opti.variable(6, _H);
                _c = _opti.variable(4, _H);

                DM init_x_dm = DM::vertcat({init[0], init[1], init[2], init[3], init[4], init[5], init[6]});
                DM target_x_dm = DM::vertcat({target[0], target[1], target[2], target[3], target[4], target[5], target[6]});

                DM init_u_dm = DM::vertcat({init[7], init[8], init[9], init[10], init[11], init[12]});
                DM target_u_dm = DM::vertcat({target[7], target[8], target[9], target[10], target[11], target[12]});

                // _opti.minimize(8 * sumsqr(_x - target_x_dm) + sumsqr(_u - target_u_dm));
                _opti.minimize(8 * sumsqr(_x(Slice(0, 3), Slice()) - target_x_dm(Slice(0, 3), Slice())) + 2 * sumsqr(_x(Slice(3, 4), Slice()) - target_x_dm(Slice(3, 4), Slice())) + sumsqr(_u - target_u_dm));

                _opti.subject_to(0 <= _c <= quadrotor::Value::Param::NumOpt::control_max);

                _opti.subject_to(_x(Slice(), 0) == init_x_dm);
                //_opti.subject_to(sumsqr(_x(Slice(0, 3), _H) - target_x_dm(Slice(0, 3))) <= 1e-6);
                _opti.subject_to(_u(Slice(), 0) == init_u_dm);
                //_opti.subject_to(_u(Slice(), _H) == target_u_dm);

                if (quadrotor::Value::Param::NumOpt::use_learned)
                {
                    for (int i = 0; i < _H; ++i)
                    {
                        MX state = vertcat(_x(Slice(), i), _u(Slice(), i), _c(Slice(), i));
                        MX l = quadrotor::Value::Param::SymNN::learned_model->forward(state);

                        // THRUST
                        _opti.subject_to(_F(0, i) == -2 * (_x(4, i) * _x(6, i) - _x(3, i) * _x(5, i)) * _m * _g);
                        _opti.subject_to(_F(1, i) == -2 * (_x(5, i) * _x(6, i) + _x(3, i) * _x(4, i)) * _m * _g);
                        _opti.subject_to(_F(2, i) == -2 * (_x(3, i) * _x(3, i) + _x(6, i) * _x(6, i)) * _m * _g + _m * _g + _Kf * (_c(0, i) + _c(1, i) + _c(2, i) + _c(3, i)));
                        // TORQUE
                        _opti.subject_to(_F(3, i) == _l * _Kf * (_c(1, i) - _c(3, i)));
                        _opti.subject_to(_F(4, i) == _l * _Kf * (_c(2, i) - _c(0, i)));
                        _opti.subject_to(_F(5, i) == _Kt * (_c(0, i) - _c(1, i) + _c(2, i) - _c(3, i)));
                        // LINEAR ACCELERATION
                        _opti.subject_to(_a(0, i) == _F(0, i) / _m - (_u(4, i) * _u(2, i) - _u(5, i) * _u(1, i)) + l(0));
                        _opti.subject_to(_a(1, i) == _F(1, i) / _m - (_u(5, i) * _u(0, i) - _u(3, i) * _u(2, i)) + l(1));
                        _opti.subject_to(_a(2, i) == _F(2, i) / _m - (_u(3, i) * _u(1, i) - _u(4, i) * _u(0, i)) + l(2));
                        // ANGULAR ACCELERATION
                        // [wix, wiy, wiz] = inertia * angular_velocity
                        MX wix = _I(0, 0) * _u(3, i) + _I(0, 1) * _u(4, i) + _I(0, 2) * _u(5, i);
                        MX wiy = _I(1, 0) * _u(3, i) + _I(1, 1) * _u(4, i) + _I(1, 2) * _u(5, i);
                        MX wiz = _I(2, 0) * _u(3, i) + _I(2, 1) * _u(4, i) + _I(2, 2) * _u(5, i);

                        // [fx, fy, fz] = torque - angular_velocity.cross(inertia * angular_velocity)
                        MX fx = _F(3, i) - (_u(4, i) * wiz - _u(5, i) * wiy);
                        MX fy = _F(4, i) - (_u(5, i) * wix - _u(3, i) * wiz);
                        MX fz = _F(5, i) - (_u(3, i) * wiy - _u(4, i) * wix);

                        _opti.subject_to(_a(3, i) == _I_inv(0, 0) * fx + _I_inv(0, 1) * fy + _I_inv(0, 2) * fz + l(3));
                        _opti.subject_to(_a(4, i) == _I_inv(1, 0) * fx + _I_inv(1, 1) * fy + _I_inv(1, 2) * fz + l(4));
                        _opti.subject_to(_a(5, i) == _I_inv(2, 0) * fx + _I_inv(2, 1) * fy + _I_inv(2, 2) * fz + l(5));
                    }
                }
                else
                {
                    for (int i = 0; i < _H; ++i)
                    {
                        // THRUST
                        _opti.subject_to(_F(0, i) == -2 * (_x(4, i) * _x(6, i) - _x(3, i) * _x(5, i)) * _m * _g);
                        _opti.subject_to(_F(1, i) == -2 * (_x(5, i) * _x(6, i) + _x(3, i) * _x(4, i)) * _m * _g);
                        _opti.subject_to(_F(2, i) == -2 * (_x(3, i) * _x(3, i) + _x(6, i) * _x(6, i)) * _m * _g + _m * _g + _Kf * (_c(0, i) + _c(1, i) + _c(2, i) + _c(3, i)));
                        // TORQUE
                        _opti.subject_to(_F(3, i) == _l * _Kf * (_c(1, i) - _c(3, i)));
                        _opti.subject_to(_F(4, i) == _l * _Kf * (_c(2, i) - _c(0, i)));
                        _opti.subject_to(_F(5, i) == _Kt * (_c(0, i) - _c(1, i) + _c(2, i) - _c(3, i)));
                        // LINEAR ACCELERATION
                        _opti.subject_to(_a(0, i) == _F(0, i) / _m - (_u(4, i) * _u(2, i) - _u(5, i) * _u(1, i)));
                        _opti.subject_to(_a(1, i) == _F(1, i) / _m - (_u(5, i) * _u(0, i) - _u(3, i) * _u(2, i)));
                        _opti.subject_to(_a(2, i) == _F(2, i) / _m - (_u(3, i) * _u(1, i) - _u(4, i) * _u(0, i)));
                        // ANGULAR ACCELERATION
                        // [wix, wiy, wiz] = inertia * angular_velocity
                        MX wix = _I(0, 0) * _u(3, i) + _I(0, 1) * _u(4, i) + _I(0, 2) * _u(5, i);
                        MX wiy = _I(1, 0) * _u(3, i) + _I(1, 1) * _u(4, i) + _I(1, 2) * _u(5, i);
                        MX wiz = _I(2, 0) * _u(3, i) + _I(2, 1) * _u(4, i) + _I(2, 2) * _u(5, i);

                        // [fx, fy, fz] = torque - angular_velocity.cross(inertia * angular_velocity)
                        MX fx = _F(3, i) - (_u(4, i) * wiz - _u(5, i) * wiy);
                        MX fy = _F(4, i) - (_u(5, i) * wix - _u(3, i) * wiz);
                        MX fz = _F(5, i) - (_u(3, i) * wiy - _u(4, i) * wix);

                        _opti.subject_to(_a(3, i) == _I_inv(0, 0) * fx + _I_inv(0, 1) * fy + _I_inv(0, 2) * fz);
                        _opti.subject_to(_a(4, i) == _I_inv(1, 0) * fx + _I_inv(1, 1) * fy + _I_inv(1, 2) * fz);
                        _opti.subject_to(_a(5, i) == _I_inv(2, 0) * fx + _I_inv(2, 1) * fy + _I_inv(2, 2) * fz);
                    }
                }

                for (int i = 1; i < _H + 1; ++i)
                {
                    // VELOCITY UPDATE
                    _opti.subject_to(_u(Slice(), i) == _u(Slice(), i - 1) + _a(Slice(), i - 1) * _dt);
                    // POSITION UPDATE
                    _opti.subject_to(_x(0, i) == _x(0, i - 1) + ((2 * (_x(3, i - 1) * _x(3, i - 1) + _x(4, i - 1) * _x(4, i - 1)) - 1) * _u(0, i) +
                                                                 (2 * (_x(4, i - 1) * _x(5, i - 1) - _x(3, i - 1) * _x(6, i - 1))) * _u(1, i) +
                                                                 (2 * (_x(4, i - 1) * _x(6, i - 1) + _x(3, i - 1) * _x(5, i - 1))) * _u(2, i)) *
                                                                    _dt);
                    _opti.subject_to(_x(1, i) == _x(1, i - 1) + ((2 * (_x(4, i - 1) * _x(5, i - 1) + _x(3, i - 1) * _x(6, i - 1))) * _u(0, i) +
                                                                 (2 * (_x(3, i - 1) * _x(3, i - 1) + _x(5, i - 1) * _x(5, i - 1)) - 1) * _u(1, i) +
                                                                 (2 * (_x(5, i - 1) * _x(6, i - 1) - _x(3, i - 1) * _x(4, i - 1))) * _u(2, i)) *
                                                                    _dt);
                    _opti.subject_to(_x(2, i) == _x(2, i - 1) + ((2 * (_x(4, i - 1) * _x(6, i - 1) - _x(3, i - 1) * _x(5, i - 1))) * _u(0, i) +
                                                                 (2 * (_x(5, i - 1) * _x(6, i - 1) + _x(3, i - 1) * _x(4, i - 1))) * _u(1, i) +
                                                                 (2 * (_x(3, i - 1) * _x(3, i - 1) + _x(6, i - 1) * _x(6, i - 1)) - 1) * _u(2, i)) *
                                                                    _dt);
                    // QUATERNION UPDATE
                    MX q = MX::zeros(4, 1);
                    q(0) = _x(3, i - 1) + 0.5 * (-_x(4, i - 1) * _u(3, i) - _x(5, i - 1) * _u(4, i) - _x(6, i - 1) * _u(5, i)) * _dt;
                    q(1) = _x(4, i - 1) + 0.5 * (_x(3, i - 1) * _u(3, i) - _x(6, i - 1) * _u(4, i) + _x(5, i - 1) * _u(5, i)) * _dt;
                    q(2) = _x(5, i - 1) + 0.5 * (_x(6, i - 1) * _u(3, i) + _x(3, i - 1) * _u(4, i) - _x(4, i - 1) * _u(5, i)) * _dt;
                    q(3) = _x(6, i - 1) + 0.5 * (-_x(5, i - 1) * _u(3, i) + _x(4, i - 1) * _u(4, i) + _x(3, i - 1) * _u(5, i)) * _dt;

                    _opti.subject_to(_x(Slice(3, 7), i) == q / if_else(norm_2(q) > 1e-6, norm_2(q), 1.0, true));
                }

                if (_prev)
                {
                    for (int i = 0; i < pq::Value::Param::NumOpt::prev_steps_init; ++i)
                    {
                        _opti.set_initial(_c(Slice(), i), _prev_result(Slice(), i + 1));
                    }
                }

                Dict opts;
                opts["ipopt.print_level"] = 0;
                opts["print_time"] = false;
                opts["ipopt.tol"] = 1e-3;
                opts["ipopt.hessian_approximation"] = "limited-memory";

                _opti.solver("ipopt", opts);
            }
        };
    }
}

#endif