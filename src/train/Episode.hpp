#ifndef PQ_EPISODE_HPP
#define PQ_EPISODE_HPP
#include <Eigen/Core>
#include <iostream>
#include <chrono>
#include <sstream>
#include <iomanip>
#include <memory>
#include <thread>

#include "src/params.hpp"
#include "src/opt/Optimizer.hpp"
#include "src/opt/DynamicModel.hpp"
#include "src/sim/PlanarQuadrotor.hpp"
#include "src/sim/Visualizer.hpp"
#include "src/sim/Quadrotor.hpp"

namespace pq
{
    namespace train
    {
        class Episode
        {
        public:
            Episode()
            {
                _train_input = Eigen::MatrixXd(8, pq::Value::Param::Train::collection_steps);
                _train_target = Eigen::MatrixXd(3, pq::Value::Param::Train::collection_steps);
            }

            std::vector<double> run(Optimizer &optimizer, pq::sim::Visualizer &v)
            {
                _visualize = true;
                _v = v;
                return _run(optimizer);
            }

            std::vector<double> run(Optimizer &optimizer)
            {
                _visualize = false;
                return _run(optimizer);
            }

            Eigen::MatrixXd get_train_input()
            {
                return _train_input;
            }

            Eigen::MatrixXd get_train_target()
            {
                return _train_target;
            }

            int get_stop_step()
            {
                return _stop_step;
            }

            void set_run(int run)
            {
                _run_iter = run;
            }

        protected:
            Eigen::MatrixXd _train_input;
            Eigen::MatrixXd _train_target;
            int _stop_step = -1;
            int _episode = 1;
            int _run_iter = -1;
            bool _visualize = false;
            pq::sim::Visualizer _v;

            std::vector<double> _run(Optimizer &optimizer)
            {
                _stop_step = -1;

                optimizer.reinit();

                pq::sim::PlanarQuadrotor p(pq::Value::Constant::mass, pq::Value::Constant::inertia, pq::Value::Constant::length);

                double control_freq = 0;
                int count = 0;
                std::chrono::duration<double> elapsed = std::chrono::duration<double>::zero();
                std::chrono::duration<double> total_time = std::chrono::duration<double>::zero();
                auto real_start = std::chrono::high_resolution_clock::now();

                // int episode_idx = (_run_iter - 1) * pq::Value::Param::Train::episodes * pq::Value::Param::Train::collection_steps + (_episode - 1) * pq::Value::Param::Train::collection_steps;

                std::vector<double> errors(pq::Value::Param::Train::collection_steps, 0);

                for (int i = 0; i < pq::Value::Param::Train::collection_steps; ++i)
                {
                    total_time += std::chrono::high_resolution_clock::now() - real_start;
                    real_start = std::chrono::high_resolution_clock::now();
                    if (pq::Value::Param::Sim::sync_with_real_time)
                    {
                        if (p.get_sim_time() > total_time.count())
                        {
                            std::this_thread::sleep_for(std::chrono::milliseconds((int)(1000 * (p.get_sim_time() - total_time.count()))));
                        }
                    }

                    pq::Value::init_state = p.get_state();

                    auto start = std::chrono::high_resolution_clock::now();

                    Eigen::VectorXd controls = optimizer.next(p.get_state(), pq::Value::target);

                    elapsed += std::chrono::high_resolution_clock::now() - start;

                    p.update(controls, pq::Value::Param::Sim::dt);
                    errors[i] = (pq::Value::target.segment(0, 6) - p.get_state().segment(0, 6)).squaredNorm();

                    _train_input.col(i) = (Eigen::Vector<double, 8>() << pq::Value::init_state, controls).finished();
                    _train_target.col(i) = p.get_last_ddq() - pq::dynamic_model_predict(pq::Value::init_state, controls, optimizer.model_params());

                    std::stringstream ss;
                    ss << std::fixed << std::setprecision(2) << "Control frequency: " << control_freq
                       << " Hz, angle: " << p.get_state()[2] * 360 / M_PI
                       << " deg, time: " << p.get_sim_time()
                       << " sec (ratio " << pq::Value::Param::Sim::dt / std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::high_resolution_clock::now() - real_start).count()
                       << "), MPC mass: " << optimizer.model_params()[0]
                       << " kg, episode: " << _episode;
                    if (_run_iter != -1)
                    {
                        ss << ", run: " << _run_iter;
                    }
                    if (_visualize)
                    {
                        _v.set_message(ss.str());
                        _v.show(p, {pq::Value::target[0], pq::Value::target[1]});
                    }

                    ++count;
                    if (elapsed.count() > 1)
                    {
                        control_freq = count / elapsed.count();
                        count = 0;
                        elapsed = std::chrono::duration<double>::zero();
                    }

                    if (pq::Value::Param::Train::big_angle_stop && std::abs(p.get_state()[2]) > pq::Value::Param::Train::big_angle_threshold && _stop_step == -1)
                    {
                        _stop_step = i;
                        if (!pq::Value::Param::Train::big_angle_view)
                        {
                            break;
                        }
                    }
                }

                ++_episode;

                return errors;
            }
        };
    }
}

namespace quadrotor
{
    namespace train
    {
        class Episode
        {
        public:
            Episode(const std::string &filename = "")
            {
                _train_input = Eigen::MatrixXd(17, quadrotor::Value::Param::Train::collection_steps);
                _train_target = Eigen::MatrixXd(6, quadrotor::Value::Param::Train::collection_steps);

                if (!filename.empty())
                {
                    _filestream = std::ofstream(filename, std::ios_base::app);
                }
            }

            ~Episode()
            {
                if (_filestream.is_open())
                {
                    _filestream.close();
                }
            }

            std::vector<double> run(pq::Optimizer &optimizer, bool full_run = false)
            {
                _stop_step = -1;
                optimizer.reinit();

                quadrotor::sim::Quadrotor q(quadrotor::Value::Constant::mass, quadrotor::Value::Constant::I, quadrotor::Value::Constant::length);

                if (_filestream.is_open())
                {
                    _filestream << "TARGET " << quadrotor::Value::target.segment(0, 3).transpose() << std::endl;
                }

                std::vector<double> errors(quadrotor::Value::Param::Train::collection_steps, 0);

                for (int i = 0; i < quadrotor::Value::Param::Train::collection_steps; ++i)
                {
                    Eigen::VectorXd init_state = q.get_state();

                    if (_filestream.is_open())
                    {
                        _filestream << init_state.transpose() << std::endl;
                    }
                    else
                    {
                        std::cout << "Current state: " << init_state.transpose() << std::endl;
                    }

                    if (!full_run && quadrotor::Value::Param::Train::bad_episode_stop)
                    {
                        Eigen::Vector3d normal = q.get_normal();
                        double angle = acos(normal[2]);

                        if (angle > quadrotor::Value::Param::Train::bad_episode_angle_threshold ||
                            q.get_state().segment(7, 6).cwiseAbs().maxCoeff() > quadrotor::Value::Param::Train::bad_episode_speed_threshold)
                        {
                            _stop_step = i - 2 < 0 ? 0 : i - 2;
                            break;
                        }
                    }

                    Eigen::Vector4d controls;

                    try
                    {
                        auto start = std::chrono::high_resolution_clock::now();
                        controls = optimizer.next(init_state, quadrotor::Value::target);
                        auto elapsed = std::chrono::high_resolution_clock::now() - start;
                        std::cout << "Control frequency: " << 1.0 / std::chrono::duration<double>(elapsed).count() << " Hz" << std::endl;
                    }
                    catch (std::exception &e)
                    {
                        std::cout << e.what() << std::endl;
                        std::cout << "Optimization failed, stopping episode" << std::endl;
                        _stop_step = i - 2 < 0 ? 0 : i - 2;
                        break;
                    }

                    q.update(controls, quadrotor::Value::Param::Sim::dt);

                    errors[i] = (quadrotor::Value::target.segment(0, 3) - q.get_state().segment(0, 3)).squaredNorm();

                    _train_input.col(i) = (Eigen::Vector<double, 17>() << init_state, controls).finished();
                    _train_target.col(i) = q.get_last_ddq() - quadrotor::dynamic_model_predict(init_state, controls, optimizer.model_params());
                }

                if (_stop_step == -1)
                {
                    _stop_step = quadrotor::Value::Param::Train::collection_steps - 1;
                }

                std::cout << "Final error: " << errors[_stop_step] << std::endl;

                return errors;
            }

            Eigen::MatrixXd get_train_input()
            {
                return _stop_step == -1 ? _train_input : _train_input.block(0, 0, 17, _stop_step);
            }

            Eigen::MatrixXd get_train_target()
            {
                return _stop_step == -1 ? _train_target : _train_target.block(0, 0, 6, _stop_step);
            }

            int get_stop_step()
            {
                return _stop_step;
            }

        protected:
            std::ofstream _filestream;
            Eigen::MatrixXd _train_input;
            Eigen::MatrixXd _train_target;
            int _stop_step = -1;
        };
    }
}

#endif
