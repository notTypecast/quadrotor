#ifndef PQ_EPISODE_HPP
#define PQ_EPISODE_HPP
#include <Eigen/Core>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <thread>

#include "src/opt/DynamicModel.hpp"
#include "src/opt/Optimizer.hpp"
#include "src/params.hpp"
#include "src/sim/PlanarQuadrotor.hpp"
#include "src/sim/Quadrotor.hpp"
#include "src/sim/Visualizer.hpp"

namespace pq
{
namespace train
{
struct Errors
{
    bool   failed           = false;
    int    steps            = 0;
    double position         = 0.0;
    double orientation      = 0.0;
    double velocity         = 0.0;
    double angular_velocity = 0.0;

    friend std::ostream &operator<<(std::ostream &os, const Errors &err);
};

std::ostream &operator<<(std::ostream &os, const Errors &err)
{
    os << err.failed << "," << err.steps << "," << err.position << ","
       << err.orientation << "," << err.velocity << "," << err.angular_velocity;
    return os;
}

class Episode
{
  public:
    Episode()
    {
        _train_input =
          Eigen::MatrixXd(8, pq::Value::Param::Train::collection_steps);
        _train_target =
          Eigen::MatrixXd(3, pq::Value::Param::Train::collection_steps);
    }

    Errors run(Optimizer &optimizer, pq::sim::Visualizer &v)
    {
        _visualize = true;
        _v         = v;
        return _run(optimizer);
    }

    Errors run(Optimizer &optimizer)
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
    Eigen::MatrixXd     _train_input;
    Eigen::MatrixXd     _train_target;
    int                 _stop_step = -1;
    int                 _episode   = 1;
    int                 _run_iter  = -1;
    bool                _visualize = false;
    pq::sim::Visualizer _v;

    Errors _run(Optimizer &optimizer)
    {
        _stop_step = -1;

        optimizer.reinit();

        pq::sim::PlanarQuadrotor p(pq::Value::Constant::mass,
                                   pq::Value::Constant::inertia,
                                   pq::Value::Constant::length);

        double                        control_freq = 0;
        int                           count        = 0;
        std::chrono::duration<double> elapsed =
          std::chrono::duration<double>::zero();
        std::chrono::duration<double> total_time =
          std::chrono::duration<double>::zero();
        auto real_start = std::chrono::high_resolution_clock::now();

        Errors errors;

        for (int i = 0; i < pq::Value::Param::Train::collection_steps; ++i)
        {
            if (pq::Value::Param::Train::big_angle_stop &&
                std::abs(p.get_state()[2]) >
                  pq::Value::Param::Train::big_angle_threshold)
            {
                _stop_step = i;
                break;
            }

            total_time +=
              std::chrono::high_resolution_clock::now() - real_start;
            real_start = std::chrono::high_resolution_clock::now();
            if (pq::Value::Param::Sim::sync_with_real_time)
            {
                if (p.get_sim_time() > total_time.count())
                {
                    std::this_thread::sleep_for(std::chrono::milliseconds(
                      (int)(1000 * (p.get_sim_time() - total_time.count()))));
                }
            }

            pq::Value::init_state = p.get_state();

            Eigen::Vector2d controls;

            try
            {
                auto start = std::chrono::high_resolution_clock::now();

                controls = optimizer.next(p.get_state(), pq::Value::target);

                elapsed += std::chrono::high_resolution_clock::now() - start;
            }
            catch (std::exception &e)
            {
                _stop_step    = std::max(i - 2, 0);
                errors.failed = true;
                break;
            }

            p.update(controls, pq::Value::Param::Sim::dt);

            _train_input.col(i) =
              (Eigen::Vector<double, 8>() << pq::Value::init_state, controls)
                .finished();
            _train_target.col(i) =
              p.get_last_ddq() -
              pq::dynamic_model_predict(pq::Value::init_state,
                                        controls,
                                        optimizer.model_params());

            if (_visualize)
            {
                std::stringstream ss;
                ss << std::fixed << std::setprecision(2)
                   << "Control frequency: " << control_freq
                   << " Hz, angle: " << p.get_state()[2] * 360 / M_PI
                   << " deg, time: " << p.get_sim_time() << " sec (ratio "
                   << pq::Value::Param::Sim::dt /
                        std::chrono::duration_cast<
                          std::chrono::duration<double>>(
                          std::chrono::high_resolution_clock::now() -
                          real_start)
                          .count()
                   << "), MPC mass: " << optimizer.model_params()[0]
                   << " kg, episode: " << _episode;
                if (_run_iter != -1)
                {
                    ss << ", run: " << _run_iter;
                }

                _v.set_message(ss.str());
                _v.show(p, { pq::Value::target[0], pq::Value::target[1] });
            }

            ++count;
            if (elapsed.count() > 1)
            {
                control_freq = count / elapsed.count();
                count        = 0;
                elapsed      = std::chrono::duration<double>::zero();
            }

            Eigen::VectorXd state = p.get_state();
            errors.position +=
              (pq::Value::target.segment(0, 2) - state.segment(0, 2))
                .squaredNorm();
            errors.orientation += std::abs(pq::Value::target(3) - state(3));
            errors.velocity    += state.segment(3, 2).squaredNorm();
            errors.angular_velocity += std::abs(state(5));

            ++errors.steps;
        }

        ++_episode;

        if (_stop_step == -1)
        {
            _stop_step = pq::Value::Param::Train::collection_steps - 1;
        }

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
        _train_input =
          Eigen::MatrixXd(17, quadrotor::Value::Param::Train::collection_steps);
        _train_target =
          Eigen::MatrixXd(6, quadrotor::Value::Param::Train::collection_steps);

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

    pq::train::Errors run(pq::Optimizer &optimizer, bool full_run = false,
                          bool print = true)
    {
        _stop_step = -1;
        optimizer.reinit();

        quadrotor::sim::Quadrotor q(quadrotor::Value::Constant::mass,
                                    quadrotor::Value::Constant::I,
                                    quadrotor::Value::Constant::length);

        if (print && _filestream.is_open())
        {
            _filestream << "TARGET "
                        << quadrotor::Value::target.segment(0, 3).transpose()
                        << std::endl;
        }

        Eigen::RowVector4d target_q_T =
          quadrotor::Value::target.segment(3, 4).transpose();
        pq::train::Errors errors;

        for (int i = 0; i < quadrotor::Value::Param::Train::collection_steps;
             ++i)
        {
            Eigen::VectorXd init_state = q.get_state();

            if (print)
            {
                if (_filestream.is_open())
                {
                    _filestream << init_state.transpose() << std::endl;
                }
                else
                {
                    std::cout << "Current state: " << init_state.transpose()
                              << std::endl;
                }
            }

            if (!full_run && quadrotor::Value::Param::Train::bad_episode_stop &&
                _stop_step == -1)
            {
                Eigen::Vector3d normal = q.get_normal();
                double          angle  = acos(normal[2]);

                if (angle > quadrotor::Value::Param::Train::
                              bad_episode_angle_threshold ||
                    q.get_state().segment(7, 6).cwiseAbs().maxCoeff() >
                      quadrotor::Value::Param::Train::
                        bad_episode_speed_threshold)
                {
                    _stop_step = std::max(i - 2, 0);
                    break;
                }
            }

            Eigen::Vector4d controls;

            try
            {
                auto start = std::chrono::high_resolution_clock::now();
                controls = optimizer.next(init_state, quadrotor::Value::target);
                auto elapsed =
                  std::chrono::high_resolution_clock::now() - start;
                if (print)
                {
                    std::cout
                      << "Control frequency: "
                      << 1.0 / std::chrono::duration<double>(elapsed).count()
                      << " Hz" << std::endl;
                }
            }
            catch (std::exception &e)
            {
                if (print)
                {
                    std::cout << e.what() << std::endl;
                    std::cout << "Optimization failed, stopping episode"
                              << std::endl;
                }
                _stop_step    = std::max(i - 2, 0);
                errors.failed = true;
                break;
            }

            q.update(controls, quadrotor::Value::Param::Sim::dt);

            _train_input.col(i) =
              (Eigen::Vector<double, 17>() << init_state, controls).finished();
            _train_target.col(i) =
              q.get_last_ddq() -
              quadrotor::dynamic_model_predict(init_state,
                                               controls,
                                               optimizer.model_params());

            errors.position +=
              (quadrotor::Value::target.segment(0, 3) - q.get_position())
                .squaredNorm();
            errors.orientation += std::pow(
              1 -
                std::pow((target_q_T * q.get_orientation_vector()).value(), 2),
              2);
            errors.velocity +=
              (quadrotor::Value::target.segment(7, 3) - q.get_velocity())
                .squaredNorm();
            errors.angular_velocity +=
              (quadrotor::Value::target.segment(10, 3) -
               q.get_angular_velocity())
                .squaredNorm();

            ++errors.steps;
        }

        if (_stop_step == -1)
        {
            _stop_step = quadrotor::Value::Param::Train::collection_steps - 1;
        }

        if (print) std::cout << "Final errors: " << errors << std::endl;

        return errors;
    }

    Eigen::MatrixXd get_train_input()
    {
        return _stop_step == -1 ? _train_input
                                : _train_input.block(0, 0, 17, _stop_step);
    }

    Eigen::MatrixXd get_train_target()
    {
        return _stop_step == -1 ? _train_target
                                : _train_target.block(0, 0, 6, _stop_step);
    }

    int get_stop_step()
    {
        return _stop_step;
    }

  protected:
    std::ofstream   _filestream;
    Eigen::MatrixXd _train_input;
    Eigen::MatrixXd _train_target;
    int             _stop_step = -1;
};
}
}

#endif
