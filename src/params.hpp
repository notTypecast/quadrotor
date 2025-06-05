#ifndef PQ_PARAMS_HPP
#define PQ_PARAMS_HPP
#include <Eigen/Core>
#include <memory>

namespace symnn
{
class SymNN;
}

namespace pq
{
namespace cem_opt
{
class NNModel;
}
namespace Value
{
// Constants / predefined values
namespace Constant
{
constexpr double g      = 9.81; // earth gravity
constexpr double gm     = 1.61; // moon gravity
constexpr double mass   = 1.0;  // default quadrotor mass
constexpr double length = 0.3;  // default total quadrotor length
constexpr double inertia =
  0.2 * mass * length * length; // default quadrotor inertia
}

// Parameters for simulation and optimization
namespace Param
{
namespace Sim
{
constexpr double dt = 0.05; // simulation time step
constexpr bool   sync_with_real_time =
  true; // whether to sync simulation with real time (ratio <= 1)
}
namespace CEMOpt
{
constexpr int    target_x = 10;             // target x position
constexpr int    target_y = 10;             // target y position
constexpr double g        = Constant::g;    // MPC gravity
double           mass;                      // MPC mass
constexpr double length = Constant::length; // MPC length
double           inertia;                   // MPC inertia
constexpr int    steps = 80;                // MPC steps
constexpr int    horizon =
  20; // MPC horizon (number of control inputs per individual)
constexpr int    pop_size   = 200; // population size
constexpr int    num_elites = 32;  // number of elites
constexpr double max_value =
  Constant::mass * Constant::g;   // maximum control input force
constexpr double min_value = 0.0; // minimum control input force
constexpr double init_mu =
  0.5 * Constant::mass * Constant::g; // initial mean for CEM
constexpr double init_std = 0.3;      // initial standard deviation for CEM
Eigen::Vector4d  model_params;
}
namespace SimpleNN
{
#ifdef CEM_OPT
std::unique_ptr<pq::cem_opt::NNModel> learned_model;
#endif
constexpr int  epochs       = 4000; // number of epochs for training
constexpr bool use_all_data = true;
}
namespace NumOpt
{
constexpr double target_x = 4.5;     // target x position
constexpr double target_y = 4;       // target y position
constexpr int    horizon  = 20;      // Horizon
constexpr double dt       = Sim::dt; // optimization time step
constexpr double F_max    = Constant::mass * Constant::g; // maximum force
constexpr int    prev_steps_init =
  horizon - 1;        // number of previous steps to use for warm start
bool baseline = true; // whether to calculate baseline error with correct model
                      // before starting runs
}
namespace SymNN
{
// Gradient-based only parameters
constexpr int    epochs           = 2000;
constexpr double learning_rate    = 0.01;
constexpr double momentum         = 0;
constexpr double max_grad         = 1.0;
double           dropout_rate     = 0.3;
int              inference_passes = 10;
}
namespace Train
{
constexpr bool big_angle_stop =
  true; // whether to stop training after angle values get too big
constexpr double big_angle_threshold =
  3 * M_PI / 8; // threshold for big angle values
constexpr int collection_steps =
  200; // number of steps to collect data for training (per episode)
constexpr int episodes = 10; // number of episodes to train
constexpr int runs     = 32; // number of runs to train (for averaging)
}
}

Eigen::Vector<double, 6> init_state;
Eigen::Vector<double, 6> target;
}
}

namespace quadrotor
{
namespace Value
{
namespace Constant
{
constexpr double            g = 9.81;
Eigen::Matrix<double, 4, 3> H_mat =
  (Eigen::Matrix<double, 4, 3>() << 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1)
    .finished();
constexpr double mass   = 0.5;
constexpr double length = 0.175;
Eigen::Matrix3d  I =
  (Eigen::Matrix3d() << 0.0023, 0, 0, 0, 0.0023, 0, 0, 0, 0.004).finished();
constexpr double Kf = 1.0;
constexpr double Kt = 0.0245;

}

namespace Param
{
namespace Sim
{
constexpr double dt = 0.05;
}
namespace NumOpt
{
constexpr double target_x           = 4;
constexpr double target_y           = 4;
constexpr double target_z           = 2;
constexpr int    horizon            = 8;
constexpr double dt                 = Sim::dt;
constexpr double control_max        = 7 * Constant::mass * Constant::g / 24;
constexpr int    prev_steps_init    = horizon - 1;
double           nn_variance_weight = 0.1;
bool             baseline           = true;
}
namespace SymNN
{
constexpr bool use_all_data     = true;
double         dropout_rate     = 0;
int            inference_passes = 1;
}
namespace Train
{
bool             bad_episode_stop = true; // whether to stop bad episodes
constexpr double bad_episode_angle_threshold =
  1.0; // threshold for angle between World Z and Quadrotor Z axis
constexpr double bad_episode_speed_threshold = 8.0; // threshold for speed
constexpr int    collection_steps =
  200; // number of steps to collect data for training (per episode)
constexpr int episodes = 8; // number of episodes to train
constexpr int runs     = 1; // number of runs to train (for averaging)
}
}

Eigen::Vector<double, 13> init_state;
Eigen::Vector<double, 13> target;
}
}

#endif