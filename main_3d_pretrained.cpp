#include <Eigen/Core>
#include <iostream>

#define NUM_OPT 1

#include "src/opt/num-opt/NumOptimizer.hpp"
#include "src/opt/num-opt/SymNN.hpp"
#include "src/params.hpp"
#include "src/sim/Quadrotor.hpp"
#include "src/train/Episode.hpp"

int main()
{
    quadrotor::Value::target << 0, 0, 3, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0;

    double mass = 4;

    quadrotor::num_opt::Params params;
    params.m = mass;
    params.I =
      mass / quadrotor::Value::Constant::mass * quadrotor::Value::Constant::I;

    symnn::Params nn_params;

    quadrotor::num_opt::NumOptimizer optimizer(
      params,
      nn_params,
      "src/train/models/D_quad_model_4_0");

    quadrotor::Value::Param::Train::bad_episode_stop = false;

    quadrotor::train::Episode episode("src/train/data/quad.txt");
    episode.run(optimizer);

    return 0;
}