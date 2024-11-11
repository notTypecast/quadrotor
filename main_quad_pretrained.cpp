#include <Eigen/Core>
#include <iostream>

#define NUM_OPT 1

#include "src/params.hpp"
#include "src/sim/Quadrotor.hpp"
#include "src/opt/num-opt/SymNN.hpp"
#include "src/opt/num-opt/NumOptimizer.hpp"
#include "src/train/Episode.hpp"

int main()
{
    quadrotor::Value::target << 5, -5, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0;

    symnn::Params nn_params;
    quadrotor::Value::Param::SymNN::learned_model = std::make_unique<symnn::SymNN>("quad_model_1_0", nn_params);

    double mass = 1;

    quadrotor::num_opt::Params params;
    params.m = mass;
    params.I = mass / quadrotor::Value::Constant::mass * quadrotor::Value::Constant::I;

    quadrotor::num_opt::NumOptimizer optimizer(params);

    quadrotor::Value::Param::NumOpt::use_learned = true;

    quadrotor::train::Episode episode;
    episode.run(optimizer);

    return 0;
}