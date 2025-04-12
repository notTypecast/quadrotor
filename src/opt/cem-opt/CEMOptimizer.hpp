#ifndef PQ_CEM_OPTIMIZER_HPP
#define PQ_CEM_OPTIMIZER_HPP

#include "src/opt/Optimizer.hpp"
#include "src/opt/cem-opt/Individual.hpp"
#include "src/params.hpp"

#include "algevo/src/algevo/algo/cem.hpp"

using Algo = algevo::algo::CrossEntropyMethod<pq::cem_opt::ControlIndividual>;

namespace pq
{
namespace cem_opt
{
struct Params
{
    int             dim        = pq::cem_opt::ControlIndividual::dim;
    int             pop_size   = pq::Value::Param::CEMOpt::pop_size;
    int             num_elites = pq::Value::Param::CEMOpt::num_elites;
    Eigen::VectorXd max_value  = Algo::x_t::Constant(
      pq::cem_opt::ControlIndividual::dim, pq::Value::Param::CEMOpt::max_value);
    Eigen::VectorXd min_value = Algo::x_t::Constant(
      pq::cem_opt::ControlIndividual::dim, pq::Value::Param::CEMOpt::min_value);
    Eigen::VectorXd init_mu = Algo::x_t::Constant(
      pq::cem_opt::ControlIndividual::dim, pq::Value::Param::CEMOpt::init_mu);
    Eigen::VectorXd init_std = Algo::x_t::Constant(
      pq::cem_opt::ControlIndividual::dim, pq::Value::Param::CEMOpt::init_std);
    double m = pq::Value::Constant::mass;
    double I = pq::Value::Constant::inertia;
    double l = pq::Value::Constant::length;
    double g = pq::Value::Constant::g;
};

class CEMOptimizer : public Optimizer
{
  public:
    CEMOptimizer(Params params)
    {
        _params.dim        = params.dim;
        _params.pop_size   = params.pop_size;
        _params.num_elites = params.num_elites;
        _params.max_value  = params.max_value;
        _params.min_value  = params.min_value;
        _params.init_mu    = params.init_mu;
        _params.init_std   = params.init_std;

        _model_params = Eigen::VectorXd::Zero(4);
        _model_params << params.m, params.I, params.l, params.g;
        pq::Value::Param::CEMOpt::model_params = _model_params;
    }

    virtual void reinit()
    {
        _params.init_mu =
          Algo::x_t::Constant(_params.dim, pq::Value::Param::CEMOpt::init_mu);
    }

    virtual Eigen::VectorXd next(const Eigen::VectorXd &init,
                                 const Eigen::VectorXd &target)
    {
        pq::Value::init_state = init;
        pq::Value::target     = target;

        Algo cem(_params);

        for (int j = 0; j < pq::Value::Param::CEMOpt::steps; ++j)
        {
            cem.step();
        }

        _params.init_mu          = cem.best();
        Eigen::Vector2d controls = cem.best().segment(0, 2);

        return controls;
    }

    virtual Eigen::VectorXd model_params()
    {
        return _model_params;
    }

  protected:
    Algo::Params    _params;
    Eigen::VectorXd _model_params;
};
}
}

#endif