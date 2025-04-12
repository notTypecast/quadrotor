#ifndef PQ_OPTIMIZER_HPP
#define PQ_OPTIMIZER_HPP

#include <Eigen/Core>

namespace pq
{
class Optimizer
{
  public:
    virtual void            reinit()                            = 0;
    virtual Eigen::VectorXd next(const Eigen::VectorXd &init,
                                 const Eigen::VectorXd &target) = 0;
    virtual Eigen::VectorXd model_params()                      = 0;
};
}

#endif