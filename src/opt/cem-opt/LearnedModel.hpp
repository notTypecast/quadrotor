#ifndef PQ_LEARNED_MODEL_HPP
#define PQ_LEARNED_MODEL_HPP
#include <Eigen/Core>
#include <vector>

#include "simple_nn/src/simple_nn/neural_net.hpp"
#include "simple_nn/src/simple_nn/layer.hpp"
#include "simple_nn/src/simple_nn/activation.hpp"
#include "simple_nn/src/simple_nn/loss.hpp"
#include "simple_nn/src/simple_nn/opt.hpp"

#include "src/params.hpp"

namespace pq
{
    namespace cem_opt
    {
        class NNModel
        {
        public:
            NNModel(const std::vector<int> &hidden_layers)
            {
                assert(hidden_layers.size() > 0 && "expected at least one hidden layer");
                _network.add_layer<simple_nn::FullyConnectedLayer<simple_nn::Sigmoid>>(8, hidden_layers[0]);
                for (int i = 1; i < hidden_layers.size(); ++i)
                {
                    _network.add_layer<simple_nn::FullyConnectedLayer<simple_nn::Sigmoid>>(hidden_layers[i - 1], hidden_layers[i]);
                }
                _network.add_layer<simple_nn::FullyConnectedLayer<simple_nn::Linear>>(hidden_layers.back(), 3);

                _network.set_weights(Eigen::VectorXd::Zero(_network.num_weights()));
                _optimizer.reset(Eigen::VectorXd::Zero(_network.num_weights()));
            }

            void train(const Eigen::Matrix<double, 8, -1> &input, const Eigen::Matrix<double, 3, -1> &target)
            {
                auto eval = [&](const Eigen::VectorXd &params)
                {
                    _network.set_weights(params);
                    Eigen::VectorXd dtheta = _network.backward<simple_nn::MeanSquaredError>(input, target);
                    return std::make_pair(0.0, dtheta);
                };

                Eigen::VectorXd theta;

                for (int i = 0; i < pq::Value::Param::SimpleNN::epochs; ++i)
                {
                    bool stop;
                    std::tie(stop, std::ignore, theta) = _optimizer.optimize_once(eval);
                    _network.set_weights(theta);
                    if (i < 10 || i % 100 == 0)
                    {
                        std::cout << "Weights: " << theta.transpose() << std::endl;
                    }

                    if (stop)
                    {
                        break;
                    }
                }
                _trained = true;
            }

            Eigen::VectorXd predict(const Eigen::Vector<double, 8> &input)
            {
                return _network.forward(input);
            }

            void reset()
            {
                _trained = false;
                _network.set_weights(Eigen::VectorXd::Zero(_network.num_weights()));
                _optimizer.reset(Eigen::VectorXd::Zero(_network.num_weights()));
            }

            bool trained()
            {
                return _trained;
            }

        private:
            simple_nn::NeuralNet _network;
            simple_nn::Adam _optimizer;
            bool _trained = false;
        };
    }
}

#endif
