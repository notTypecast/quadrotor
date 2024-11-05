#include <Eigen/Core>
#include <iostream>
#include <chrono>
#include <thread>

#define NUM_OPT 1

#include "src/params.hpp"
#include "src/sim/Quadrotor.hpp"
#include "src/opt/num-opt/SymNN.hpp"
#include "src/opt/num-opt/NumOptimizer.hpp"
#include "src/train/Episode.hpp"

int main()
{  
    quadrotor::Value::target << quadrotor::Value::Param::NumOpt::target_x, quadrotor::Value::Param::NumOpt::target_y, quadrotor::Value::Param::NumOpt::target_z, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0;

    symnn::Params params;
    params.input_size = 17;
    params.output_size = 6;
    params.hidden_layers = std::vector<int>{4};
    params.activation = symnn::activation::Sigmoid;
    params.initializer = symnn::initializers::NXavier;
    params.gradient_based = quadrotor::Value::Param::SymNN::gradient_based;
    params.epochs = quadrotor::Value::Param::SymNN::epochs;
    params.learning_rate = quadrotor::Value::Param::SymNN::learning_rate;
    params.momentum = quadrotor::Value::Param::SymNN::momentum;
    params.max_grad = quadrotor::Value::Param::SymNN::max_grad;
    quadrotor::Value::Param::SymNN::learned_model = std::make_unique<symnn::SymNN>(params);

    double masses[] = {1, 2, 4};
    std::vector<std::vector<double>> errors_per_episode(quadrotor::Value::Param::Train::runs * quadrotor::Value::Param::Train::episodes);

    Eigen::MatrixXd train_input, train_target;

    for (int i = 0; i < 3; ++i)
    {
        quadrotor::num_opt::Params params;
        params.m = masses[i];
        params.I = masses[i] / quadrotor::Value::Constant::mass * quadrotor::Value::Constant::I;

        quadrotor::num_opt::NumOptimizer optimizer(params);

        std::cout << "Running with mass = " << masses[i] << std::endl;

        for (int j = 0; j < quadrotor::Value::Param::Train::runs; ++j)
        {
            std::cout << "Run " << j << std::endl;
            quadrotor::Value::Param::SymNN::learned_model->reset();
            quadrotor::Value::Param::NumOpt::use_learned = false;
            quadrotor::train::Episode episode;

            int run_idx = j * quadrotor::Value::Param::Train::episodes;
            for (int k = 0; k < quadrotor::Value::Param::Train::episodes; ++k)
            {
                std::cout << "Episode " << k << std::endl;
                episode.run(optimizer);

                if (quadrotor::Value::Param::SymNN::use_all_data)
                {
                    Eigen::MatrixXd new_input = episode.get_train_input();
                    Eigen::MatrixXd new_target = episode.get_train_target();
                    if (train_input.size() == 0)
                    {
                        train_input = new_input;
                        train_target = new_target;
                    }
                    else
                    {
                        train_input.conservativeResize(train_input.rows(), train_input.cols() + new_input.cols());
                        train_target.conservativeResize(train_target.rows(), train_target.cols() + new_target.cols());
                        train_input.block(0, train_input.cols() - new_input.cols(), train_input.rows(), new_input.cols()) = new_input;
                        train_target.block(0, train_target.cols() - new_target.cols(), train_target.rows(), new_target.cols()) = new_target;
                    }
                }
                else
                {
                    train_input = episode.get_train_input();
                    train_target = episode.get_train_target();
                }

                quadrotor::Value::Param::SymNN::learned_model->train(train_input, train_target);
                
                /*
                std::cout << "NN expected:" << std::endl;
                std::cout << episode.get_train_target().block(0, 0, 6, episode.get_stop_step()).transpose() << std::endl;
                std::cout << "NN actual:" << std::endl;
                for (int l = 0; l < episode.get_stop_step(); ++l)
                {
                    std::cout << quadrotor::Value::Param::SymNN::learned_model->forward(episode.get_train_input().col(l)).transpose() << std::endl;
                }*/
                
                quadrotor::Value::Param::NumOpt::use_learned = true;
            }

            std::cout << "Episode with completed training" << std::endl;
            episode.run(optimizer);
        }
    }

    return 0;
}