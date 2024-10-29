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
    /*
    quadrotor::sim::Quadrotor q(quadrotor::Value::Constant::mass, quadrotor::Value::Constant::I, quadrotor::Value::Constant::length);

    quadrotor::num_opt::Params params;
    params.init = q.get_state();
    std::cout << "Initial state: " << params.init.transpose() << std::endl;
    params.target = Eigen::Vector<double, 13>::Zero();
    params.target[0] = quadrotor::Value::Param::NumOpt::target_x;
    params.target[1] = quadrotor::Value::Param::NumOpt::target_y;
    params.target[2] = quadrotor::Value::Param::NumOpt::target_z;
    params.target[3] = 1;

    quadrotor::num_opt::NumOptimizer opt(params);

    //while (true)
    {
        
        Eigen::Vector4d controls = opt.next(q.get_state(), params.target);
        /*
        Eigen::MatrixXd controls = opt.solve();
        std::cout << "Control matrix:" << std::endl;
        std::cout << controls << std::endl;
        

        for (int i = 0; i < 100; ++i)
        {
            Eigen::Vector4d c = opt.next(q.get_state(), params.target);
            q.update(c, params.dt);
            std::cout << "State:" << q.get_state().transpose() << std::endl;
        }
        // std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
    */
    /*
    symnn::SymNN symnn(3, 2, std::vector<int>{4, 2}, symnn::activation::Lrelu, false);

    Eigen::MatrixXd input(3, 3);
    input << 1, 1, 1,
        2, 2, 2,
        3, 2, 10;

    std::cout << "Input: " << std::endl;
    std::cout << input << std::endl;

    Eigen::MatrixXd target(2, 3);
    target << 4, 5, 50,
        5, 4, 40;

    std::cout << "Target: " << std::endl;
    std::cout << target << std::endl;

    symnn.train(input, target);

    std::cout << symnn.forward((Eigen::Vector3d() << 1, 2, 3).finished()).transpose() << std::endl;
    std::cout << symnn.forward((Eigen::Vector3d() << 1, 2, 2).finished()).transpose() << std::endl;
    std::cout << symnn.forward((Eigen::Vector3d() << 1, 2, 10).finished()).transpose() << std::endl;
    */

    
    quadrotor::Value::target << quadrotor::Value::Param::NumOpt::target_x, quadrotor::Value::Param::NumOpt::target_y, quadrotor::Value::Param::NumOpt::target_z, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0;

    symnn::Params params;
    params.input_size = 17;
    params.output_size = 6;
    params.hidden_layers = std::vector<int>{6, 4};
    params.activation = symnn::activation::sigmoid;
    params.gradient_based = quadrotor::Value::Param::SymNN::gradient_based;
    params.epochs = quadrotor::Value::Param::SymNN::epochs;
    params.learning_rate = quadrotor::Value::Param::SymNN::learning_rate;
    params.momentum = quadrotor::Value::Param::SymNN::momentum;
    quadrotor::Value::Param::SymNN::learned_model = std::make_unique<symnn::SymNN>(params);

    double masses[] = {2, 4, 8};
    std::vector<std::vector<double>> errors_per_episode(pq::Value::Param::Train::runs * pq::Value::Param::Train::episodes);

    for (int i = 0; i < 3; ++i)
    {
        quadrotor::num_opt::Params params;
        params.m = masses[i];
        params.I = masses[i] / quadrotor::Value::Constant::mass * quadrotor::Value::Constant::I;

        quadrotor::num_opt::NumOptimizer optimizer(params);

        std::cout << "Running with mass = " << masses[i] << std::endl;

        for (int j = 0; j < pq::Value::Param::Train::runs; ++j)
        {
            std::cout << "Run " << j << std::endl;
            quadrotor::Value::Param::SymNN::learned_model->reset();
            quadrotor::Value::Param::NumOpt::use_learned = false;
            quadrotor::train::Episode episode;

            int run_idx = j * pq::Value::Param::Train::episodes;
            for (int k = 0; k < pq::Value::Param::Train::episodes; ++k)
            {
                std::cout << "Episode " << k << std::endl;
                episode.run(optimizer);

                //std::cout << episode.get_train_input().block(0, 0, 17, episode.get_stop_step()) << std::endl;
                //std::cout << episode.get_train_target().block(0, 0, 6, episode.get_stop_step()) << std::endl;

                quadrotor::Value::Param::SymNN::learned_model->train(episode.get_train_input(), episode.get_train_target(), episode.get_stop_step());
                /*
                std::cout << "NN expected:" << std::endl;
                std::cout << episode.get_train_target().block(0, 0, 6, episode.get_stop_step()).transpose() << std::endl;
                std::cout << "NN actual:" << std::endl;
                for (int l = 0; l < episode.get_stop_step(); ++l)
                {
                    std::cout << quadrotor::Value::Param::SymNN::learned_model->forward(episode.get_train_input().col(l)).transpose() << std::endl;
                }
                */
                quadrotor::Value::Param::NumOpt::use_learned = true;
            }
        }
    }

    return 0;
}