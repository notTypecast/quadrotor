#include <Eigen/Core>
#include <iostream>
#include <chrono>
#include <sstream>
#include <iomanip>
#include <fstream>
#include <memory>

#define NUM_OPT 1

#include "src/params.hpp"
#include "src/sim/PlanarQuadrotor.hpp"
#include "src/sim/Visualizer.hpp"
#include "src/opt/num-opt/SymNN.hpp"
#include "src/opt/num-opt/NumOptimizer.hpp"
#include "src/train/Episode.hpp"

int main(int argc, char **argv)
{
    /*
    symnn::SymNN symnn(3, 2, std::vector<int>{4, 2});

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

    pq::Value::target << pq::Value::Param::CEMOpt::target_x, pq::Value::Param::CEMOpt::target_y, 0, 0, 0, 0;

    symnn::Params params;
    params.input_size = 8;
    params.output_size = 3;
    params.hidden_layers = std::vector<int>{6, 4};
    params.gradient_based = pq::Value::Param::SymNN::gradient_based;
    params.epochs = pq::Value::Param::SymNN::epochs;
    params.learning_rate = pq::Value::Param::SymNN::learning_rate;
    params.momentum = pq::Value::Param::SymNN::momentum;
    pq::Value::Param::SymNN::learned_model = std::make_unique<symnn::SymNN>(params);


    double masses[] = {4, 8, 16};
    std::vector<std::vector<double>> errors_per_episode(pq::Value::Param::Train::runs * pq::Value::Param::Train::episodes);
    pq::sim::Visualizer v;

    for (int i = 0; i < 3; ++i)
    {
        pq::num_opt::Params params;
        params.m = masses[i];
        params.I = 0.2 * masses[i] * pq::Value::Param::CEMOpt::length * pq::Value::Param::CEMOpt::length;
        params.init = Eigen::Vector<double, 6>::Zero();

        pq::num_opt::NumOptimizer optimizer(params);

        std::cout << "Running with mass = " << masses[i] << std::endl;

        for (int j = 0; j < pq::Value::Param::Train::runs; ++j)
        {
            std::srand(std::time(NULL));
            std::cout << "Run " << j << std::endl;
            pq::Value::Param::SymNN::learned_model->reset();
            pq::Value::Param::NumOpt::use_learned = false;
            pq::train::Episode episode;
            episode.set_run(j + 1);

            int run_idx = j * pq::Value::Param::Train::episodes;
            std::cout << "Episode: ";
            for (int k = 0; k < pq::Value::Param::Train::episodes; ++k)
            {
                std::cout << k << " " << std::flush;
                errors_per_episode[run_idx + k] = episode.run(optimizer, v);
                /*
                // Early stopping if the change in error is smaller than a threshold
                // This will not work if writing to a file
                if (k > 0 && errors_per_episode[run_idx + k - 1][pq::Value::Param::Train::collection_steps - 1] - errors_per_episode[run_idx + k][pq::Value::Param::Train::collection_steps - 1] < 10e-4)
                {
                    std::cout << ":done";
                    break;
                }*/

                pq::Value::Param::SymNN::learned_model->train(episode.get_train_input(), episode.get_train_target(), episode.get_stop_step());
                pq::Value::Param::NumOpt::use_learned = true;
                std::cout << episode.get_train_target().col(0).transpose() << std::endl;
                std::cout << "NN sample: " << pq::Value::Param::SymNN::learned_model->forward(episode.get_train_input().col(0)).transpose() << std::endl;
            }
            std::cout << std::endl;
        }

        std::ofstream out("sample_error/error_" + std::to_string(pq::Value::Param::CEMOpt::mass) + ".txt");
        out << pq::Value::Param::CEMOpt::mass << " "
            << pq::Value::Param::Train::collection_steps << " "
            << pq::Value::Param::Train::episodes << " "
            << pq::Value::Param::Train::runs << " "
            << std::endl;
        for (int j = 0; j < pq::Value::Param::Train::runs; ++j)
        {
            int run_idx = j * pq::Value::Param::Train::episodes;
            for (int k = 0; k < pq::Value::Param::Train::episodes; ++k)
            {
                for (int l = 0; l < pq::Value::Param::Train::collection_steps; ++l)
                {
                    out << errors_per_episode[run_idx + k][l] << " ";
                }
                out << std::endl;
            }
        }
        out.close();
    }

    return 0;
}