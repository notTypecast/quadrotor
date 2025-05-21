#include <Eigen/Core>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>

#define CEM_OPT 1

#include "src/opt/cem-opt/CEMOptimizer.hpp"
#include "src/opt/cem-opt/LearnedModel.hpp"
#include "src/params.hpp"
#include "src/sim/PlanarQuadrotor.hpp"
#include "src/sim/Visualizer.hpp"
#include "src/train/Episode.hpp"

int main(int argc, char **argv)
{
    pq::Value::target << pq::Value::Param::CEMOpt::target_x,
      pq::Value::Param::CEMOpt::target_y, 0, 0, 0, 0;
    pq::Value::Param::SimpleNN::learned_model =
      std::make_unique<pq::cem_opt::NNModel>(std::vector<int> { 6, 4 });

    double                           masses[] = { 4, 8, 16 };
    std::vector<std::vector<double>> errors_per_episode(
      pq::Value::Param::Train::runs * pq::Value::Param::Train::episodes);
    pq::sim::Visualizer v;

    Eigen::MatrixXd train_input, train_target;

    for (int i = 0; i < 3; ++i)
    {
        pq::cem_opt::Params params;
        params.m = masses[i];
        params.I = 0.2 * masses[i] * pq::Value::Param::CEMOpt::length *
                   pq::Value::Param::CEMOpt::length;

        pq::cem_opt::CEMOptimizer optimizer(params);

        std::cout << "Running with mass = " << masses[i] << std::endl;

        for (int j = 0; j < pq::Value::Param::Train::runs; ++j)
        {
            std::srand(std::time(NULL));
            std::cout << "Run " << j << std::endl;
            pq::Value::Param::SimpleNN::learned_model->reset();
            pq::train::Episode episode;
            episode.set_run(j + 1);

            int run_idx = j * pq::Value::Param::Train::episodes;
            std::cout << "Episode: ";
            for (int k = 0; k < pq::Value::Param::Train::episodes; ++k)
            {
                std::cout << k << " " << std::flush;
                errors_per_episode[run_idx + k] = episode.run(optimizer, v);
                /*
                // Early stopping if the change in error is smaller than a
                threshold
                // This will not work if writing to a file
                if (k > 0 && errors_per_episode[run_idx + k -
                1][pq::Value::Param::Train::collection_steps - 1] -
                errors_per_episode[run_idx +
                k][pq::Value::Param::Train::collection_steps - 1] < 10e-4)
                {
                    std::cout << ":done";
                    break;
                }
                */

                /*
                std::cout << episode.get_train_target().col(0).transpose() <<
                std::endl; std::cout << "NN sample: " <<
                pq::Value::Param::SimpleNN::learned_model->predict(episode.get_train_input().col(0)).transpose()
                << std::endl;
                */

                if (pq::Value::Param::SimpleNN::use_all_data)
                {
                    Eigen::MatrixXd new_input  = episode.get_train_input();
                    Eigen::MatrixXd new_target = episode.get_train_target();
                    if (train_input.size() == 0)
                    {
                        train_input  = new_input;
                        train_target = new_target;
                    }
                    else
                    {
                        train_input.conservativeResize(train_input.rows(),
                                                       train_input.cols() +
                                                         new_input.cols());
                        train_target.conservativeResize(train_target.rows(),
                                                        train_target.cols() +
                                                          new_target.cols());
                        train_input.block(0,
                                          train_input.cols() - new_input.cols(),
                                          train_input.rows(),
                                          new_input.cols())   = new_input;
                        train_target.block(0,
                                           train_target.cols() -
                                             new_target.cols(),
                                           train_target.rows(),
                                           new_target.cols()) = new_target;
                    }
                }
                else
                {
                    train_input  = episode.get_train_input();
                    train_target = episode.get_train_target();
                }

                pq::Value::Param::SimpleNN::learned_model->train(train_input,
                                                                 train_target);

                std::cout << "NN expected:" << std::endl;
                std::cout << episode.get_train_target()
                               .block(0,
                                      0,
                                      3,
                                      episode.get_train_target().cols())
                               .transpose()
                          << std::endl;
                std::cout << "NN actual:" << std::endl;
                for (int l = 0; l < episode.get_train_target().cols(); ++l)
                {
                    std::cout << pq::Value::Param::SimpleNN::learned_model
                                   ->predict(episode.get_train_input().col(l))
                                   .transpose()
                              << std::endl;
                }
            }
            std::cout << std::endl;
        }

        std::ofstream out("sample_error/error_" +
                          std::to_string(pq::Value::Param::CEMOpt::mass) +
                          ".txt");
        out << pq::Value::Param::CEMOpt::mass << " "
            << pq::Value::Param::Train::collection_steps << " "
            << pq::Value::Param::Train::episodes << " "
            << pq::Value::Param::Train::runs << " " << std::endl;
        for (int j = 0; j < pq::Value::Param::Train::runs; ++j)
        {
            int run_idx = j * pq::Value::Param::Train::episodes;
            for (int k = 0; k < pq::Value::Param::Train::episodes; ++k)
            {
                for (int l = 0; l < pq::Value::Param::Train::collection_steps;
                     ++l)
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