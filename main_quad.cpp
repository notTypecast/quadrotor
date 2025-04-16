#include <Eigen/Core>
#include <chrono>
#include <iostream>
#include <thread>

#define NUM_OPT 1

#include "src/opt/num-opt/NumOptimizer.hpp"
#include "src/opt/num-opt/SymNN.hpp"
#include "src/params.hpp"
#include "src/sim/Quadrotor.hpp"
#include "src/train/Episode.hpp"

void execute(std::string name_prefix = "")
{
    double masses[] = { 1, 2, 4, 6 };
    std::vector<std::vector<quadrotor::train::Errors>> errors(
      quadrotor::Value::Param::Train::runs,
      std::vector<quadrotor::train::Errors>(
        quadrotor::Value::Param::Train::episodes));

    for (int i = 0; i < sizeof(masses) / sizeof(masses[0]); ++i)
    {
        std::string mass_str = std::to_string(masses[i]);
        mass_str.erase(mass_str.find_last_not_of('0') + 1, std::string::npos);
        mass_str.erase(mass_str.find_last_not_of('.') + 1, std::string::npos);

        Eigen::MatrixXd train_input, train_target;

        quadrotor::num_opt::Params params;
        params.m = masses[i];
        params.I = masses[i] / quadrotor::Value::Constant::mass *
                   quadrotor::Value::Constant::I;

        quadrotor::num_opt::NumOptimizer optimizer(params);

        std::cout << "Running with mass = " << masses[i] << std::endl;

        for (int j = 0; j < quadrotor::Value::Param::Train::runs; ++j)
        {
            std::cout << "Run " << j << std::endl;
            quadrotor::Value::Param::SymNN::learned_model->reset();
            quadrotor::Value::Param::NumOpt::use_learned = false;
            quadrotor::train::Episode episode("src/train/data/quad.txt");

            for (int k = 0; k < quadrotor::Value::Param::Train::episodes; ++k)
            {
                std::cout << "Episode " << k << std::endl;
                errors[j][k] = episode.run(optimizer);

                if (quadrotor::Value::Param::SymNN::use_all_data)
                {
                    Eigen::MatrixXd new_input  = episode.get_train_input();
                    Eigen::MatrixXd new_target = episode.get_train_target();
                    if (k == 0)
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

                quadrotor::Value::Param::SymNN::learned_model->train(
                  train_input,
                  train_target);

                /*
                std::cout << "NN expected:" << std::endl;
                std::cout << episode.get_train_target()
                               .block(0, 0, 6, episode.get_stop_step())
                               .transpose()
                          << std::endl;
                std::cout << "NN actual:" << std::endl;
                for (int l = 0; l < episode.get_stop_step(); ++l)
                {
                    std::cout << quadrotor::Value::Param::SymNN::learned_model
                                   ->forward(episode.get_train_input().col(l))
                                   .transpose()
                              << std::endl;
                }
                */

                quadrotor::Value::Param::NumOpt::use_learned = true;
            }

            // std::cout << "Episode with completed training" << std::endl;
            // episode.run(optimizer, true);

            quadrotor::Value::Param::SymNN::learned_model->save(
              "src/train/models/" + name_prefix + "quad_model_" + mass_str +
              "_" + std::to_string(j));
        }

        std::ofstream out("sample_error/" + name_prefix + "quad_error_" +
                          mass_str + ".txt");
        out << mass_str << " "
            << quadrotor::Value::Param::Train::collection_steps << " "
            << quadrotor::Value::Param::Train::episodes << " "
            << quadrotor::Value::Param::Train::runs << " " << std::endl;
        for (int j = 0; j < quadrotor::Value::Param::Train::runs; ++j)
        {
            int run_idx = j * quadrotor::Value::Param::Train::episodes;
            for (int k = 0; k < quadrotor::Value::Param::Train::episodes; ++k)
            {
                out << errors[j][k] << " ";
            }
            out << std::endl;
        }
        out.close();
    }
}

int main()
{
    quadrotor::Value::target << quadrotor::Value::Param::NumOpt::target_x,
      quadrotor::Value::Param::NumOpt::target_y,
      quadrotor::Value::Param::NumOpt::target_z, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0;

    symnn::Params params;
    params.input_size    = 17;
    params.output_size   = 6;
    params.hidden_layers = std::vector<int> { 4 };
    params.activation    = symnn::activation::SIGMOID;
    params.initializer   = symnn::initializers::NXavier;
    params.optimizer     = symnn::OPTIMIZER::ADAM;
    params.epochs        = 2000;
    quadrotor::Value::Param::SymNN::learned_model =
      std::make_unique<symnn::SymNN>(params);

#if defined(DV)
    // Dropout with Variance
    std::cout << "Running DV" << std::endl;
    execute("DV_");
#elif defined(DNV)

    // Dropout without Variance
    quadrotor::Value::Param::NumOpt::nn_variance_weight = 0.0;
    std::cout << "Running DNV" << std::endl;
    execute("D_");
#elif defined(REG)
    // No dropout
    quadrotor::Value::Param::NumOpt::nn_variance_weight = 0.0;
    params.dropout_rate                                 = 0;
    params.inference_passes                             = 1;
    quadrotor::Value::Param::SymNN::learned_model =
      std::make_unique<symnn::SymNN>(params);
    std::cout << "Running reg" << std::endl;
    execute("reg_");
#else
    execute();
#endif

    return 0;
}