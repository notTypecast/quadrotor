#include <Eigen/Core>
#include <casadi/casadi.hpp>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>

#ifdef QUAD_WITH_MPI
#include <mpi/mpi.h>
MPI_Datatype error_type;
#endif

#define NUM_OPT 1

#include "src/opt/num-opt/NumOptimizer.hpp"
#include "src/opt/num-opt/SymNN.hpp"
#include "src/params.hpp"
#include "src/sim/PlanarQuadrotor.hpp"
#include "src/sim/Visualizer.hpp"
#include "src/train/Episode.hpp"

void execute(pq::train::Errors base_error, std::string name_prefix = "",
             bool print = true)
{
    double              masses[] = { 4, 8, 16 };
    pq::sim::Visualizer v;

    symnn::Params nn_params;
    nn_params.input_size       = 8;
    nn_params.output_size      = 3;
    nn_params.hidden_layers    = std::vector<int> { 8, 4 };
    nn_params.activation       = symnn::activation::SIGMOID;
    nn_params.initializer      = symnn::initializers::NXavier;
    nn_params.optimizer        = symnn::OPTIMIZER::ADAM;
    nn_params.epochs           = pq::Value::Param::SymNN::epochs;
    nn_params.dropout_rate     = pq::Value::Param::SymNN::dropout_rate;
    nn_params.inference_passes = pq::Value::Param::SymNN::inference_passes;

    for (int i = 0; i < sizeof(masses) / sizeof(masses[0]); ++i)
    {
        std::string mass_str = std::to_string(masses[i]);
        mass_str.erase(mass_str.find_last_not_of('0') + 1, std::string::npos);
        mass_str.erase(mass_str.find_last_not_of('.') + 1, std::string::npos);

        pq::num_opt::Params opt_params;
        opt_params.m = masses[i];
        opt_params.I = 0.2 * masses[i] * pq::Value::Constant::length *
                       pq::Value::Constant::length;
        opt_params.init = Eigen::Vector<double, 6>::Zero();

        if (print)
            std::cout << "Running with mass = " << masses[i] << std::endl;

        int runs = pq::Value::Param::Train::runs;

#ifdef QUAD_WITH_MPI
        int size, rank;
        MPI_Comm_size(MPI_COMM_WORLD, &size);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);

        runs /= size;

        if (rank == 0 && pq::Value::Param::Train::runs % size != 0)
        {
            std::cout << "Number of runs (" << pq::Value::Param::Train::runs
                      << ") not perfectly divisible by number of processes ("
                      << size << ")" << std::endl;
            std::cout << "Performing " << runs * size << " runs instead"
                      << std::endl;
        }
#endif

        std::vector<pq::train::Errors> errors(
          runs * pq::Value::Param::Train::episodes);

        for (int j = 0; j < runs; ++j)
        {
            if (print) std::cout << "Run " << j << std::endl;
            Eigen::MatrixXd           train_input, train_target;
            pq::num_opt::NumOptimizer optimizer(opt_params, nn_params);
            pq::train::Episode        episode;

            for (int k = 0; k < pq::Value::Param::Train::episodes; ++k)
            {
                if (print)
                {
                    std::cout << "Episode " << k << std::endl;
                    errors[j * pq::Value::Param::Train::episodes + k] =
                      episode.run(optimizer, v);
                }
                else
                {
                    errors[j * pq::Value::Param::Train::episodes + k] =
                      episode.run(optimizer);
                }

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
                                       train_target.cols() - new_target.cols(),
                                       train_target.rows(),
                                       new_target.cols()) = new_target;
                }

                optimizer.learned_model.train(train_input, train_target);
            }

            optimizer.learned_model.save("src/train/models/" + name_prefix +
                                         "pq_model_" + mass_str + "_" +
                                         std::to_string(j));
        }

#ifdef QUAD_WITH_MPI
        std::vector<pq::train::Errors> all_errors;

        if (rank == 0)
        {
            all_errors.resize(runs * size * pq::Value::Param::Train::episodes);
        }

        MPI_Gather(errors.data(),
                   errors.size(),
                   error_type,
                   all_errors.data(),
                   errors.size(),
                   error_type,
                   0,
                   MPI_COMM_WORLD);

        if (rank == 0)
        {
            std::ofstream out("sample_error/" + name_prefix +
                              symnn::layer_str(nn_params) + "_pq_error_" +
                              mass_str + ".txt");
            out << mass_str << " " << pq::Value::Param::Train::collection_steps
                << " " << pq::Value::Param::Train::episodes << " "
                << runs * size << " " << std::endl;
            out << base_error << std::endl;
            for (int j = 0; j < pq::Value::Param::Train::runs; ++j)
            {
                int run_idx = j * pq::Value::Param::Train::episodes;
                for (int k = 0; k < pq::Value::Param::Train::episodes; ++k)
                {
                    out << all_errors[j * pq::Value::Param::Train::episodes + k]
                        << " ";
                }
                out << std::endl;
            }
            out.close();
        }
#else

        std::ofstream out("sample_error/" + name_prefix +
                          symnn::layer_str(nn_params) + "_pq_error_" +
                          mass_str + ".txt");
        out << mass_str << " " << pq::Value::Param::Train::collection_steps
            << " " << pq::Value::Param::Train::episodes << " "
            << pq::Value::Param::Train::runs << " " << std::endl;
        out << base_error << std::endl;
        for (int j = 0; j < pq::Value::Param::Train::runs; ++j)
        {
            int run_idx = j * pq::Value::Param::Train::episodes;
            for (int k = 0; k < pq::Value::Param::Train::episodes; ++k)
            {
                out << errors[j * pq::Value::Param::Train::episodes + k] << " ";
            }
            out << std::endl;
        }
        out.close();
#endif
    }
}

pq::train::Errors baseline()
{
    pq::num_opt::Params opt_params;
    opt_params.init = Eigen::Vector<double, 6>::Zero();

    symnn::Params nn_params;
    nn_params.input_size  = 1;
    nn_params.output_size = 1;

    pq::num_opt::NumOptimizer optimizer(opt_params, nn_params);
    pq::train::Episode        episode;

    return episode.run(optimizer);
}

int main(int argc, char **argv)
{
    pq::train::Errors base_error;

#ifdef QUAD_WITH_MPI
    MPI_Init(NULL, NULL);

    int lengths[6] = { 1, 1, 1, 1, 1, 1 };

    MPI_Aint          displacements[6];
    pq::train::Errors dummy_error;
    MPI_Aint          base_addr;
    MPI_Get_address(&dummy_error, &base_addr);
    MPI_Get_address(&dummy_error.failed, &displacements[0]);
    MPI_Get_address(&dummy_error.steps, &displacements[1]);
    MPI_Get_address(&dummy_error.position, &displacements[2]);
    MPI_Get_address(&dummy_error.orientation, &displacements[3]);
    MPI_Get_address(&dummy_error.velocity, &displacements[4]);
    MPI_Get_address(&dummy_error.angular_velocity, &displacements[5]);
    displacements[0] = MPI_Aint_diff(displacements[0], base_addr);
    displacements[1] = MPI_Aint_diff(displacements[1], base_addr);
    displacements[2] = MPI_Aint_diff(displacements[2], base_addr);
    displacements[3] = MPI_Aint_diff(displacements[3], base_addr);
    displacements[4] = MPI_Aint_diff(displacements[4], base_addr);
    displacements[5] = MPI_Aint_diff(displacements[5], base_addr);

    MPI_Datatype types[6] = { MPI_CXX_BOOL, MPI_INT,    MPI_DOUBLE,
                              MPI_DOUBLE,   MPI_DOUBLE, MPI_DOUBLE };
    MPI_Type_create_struct(6, lengths, displacements, types, &error_type);
    MPI_Type_commit(&error_type);

    pq::Value::target << pq::Value::Param::NumOpt::target_x,
      pq::Value::Param::NumOpt::target_y, 0, 0, 0, 0;

    if (pq::Value::Param::NumOpt::baseline)
    {
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        if (rank == 0)
        {
            base_error = baseline();
        }
    }
#else
    pq::Value::target << pq::Value::Param::NumOpt::target_x,
      pq::Value::Param::NumOpt::target_y, 0, 0, 0, 0;
    if (pq::Value::Param::NumOpt::baseline)
    {
        base_error = baseline();
    }
#endif

#if defined(TEST)
    execute(base_error);
#else
    // Dropout with Variance
    // execute(base_error, "DV_", false);

    // Dropout without Variance
    // pq::Value::Param::NumOpt::nn_variance_weight = 0.0;
    // execute(base_error, "D_", false);

    // No dropout
    pq::Value::Param::SymNN::dropout_rate     = 0.0;
    pq::Value::Param::SymNN::inference_passes = 1;
    execute(base_error, "reg_", false);
#endif

#ifdef QUAD_WITH_MPI
    MPI_Finalize();
#endif

    return 0;
}