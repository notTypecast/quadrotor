#ifndef PQ_SYMNN_HPP
#define PQ_SYMNN_HPP

#include <vector>
#include <functional>
#include <string>
#include <unordered_map>

#include <Eigen/Core>
#include <casadi/casadi.hpp>

#include "src/params.hpp"

using namespace casadi;

namespace symnn
{
    namespace activation
    {
        MX Sigmoid(const MX &x)
        {
            return 1 / (1 + exp(-x));
        }

        MX Softmax(const MX &x)
        {
            return exp(x) / sum1(exp(x));
        }

        MX Relu(const MX &x)
        {
            return fmax(0, x);
        }

        MX Lrelu(const MX &x)
        {
            return fmax(0.01 * x, x);
        }

        MX ELU(const MX &x)
        {
            return if_else(x > 0, x, 0.01 * (exp(x) - 1));
        }

        MX Tanh(const MX &x)
        {
            return tanh(x);
        }

        std::unordered_map<std::string, std::function<MX(const MX &)>> activation_map = {
            {"sigmoid", Sigmoid},
            {"softmax", Softmax},
            {"relu", Relu},
            {"lrelu", Lrelu},
            {"elu", ELU},
            {"tanh", Tanh}};
    }

    namespace initializers
    {
        /* Initializer inputs:
         * rows: number of rows in the matrix
         *  also represents the number of neurons in the current layer
         * cols: number of columns in the matrix
         *  also represents the number of neurons in the previous layer (input size)
         * next: number of neurons in the next layer (output size)
         */

        // Zero initialization
        // next is used for signature only
        DM Zero(int rows, int cols, int next)
        {
            return DM::zeros(rows, cols);
        }

        // Random initialization
        // next is used for signature only
        DM Random(int rows, int cols, int next)
        {
            return DM::rand(rows, cols);
        }

        // He initialization
        // next is used for signature only
        DM He(int rows, int cols, int next)
        {
            return DM::rand(rows, cols) * sqrt(2.0 / rows);
        }

        // Xavier initialization
        // next is used for signature only
        DM Xavier(int rows, int cols, int next)
        {
            double lower = -sqrt(1 / cols);
            double upper = -lower;
            return DM::rand(rows, cols) * (upper - lower) + lower;
        }

        // Normalized Xavier initialization
        DM NXavier(int rows, int cols, int next)
        {
            double lower = -sqrt(6.0 / (cols + next));
            double upper = -lower;
            return DM::rand(rows, cols) * (upper - lower) + lower;
        }
    }

    struct Params
    {
        int input_size;
        int output_size;
        std::vector<int> hidden_layers;
        std::string activation;
        std::function<DM(int, int, int)> initializer = initializers::NXavier;
        bool gradient_based = true;
        // Gradient-based only parameters
        int epochs = 10000;
        double learning_rate = 0.01;
        double momentum = 0;
        double max_grad = -1;
    };

    // Fully connected NN
    class SymNN
    {
    public:
        SymNN(const Params &params) : _input_size(params.input_size),
                                      _output_size(params.output_size),
                                      _gradient_based(params.gradient_based),
                                      _epochs(params.epochs),
                                      _learning_rate(params.learning_rate),
                                      _momentum(params.momentum),
                                      _max_grad(params.max_grad),
                                      _activation_name(params.activation),
                                      _initializer(params.initializer)
        {
            _construct(params.hidden_layers);
        }

        SymNN(const std::string &filename, Params &params) : _gradient_based(params.gradient_based),
                                                             _epochs(params.epochs),
                                                             _learning_rate(params.learning_rate),
                                                             _momentum(params.momentum),
                                                             _max_grad(params.max_grad),
                                                             _initializer(params.initializer)
        {
            std::ifstream file(filename);

            file >> _activation_name;
            file >> _input_size;
            file >> _output_size;

            params.hidden_layers.clear();

            std::string line;
            std::getline(file, line);
            std::getline(file, line);
            std::istringstream iss(line);

            int num;
            while (iss >> num)
            {
                params.hidden_layers.push_back(num);
            }

            int prev_size = _input_size;
            double val;

            params.hidden_layers.push_back(_output_size);

            for (int i = 0; i < params.hidden_layers.size(); ++i)
            {
                _nn_values.push_back(DM(params.hidden_layers[i], prev_size));

                for (int j = 0; j < params.hidden_layers[i]; ++j)
                {
                    for (int k = 0; k < prev_size; ++k)
                    {
                        file >> val;
                        _nn_values.back()(j, k) = val;
                    }
                }

                _nn_values.push_back(DM(params.hidden_layers[i], 1));

                for (int j = 0; j < params.hidden_layers[i]; ++j)
                {
                    file >> val;
                    _nn_values.back()(j) = val;
                }

                prev_size = params.hidden_layers[i];
            }

            params.hidden_layers.pop_back();
            
            _construct(params.hidden_layers);
        }

        Eigen::VectorXd forward(const Eigen::VectorXd &input)
        {
            DM X = DM(input.size(), 1);
            for (int i = 0; i < input.size(); ++i)
            {
                X(i) = input(i);
            }

            std::vector<DM> params(_nn_values.begin(), _nn_values.end());
            params.insert(params.begin(), X);

            DM out = _out_fn(params)[0];

            Eigen::VectorXd output(_output_size);

            for (int i = 0; i < _output_size; ++i)
            {
                output(i) = static_cast<double>(out(i));
            }

            return output;
        }

        MX forward(MX &input)
        {
            return substitute(_out_substituted, _X, input);
        }

        void train(const Eigen::MatrixXd &input, const Eigen::MatrixXd &target, int stop_col = -1)
        {
            if (_gradient_based)
            {
                _train_gd(input, target, stop_col);
            }
            else
            {
                _train_ipopt(input, target, stop_col);
            }

            _out_substituted = _out;
            for (int i = 0; i < _W.size(); ++i)
            {
                _out_substituted = substitute(_out_substituted, _W[i], _nn_values[2 * i]);
                _out_substituted = substitute(_out_substituted, _b[i], _nn_values[2 * i + 1]);
            }
        }

        void reset()
        {
            for (int i = 0; i < _nn_values.size() - 2; ++i)
            {
                if (i % 2)
                {
                    _nn_values[i] = DM::rand(_nn_values[i].size1(), _nn_values[i].size2());
                }
                else
                {
                    _nn_values[i] = _initializer(_nn_values[i].size1(), _nn_values[i].size2(), _nn_values[i + 2].size1());
                }
            }

            _nn_values[_nn_values.size() - 2] = DM::rand(_nn_values[_nn_values.size() - 2].size1(), _nn_values[_nn_values.size() - 2].size2());
            _nn_values[_nn_values.size() - 1] = DM::rand(_nn_values[_nn_values.size() - 1].size1(), _nn_values[_nn_values.size() - 1].size2());

            _out_substituted = _out;
            for (int i = 0; i < _W.size(); ++i)
            {
                _out_substituted = substitute(_out_substituted, _W[i], _nn_values[2 * i]);
                _out_substituted = substitute(_out_substituted, _b[i], _nn_values[2 * i + 1]);
            }
        }

        void save(const std::string &filename)
        {
            std::ofstream file(filename);

            file << _activation_name << std::endl;
            file << _input_size << std::endl;
            file << _output_size << std::endl;

            for (int i = 0; i < _W.size() - 1; ++i)
            {
                file << _W[i].size1();
                if (i < _W.size() - 1)
                {
                    file << " ";
                }
            }
            file << std::endl;

            for (int i = 0; i < _nn_values.size(); ++i)
            {
                for (int j = 0; j < _nn_values[i].size1(); ++j)
                {
                    for (int k = 0; k < _nn_values[i].size2(); ++k)
                    {
                        file << _nn_values[i](j, k);
                        if (k < _nn_values[i].size2() - 1)
                        {
                            file << " ";
                        }
                    }
                    file << std::endl;
                }
            }

            file.close();
        }

    protected:
        bool _gradient_based;
        int _input_size, _output_size;
        int _epochs;
        double _learning_rate, _momentum;
        double _max_grad;
        int _total_size;
        std::string _activation_name;
        std::function<DM(int, int, int)> _initializer;
        MX _X, _Y;
        std::vector<MX> _W, _b;
        MX _out;
        Function _out_fn;
        MX _loss;
        MX _gradients;
        Function _gradient_fn;

        std::vector<DM> _nn_values;
        MX _out_substituted;

        MX _opt_vars;

        void _construct(const std::vector<int> &hidden_layers)
        {
            if (activation::activation_map.find(_activation_name) == activation::activation_map.end())
            {
                _activation_name = "sigmoid";
            }

            std::function<MX(const MX &)> activation = activation::activation_map[_activation_name];

            _X = MX::sym("X", _input_size);
            _Y = MX::sym("Y", _output_size);

            std::vector<MX> all_params;
            all_params.push_back(_X);

            std::vector<MX> flat_params;

            MX prev = _X;

            for (int i = 0; i < hidden_layers.size(); ++i)
            {
                _W.push_back(MX::sym("W" + std::to_string(i), hidden_layers[i], prev.size1()));
                _b.push_back(MX::sym("b" + std::to_string(i), hidden_layers[i]));
                prev = activation(mtimes(_W[i], prev) + _b[i]);

                all_params.push_back(_W[i]);
                all_params.push_back(_b[i]);

                flat_params.push_back(reshape(_W[i], -1, 1));
                flat_params.push_back(_b[i]);
            }

            _W.push_back(MX::sym("Wout", _output_size, hidden_layers.back()));
            _b.push_back(MX::sym("bout", _output_size));

            all_params.push_back(_W.back());
            all_params.push_back(_b.back());

            flat_params.push_back(reshape(_W.back(), -1, 1));
            flat_params.push_back(_b.back());

            MX flat_params_var = vertcat(flat_params);

            _total_size = flat_params_var.size1();

            _out = mtimes(_W.back(), prev) + _b.back();
            _out_fn = Function("out", all_params, {_out});

            _loss = sumsqr(_Y - _out);

            _gradients = gradient(_loss, flat_params_var);
            _gradient_fn = Function("gradient_fn", {flat_params_var, _X, _Y}, {_gradients});

            std::vector<MX> opt_vars(all_params.size() - 1);
            for (int i = 0; i < _W.size(); ++i)
            {
                opt_vars[2 * i] = reshape(_W[i], -1, 1);
                opt_vars[2 * i + 1] = _b[i];
            }

            _opt_vars = vertcat(opt_vars);

            if (_nn_values.empty())
            {
                for (int i = 1; i < all_params.size() - 2; ++i)
                {
                    if (i % 2)
                    {
                        _nn_values.push_back(_initializer(all_params[i].size1(), all_params[i].size2(), all_params[i + 2].size1()));
                    }
                    else
                    {
                        _nn_values.push_back(DM::rand(all_params[i].size1(), all_params[i].size2()));
                    }
                }

                _nn_values.push_back(DM::rand(all_params[all_params.size() - 2].size1(), all_params[all_params.size() - 2].size2()));
                _nn_values.push_back(DM::rand(all_params[all_params.size() - 1].size1(), all_params[all_params.size() - 1].size2()));
            }

            _out_substituted = _out;
            for (int i = 0; i < _W.size(); ++i)
            {
                _out_substituted = substitute(_out_substituted, _W[i], _nn_values[2 * i]);
                _out_substituted = substitute(_out_substituted, _b[i], _nn_values[2 * i + 1]);
            }
        }

        MX _instance_loss(const Eigen::VectorXd &input, const Eigen::VectorXd &target)
        {
            DM X = DM(input.size(), 1);
            for (int i = 0; i < input.size(); ++i)
            {
                X(i) = input(i);
            }

            DM Y = DM(target.size(), 1);
            for (int i = 0; i < target.size(); ++i)
            {
                Y(i) = target(i);
            }

            MX loss_sub = substitute(_loss, _X, X);
            loss_sub = substitute(loss_sub, _Y, Y);

            return loss_sub;
        }

        void _train_gd(const Eigen::MatrixXd &input, const Eigen::MatrixXd &target, int stop_col = -1)
        {
            DM current_params;
            // std::cout << "First value: " << _nn_values[0] << std::endl;
            for (int i = 0; i < _nn_values.size(); i += 2)
            {
                current_params = vertcat(current_params, reshape(_nn_values[i], -1, 1));
                current_params = vertcat(current_params, _nn_values[i + 1]);
            }
            // std::cout << current_params << std::endl;

            DM X = DM(input.rows(), input.cols());
            for (int i = 0; i < input.rows(); ++i)
            {
                for (int j = 0; j < input.cols(); ++j)
                {
                    X(i, j) = input(i, j);
                }
            }

            DM Y = DM(target.rows(), target.cols());
            for (int i = 0; i < target.rows(); ++i)
            {
                for (int j = 0; j < target.cols(); ++j)
                {
                    Y(i, j) = target(i, j);
                }
            }

            for (int epoch = 0; epoch < _epochs; ++epoch)
            {
                DM prev_update = DM::zeros(current_params.size1(), current_params.size2());
                for (int j = 0; j < (stop_col == -1 ? input.cols() : stop_col); ++j)
                {
                    std::vector<DM> params = {current_params, X(Slice(), j), Y(Slice(), j)};

                    DM grad_values = _gradient_fn(params)[0];
                    // std::cout << "Gradient values: " << grad_values << std::endl;

                    DM update = _learning_rate * grad_values;
                    if (_max_grad > 0)
                    {
                        update = update / norm_2(update) * _max_grad;
                        // update = if_else(fabs(update) > _max_grad, sign(update) * _max_grad, update);
                    }
                    current_params -= update + _momentum * prev_update;
                    prev_update = update;
                }
            }

            int offset = 0;
            for (int i = 0; i < _nn_values.size(); ++i)
            {
                int param_size = _nn_values[i].size1() * _nn_values[i].size2();
                _nn_values[i] = reshape(current_params(Slice(offset, offset + param_size), 0), _nn_values[i].size1(), _nn_values[i].size2());
                offset += param_size;
            }

            // std::cout << "Weights:" << std::endl;
            // std::cout << _nn_values << std::endl;

            // std::cout << "First value now: " << current_params(0) << std::endl;
        }

        void _train_ipopt(const Eigen::MatrixXd &input, const Eigen::MatrixXd &target, int stop_col = -1)
        {
            MX loss = _instance_loss(input.col(0), target.col(0));

            for (int j = 1; j < (stop_col == -1 ? input.cols() : stop_col); ++j)
            {
                loss += _instance_loss(input.col(j), target.col(j));
            }

            MXDict nlp = {
                {"x", _opt_vars},
                {"f", loss}};

            Dict opts;
            opts["ipopt.print_level"] = 0;
            opts["print_time"] = false;
            opts["ipopt.tol"] = 1e-4;

            Function solver = nlpsol("solver", "ipopt", nlp, opts);

            std::vector<DM> params(_nn_values.size());
            for (int i = 0; i < _nn_values.size(); i += 2)
            {
                params[i] = reshape(_nn_values[i], -1, 1);
                params[i + 1] = _nn_values[i + 1];
            }

            DMDict args;
            args["x0"] = vertcat(params);

            DMDict result = solver(args);

            DM out = result.at("x");

            int offset = 0;
            for (int i = 0; i < _nn_values.size(); ++i)
            {
                int param_size = _nn_values[i].size1() * _nn_values[i].size2();
                _nn_values[i] = reshape(out(Slice(offset, offset + param_size), 0), _nn_values[i].size1(), _nn_values[i].size2());
                offset += param_size;
            }
        }
    };
}

#endif