#ifndef PQ_SYMNN_HPP
#define PQ_SYMNN_HPP

#include <vector>
#include <functional>

#include <Eigen/Core>
#include <casadi/casadi.hpp>

#include "src/params.hpp"

using namespace casadi;

namespace symnn
{
    namespace activation
    {
        MX sigmoid(const MX &x)
        {
            return 1 / (1 + exp(-x));
        }

        MX softmax(const MX &x)
        {
            return exp(x) / sum1(exp(x));
        }

        MX relu(const MX &x)
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
    }

    struct Params
    {
        int input_size;
        int output_size;
        std::vector<int> hidden_layers;
        std::function<MX(MX)> activation = activation::sigmoid;
        bool gradient_based = true;
        // Gradient-based only parameters
        int epochs = 10000;
        double learning_rate = 0.01;
        double momentum = 0;
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
                                            _momentum(params.momentum)
        {
            _X = MX::sym("X", _input_size);
            _Y = MX::sym("Y", _output_size);

            std::vector<MX> all_params;
            all_params.push_back(_X);

            std::vector<MX> flat_params;

            _W.push_back(MX::sym("W0", params.hidden_layers[0], _input_size));
            _b.push_back(MX::sym("b0", params.hidden_layers[0]));
            MX prev = params.activation(mtimes(_W[0], _X) + _b[0]);

            all_params.push_back(_W[0]);
            all_params.push_back(_b[0]);

            flat_params.push_back(reshape(_W[0], -1, 1));
            flat_params.push_back(_b[0]);

            for (int i = 1; i < params.hidden_layers.size(); ++i)
            {
                _W.push_back(MX::sym("W" + std::to_string(i), params.hidden_layers[i], params.hidden_layers[i - 1]));
                _b.push_back(MX::sym("b" + std::to_string(i), params.hidden_layers[i]));
                prev = params.activation(mtimes(_W[i], prev) + _b[i]);

                all_params.push_back(_W[i]);
                all_params.push_back(_b[i]);

                flat_params.push_back(reshape(_W[i], -1, 1));
                flat_params.push_back(_b[i]);
            }

            _W.push_back(MX::sym("Wout", params.output_size, params.hidden_layers.back()));
            _b.push_back(MX::sym("bout", params.output_size));

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

            for (int i = 1; i < all_params.size(); ++i)
            {
                _nn_values.push_back(DM::rand(all_params[i].size1(), all_params[i].size2()) * (i % 2 ? sqrt(2.0 / all_params[i].size1()) : 1));
            }

            std::vector<MX> opt_vars(all_params.size() - 1);
            for (int i = 0; i < _W.size(); ++i)
            {
                opt_vars[2 * i] = reshape(_W[i], -1, 1);
                opt_vars[2 * i + 1] = _b[i];
            }

            _opt_vars = vertcat(opt_vars);
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
            for (int i = 0; i < _nn_values.size(); ++i)
            {
                _nn_values[i] = DM::rand(_nn_values[i].size1(), _nn_values[i].size2()) * (i % 2 ? sqrt(2.0 / _nn_values[i].size1()) : 1);
            }

            _out_substituted = _out;
            for (int i = 0; i < _W.size(); ++i)
            {
                _out_substituted = substitute(_out_substituted, _W[i], _nn_values[2 * i]);
                _out_substituted = substitute(_out_substituted, _b[i], _nn_values[2 * i + 1]);
            }
        }

    protected:
        bool _gradient_based;
        int _input_size, _output_size;
        int _epochs;
        double _learning_rate, _momentum;
        int _total_size;
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