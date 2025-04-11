#ifndef PQ_SYMNN_HPP
#define PQ_SYMNN_HPP

#include <vector>
#include <functional>
#include <string>
#include <unordered_map>
#include <random>
#include <chrono>

#include <Eigen/Core>
#include <casadi/casadi.hpp>

#include "src/params.hpp"

using namespace casadi;

namespace symnn
{
    /* randn_DM function
     * Generate a matrix of random numbers ~ N(mean, std)
     */
    DM randn_DM(int rows, int cols, double mean, double std)
    {
        std::default_random_engine gen(time(NULL));
        std::normal_distribution<double> dist(mean, std);
        DM r(rows, cols);

        for (int i = 0; i < rows; ++i)
        {
            for (int j = 0; j < cols; ++j)
            {
                r(i, j) = dist(gen);
            }
        }

        return r;
    }

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

        MX Gaussian(const MX &x)
        {
            return exp(-x * x);
        }

        std::unordered_map<std::string, std::function<MX(const MX &)>> name_to_activation = {
            {"sigmoid", Sigmoid},
            {"softmax", Softmax},
            {"relu", Relu},
            {"lrelu", Lrelu},
            {"elu", ELU},
            {"tanh", Tanh},
            {"gaussian", Gaussian}};

        std::unordered_map<std::function<MX(const MX &)>, std::string> activation_to_name = {
            {Sigmoid, "sigmoid"},
            {Softmax, "softmax"},
            {Relu, "relu"},
            {Lrelu, "lrelu"},
            {ELU, "elu"},
            {Tanh, "tanh"},
            {Gaussian, "gaussian"}};
    }

    namespace initializers
    {
        /* Initializer inputs:
         * rows: number of rows in the matrix
         *  also represents the number of neurons in the current layer
         * cols: number of columns in the matrix
         *  also represents the number of neurons in the previous layer (input size)
         */

        // Zero initialization
        DM Zero(int rows, int cols)
        {
            return DM::zeros(rows, cols);
        }

        // Uniform random initialization
        // ~U(0, 1)
        DM Random(int rows, int cols)
        {
            return DM::rand(rows, cols);
        }

        // LeCun initialization
        // ~N(0, 1 / n_in)
        DM LeCun(int rows, int cols)
        {
            return randn_DM(rows, cols, 0.0, 1.0 / cols);
        }

        // He initialization
        // ~N(0, sqrt(2 / n_in))
        DM He(int rows, int cols)
        {
            return randn_DM(rows, cols, 0.0, sqrt(2.0 / cols));
        }

        // Xavier initialization
        // ~U(-1 / sqrt(n_in), 1 / sqrt(n_in))
        DM Xavier(int rows, int cols)
        {
            double lower = -sqrt(1 / cols);
            double upper = -lower;
            return DM::rand(rows, cols) * (upper - lower) + lower;
        }

        // Normalized Xavier initialization
        // ~U(-sqrt(6) / sqrt(n_in + n_out), sqrt(6) / sqrt(n_in + n_out))
        DM NXavier(int rows, int cols)
        {
            double lower = -sqrt(6.0 / (cols + rows));
            double upper = -lower;
            return DM::rand(rows, cols) * (upper - lower) + lower;
        }
    }

    enum OPTIMIZER
    {
        GD,
        ADAM,
        IPOPT
    };

    struct Params
    {
        int input_size;
        int output_size;
        std::vector<int> hidden_layers;
        std::function<MX(const MX &)> activation = activation::Sigmoid;
        std::function<DM(int, int)> initializer = initializers::NXavier;
        OPTIMIZER optimizer = GD;
        // Gradient-based only parameters
        int epochs = 1000;
        double learning_rate = 0.01;
        double momentum = 0;
        double max_grad = -1;
        // ADAM parameters
        double beta1 = 0.9;
        double beta2 = 0.999;
        double epsilon = 1e-8;
        // Dropout parameters
        double dropout_rate = 0.5;
        int inference_passes = 10;
    };

    // Fully connected NN
    class SymNN
    {
    public:
        SymNN(const Params &params) : _params(params)
        {
            _construct(params.hidden_layers);
        }

        SymNN(const std::string &filename, Params &params) : _params(params)
        {
            std::ifstream file(filename);

            std::string activation_name;
            file >> activation_name;

            _params.activation = activation::name_to_activation.find(activation_name) == activation::name_to_activation.end() ? activation::Sigmoid : activation::name_to_activation[activation_name];
            file >> _params.input_size;
            file >> _params.output_size;

            _params.hidden_layers.clear();

            std::string line;
            std::getline(file, line);
            std::getline(file, line);
            std::istringstream iss(line);

            int num;
            while (iss >> num)
            {
                _params.hidden_layers.push_back(num);
            }

            int prev_size = _params.input_size;
            double val;

            _params.hidden_layers.push_back(_params.output_size);

            for (int i = 0; i < _params.hidden_layers.size(); ++i)
            {
                _nn_values.push_back(DM(params.hidden_layers[i], prev_size));

                for (int j = 0; j < _params.hidden_layers[i]; ++j)
                {
                    for (int k = 0; k < prev_size; ++k)
                    {
                        file >> val;
                        _nn_values.back()(j, k) = val;
                    }
                }

                _nn_values.push_back(DM(params.hidden_layers[i], 1));

                for (int j = 0; j < _params.hidden_layers[i]; ++j)
                {
                    file >> val;
                    _nn_values.back()(j) = val;
                }

                prev_size = _params.hidden_layers[i];
            }

            _params.hidden_layers.pop_back();

            file.close();

            _construct(params.hidden_layers);
        }

        Eigen::VectorXd forward(const Eigen::VectorXd &input)
        {
            DM X = DM(input.size(), 1);
            for (int i = 0; i < input.size(); ++i)
            {
                X(i) = input(i);
            }

            DM out = 0;

            for (int i = 0; i < _params.inference_passes; ++i)
            {
                std::vector<DM> params(_nn_values.begin(), _nn_values.end());
                params.insert(params.begin(), X);
                std::vector<DM> mask = _get_random_mask();
                params.insert(params.end(), mask.begin(), mask.end());

                out += _out_fn(params)[0];
            }

            out /= _params.inference_passes;

            Eigen::VectorXd output(_params.output_size);

            for (int i = 0; i < _params.output_size; ++i)
            {
                output(i) = static_cast<double>(out(i));
            }

            return output;
        }

        std::pair<MX, MX> forward(MX &input)
        {
            std::vector<MX> ls;
            MX l = 0;

            for (int i = 0; i < _params.inference_passes; ++i)
            {
                std::vector<DM> mask = _get_random_mask();
                MX out = substitute(_out_substituted, _r[0], mask[0]);
                for (int i = 1; i < _r.size(); ++i)
                {
                    out = substitute(out, _r[i], mask[i]);
                }

                ls.push_back(substitute(out, _X, input));
                l += ls.back();
            }

            l /= _params.inference_passes;

            MX var = 0;

            for (int i = 0; i < _params.inference_passes; ++i)
            {
                var += pow(ls[i] - l, 2);
            }

            var /= _params.inference_passes;

            return {l, var};
        }

        void train(const Eigen::MatrixXd &input, const Eigen::MatrixXd &target, int stop_col = -1)
        {
            switch (_params.optimizer)
            {
            case GD:
                _train_gd(input, target, stop_col);
                break;
            case ADAM:
                _train_adam(input, target, stop_col);
                break;
            case IPOPT:
                _train_ipopt(input, target, stop_col);
                break;
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
                    _nn_values[i] = _params.initializer(_nn_values[i].size1(), _nn_values[i].size2());
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

            file << activation::activation_to_name[_params.activation] << std::endl;
            file << _params.input_size << std::endl;
            file << _params.output_size << std::endl;

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
        Params _params;

        int _total_size;
        MX _X, _Y;
        std::vector<MX> _W, _b;
        std::vector<MX> _r;
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
            _X = MX::sym("X", _params.input_size);
            _Y = MX::sym("Y", _params.output_size);

            std::vector<MX> all_params;
            all_params.push_back(_X);

            std::vector<MX> flat_params;

            MX prev = _X;
            MX r;

            for (int i = 0; i < hidden_layers.size(); ++i)
            {
                _W.push_back(MX::sym("W" + std::to_string(i), hidden_layers[i], prev.size1()));
                _b.push_back(MX::sym("b" + std::to_string(i), hidden_layers[i]));
                r = MX::sym("r" + std::to_string(i), hidden_layers[i]);
                _r.push_back(r);

                prev = r * _params.activation(mtimes(_W[i], prev) + _b[i]);

                all_params.push_back(_W[i]);
                all_params.push_back(_b[i]);

                flat_params.push_back(reshape(_W[i], -1, 1));
                flat_params.push_back(_b[i]);
            }

            _W.push_back(MX::sym("Wout", _params.output_size, hidden_layers.back()));
            _b.push_back(MX::sym("bout", _params.output_size));

            all_params.push_back(_W.back());
            all_params.push_back(_b.back());

            flat_params.push_back(reshape(_W.back(), -1, 1));
            flat_params.push_back(_b.back());

            MX flat_params_var = vertcat(flat_params);

            _total_size = flat_params_var.size1();

            all_params.insert(all_params.end(), _r.begin(), _r.end());

            _out = mtimes(_W.back(), prev) + _b.back();
            _out_fn = Function("out", all_params, {_out});

            _loss = sumsqr(_out - _Y);

            _gradients = gradient(_loss, flat_params_var);

            _gradient_fn = Function("gradient_fn", {flat_params_var, _X, _Y, vertcat(_r)}, {_gradients});

            std::vector<MX> opt_vars(all_params.size() - 1);
            for (int i = 0; i < _W.size(); ++i)
            {
                opt_vars[2 * i] = reshape(_W[i], -1, 1);
                opt_vars[2 * i + 1] = _b[i];
            }

            _opt_vars = vertcat(opt_vars);

            if (_nn_values.empty())
            {
                for (int i = 1; i < all_params.size() - 2 - _r.size(); ++i)
                {
                    if (i % 2)
                    {
                        _nn_values.push_back(_params.initializer(all_params[i].size1(), all_params[i].size2()));
                    }
                    else
                    {
                        _nn_values.push_back(DM::rand(all_params[i].size1(), all_params[i].size2()));
                    }
                }

                _nn_values.push_back(DM::rand(all_params[all_params.size() - 2 - _r.size()].size1(), all_params[all_params.size() - 2 - _r.size()].size2()));
                _nn_values.push_back(DM::rand(all_params[all_params.size() - 1 - _r.size()].size1(), all_params[all_params.size() - 1 - _r.size()].size2()));
            }

            _out_substituted = _out;
            for (int i = 0; i < _W.size(); ++i)
            {
                _out_substituted = substitute(_out_substituted, _W[i], _nn_values[2 * i]);
                _out_substituted = substitute(_out_substituted, _b[i], _nn_values[2 * i + 1]);
            }
        }

        std::vector<DM> _get_random_mask()
        {
            static std::random_device rd;
            static std::mt19937 gen(rd());
            std::bernoulli_distribution d(1.0 - _params.dropout_rate);

            std::vector<DM> mask;
            for (int i = 0; i < _r.size(); ++i)
            {
                DM layer_mask(_b[i].size(1));

                for (int j = 0; j < layer_mask.size1(); ++j)
                {
                    layer_mask(j) = d(gen);
                }

                mask.push_back(layer_mask);
            }

            return mask;
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

        void _train_adam(const Eigen::MatrixXd &input, const Eigen::MatrixXd &target, int stop_col = -1)
        {
            DM current_params;
            for (int i = 0; i < _nn_values.size(); i += 2)
            {
                current_params = vertcat(current_params, reshape(_nn_values[i], -1, 1));
                current_params = vertcat(current_params, _nn_values[i + 1]);
            }

            DM X = DM(input.rows(), (stop_col == -1 ? input.cols() : stop_col));
            for (int i = 0; i < input.rows(); ++i)
            {
                for (int j = 0; j < (stop_col == -1 ? input.cols() : stop_col); ++j)
                {
                    X(i, j) = input(i, j);
                }
            }

            DM Y = DM(target.rows(), (stop_col == -1 ? target.cols() : stop_col));
            for (int i = 0; i < target.rows(); ++i)
            {
                for (int j = 0; j < (stop_col == -1 ? input.cols() : stop_col); ++j)
                {
                    Y(i, j) = target(i, j);
                }
            }

            DM m = DM::zeros(current_params.size1(), current_params.size2());
            DM v = DM::zeros(current_params.size1(), current_params.size2());
            int t = 1;

            for (int epoch = 0; epoch < _params.epochs; ++epoch)
            {
                for (int j = 0; j < (stop_col == -1 ? input.cols() : stop_col); ++j)
                {
                    std::vector<DM> params = {current_params, X(Slice(), j), Y(Slice(), j), vertcat(_get_random_mask())};

                    DM grad_values = _gradient_fn(params)[0];

                    m = (_params.beta1 * m + (1 - _params.beta1) * grad_values);
                    v = (_params.beta2 * v + (1 - _params.beta2) * pow(grad_values, 2));

                    current_params -= _params.learning_rate * (m / (1 - pow(_params.beta1, t))) / (sqrt((v / (1 - pow(_params.beta2, t++)))) + _params.epsilon);
                }
            }

            int offset = 0;
            for (int i = 0; i < _nn_values.size(); ++i)
            {
                int param_size = _nn_values[i].size1() * _nn_values[i].size2();
                _nn_values[i] = reshape(current_params(Slice(offset, offset + param_size), 0), _nn_values[i].size1(), _nn_values[i].size2());
                offset += param_size;
            }
        }

        void _train_gd(const Eigen::MatrixXd &input, const Eigen::MatrixXd &target, int stop_col = -1)
        {
            /*
            DM current_params;
            // std::cout << "First value: " << _nn_values[0] << std::endl;
            for (int i = 0; i < _nn_values.size(); i += 2)
            {
                current_params = vertcat(current_params, reshape(_nn_values[i], -1, 1));
                current_params = vertcat(current_params, _nn_values[i + 1]);
            }
            // std::cout << current_params << std::endl;

            DM X = DM(input.rows(), (stop_col == -1 ? input.cols() : stop_col));
            for (int i = 0; i < input.rows(); ++i)
            {
                for (int j = 0; j < (stop_col == -1 ? input.cols() : stop_col); ++j)
                {
                    X(i, j) = input(i, j);
                }
            }

            DM Y = DM(target.rows(), (stop_col == -1 ? target.cols() : stop_col));
            for (int i = 0; i < target.rows(); ++i)
            {
                for (int j = 0; j < (stop_col == -1 ? input.cols() : stop_col); ++j)
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

            // std::cout << "First value now: " << current_params(0) << std::endl;*/
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