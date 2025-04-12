#ifndef PQ_SYMNN_HPP
#define PQ_SYMNN_HPP

#include <chrono>
#include <functional>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

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
    std::default_random_engine       gen(time(NULL));
    std::normal_distribution<double> dist(mean, std);
    DM                               r(rows, cols);

    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            r(i, j) = dist(gen);
        }
    }

    return r;
}

/** eigen_to_DM function
 * Convert an Eigen matrix to a CasADi DM
 */
DM eigen_to_DM(const Eigen::MatrixXd &M)
{
    DM D(M.rows(), M.cols());

    for (int i = 0; i < M.rows(); ++i)
    {
        for (int j = 0; j < M.cols(); ++j)
        {
            D(i, j) = M(i, j);
        }
    }

    return D;
}

/** DM_to_eigen function
 * Convert a CasADi DM to an Eigen matrix
 */
Eigen::MatrixXd DM_to_eigen(const DM &D)
{
    Eigen::MatrixXd M(D.size1(), D.size2());

    for (int i = 0; i < D.size1(); ++i)
    {
        for (int j = 0; j < D.size2(); ++j)
        {
            M(i, j) = static_cast<double>(D(i, j));
        }
    }

    return M;
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

MX Elu(const MX &x)
{
    return if_else(x > 0, x, 0.01 * (exp(x) - 1));
}

MX Tanh(const MX &x)
{
    return tanh(x);
}

MX Gaussian(const MX &x)
{
    return exp(-pow(x, 2));
}

enum ACTIVATION
{
    SIGMOID  = 0,
    SOFTMAX  = 1,
    RELU     = 2,
    LRELU    = 3,
    ELU      = 4,
    TANH     = 5,
    GAUSSIAN = 6
};

std::unordered_map<ACTIVATION, std::function<MX(const MX &)>>
  activation_to_func = {
      {  SIGMOID,  Sigmoid },
      {  SOFTMAX,  Softmax },
      {     RELU,     Relu },
      {    LRELU,    Lrelu },
      {      ELU,      Elu },
      {     TANH,     Tanh },
      { GAUSSIAN, Gaussian }
};
}

namespace initializers
{
/* Initializer inputs:
 * rows: number of rows in the matrix
 *  also represents the number of neurons in the current layer
 * cols: number of columns in the matrix
 *  also represents the number of neurons in the previous layer (input size)
 *
 * Note: All matrices are returned as flat vectors
 */

// Zero initialization
DM Zero(int rows, int cols)
{
    return DM::zeros(rows * cols, 1);
}

// Uniform random initialization
// ~U(0, 1)
DM Random(int rows, int cols)
{
    return DM::rand(rows * cols, 1);
}

// LeCun initialization
// ~N(0, 1 / n_in)
DM LeCun(int rows, int cols)
{
    return randn_DM(rows * cols, 1, 0.0, 1.0 / cols);
}

// He initialization
// ~N(0, sqrt(2 / n_in))
DM He(int rows, int cols)
{
    return randn_DM(rows * cols, 1, 0.0, sqrt(2.0 / cols));
}

// Xavier initialization
// ~U(-1 / sqrt(n_in), 1 / sqrt(n_in))
DM Xavier(int rows, int cols)
{
    double lower = -sqrt(1 / cols);
    double upper = -lower;
    return DM::rand(rows * cols, 1) * (upper - lower) + lower;
}

// Normalized Xavier initialization
// ~U(-sqrt(6) / sqrt(n_in + n_out), sqrt(6) / sqrt(n_in + n_out))
DM NXavier(int rows, int cols)
{
    double lower = -sqrt(6.0 / (cols + rows));
    double upper = -lower;
    return DM::rand(rows * cols, 1) * (upper - lower) + lower;
}
}

enum OPTIMIZER
{
    GD,
    ADAM
};

struct Params
{
    int                         input_size;
    int                         output_size;
    std::vector<int>            hidden_layers;
    activation::ACTIVATION      activation  = activation::SIGMOID;
    std::function<DM(int, int)> initializer = initializers::NXavier;
    OPTIMIZER                   optimizer   = GD;
    // Gradient-based only parameters
    int    epochs        = 1000;
    double learning_rate = 0.01;
    double momentum      = 0;
    double max_grad      = -1;
    // ADAM parameters
    double beta1   = 0.9;
    double beta2   = 0.999;
    double epsilon = 1e-8;
    // Dropout parameters
    double dropout_rate     = 0;
    int    inference_passes = 1;
};

// Fully connected NN
class SymNN
{
  public:
    /** Initialize new neural network from parameters
     */
    SymNN(const Params &params) : _params(params)
    {
        _construct();
    }

    /** Initialize new neural network from file
     */
    SymNN(const std::string &filename, Params &params) : _params(params)
    {
        std::ifstream file(filename);

        // read architecture
        int num;
        file >> num;
        _params.activation =
          static_cast<activation::ACTIVATION>(_params.activation);

        file >> _params.input_size;
        file >> _params.output_size;

        _params.hidden_layers.clear();

        std::string line;
        std::getline(file, line);
        std::getline(file, line);
        std::istringstream iss(line);

        int total_params = 0;
        int prev_size    = _params.input_size;

        while (iss >> num)
        {
            _params.hidden_layers.push_back(num);
            total_params += num * (prev_size + 1);
            prev_size     = num;
        }

        total_params += _params.output_size * (prev_size + 1);

        _nn_values = DM(total_params, 1);

        int offset = 0;
        prev_size  = _params.input_size;
        double val;

        _params.hidden_layers.push_back(_params.output_size);

        // read parameter values
        for (int i = 0; i < _params.hidden_layers.size(); ++i)
        {
            // read weight matrix values
            for (int j = 0; j < _params.hidden_layers[i]; ++j)
            {
                for (int k = 0; k < prev_size; ++k)
                {
                    file >> val;
                    _nn_values(offset + j * prev_size + k) = val;
                }
            }

            offset += _params.hidden_layers[i] * prev_size;

            // read threshold vector values
            for (int j = 0; j < _params.hidden_layers[i]; ++j)
            {
                file >> val;
                _nn_values(offset + j) = val;
            }

            offset    += _params.hidden_layers[i];
            prev_size  = _params.hidden_layers[i];
        }

        _params.hidden_layers.pop_back();

        file.close();

        _construct();
    }

    Eigen::VectorXd forward(const Eigen::VectorXd &input)
    {
        DM X   = eigen_to_DM(input);
        DM out = 0;

        for (int i = 0; i < _params.inference_passes; ++i)
        {
            std::vector<DM> inputs  = { _nn_values, _get_random_mask(), X };
            out                    += _out_fn(inputs)[0];
        }

        out /= _params.inference_passes;

        return DM_to_eigen(out);
    }

    std::pair<MX, MX> forward(MX &input)
    {
        std::vector<MX> ls;
        MX              l = 0;

        for (int i = 0; i < _params.inference_passes; ++i)
        {
            ls.push_back(_out_fn({ _nn_values, _get_random_mask(), input })[0]);
            l += ls.back();
        }

        l /= _params.inference_passes;

        MX var = 0;

        for (int i = 0; i < _params.inference_passes; ++i)
        {
            var += pow(ls[i] - l, 2);
        }

        var /= _params.inference_passes;

        return { l, var };
    }

    void train(const Eigen::MatrixXd &input, const Eigen::MatrixXd &target,
               int stop_col = -1)
    {
        DM X = eigen_to_DM(
          input.block(0,
                      0,
                      input.rows(),
                      stop_col == -1 ? input.cols() : stop_col + 1));

        DM Y = eigen_to_DM(
          target.block(0,
                       0,
                       target.rows(),
                       stop_col == -1 ? target.cols() : stop_col + 1));

        switch (_params.optimizer)
        {
        case GD:
            _train_gd(X, Y);
            break;
        case ADAM:
            _train_adam(X, Y);
            break;
        }
    }

    void reset()
    {
        int prev_size     = _params.input_size;
        int params_offset = 0;

        _params.hidden_layers.push_back(_params.output_size);
        for (int i = 0; i < _params.hidden_layers.size(); ++i)
        {
            int layer_weights = _params.hidden_layers[i] * prev_size;
            // initialize weight matrix using initializer function
            _nn_values(Slice(params_offset, params_offset + layer_weights)) =
              _params.initializer(_params.hidden_layers[i], prev_size);

            // initialize thresholds with random values
            _nn_values(
              Slice(params_offset + layer_weights,
                    params_offset + layer_weights + _params.hidden_layers[i])) =
              DM::zeros(_params.hidden_layers[i], 1);

            params_offset += layer_weights + _params.hidden_layers[i];
            prev_size      = _params.hidden_layers[i];
        }
        _params.hidden_layers.pop_back();
    }

    void save(const std::string &filename)
    {
        std::ofstream file(filename);

        file << _params.activation << std::endl;
        file << _params.input_size << std::endl;
        file << _params.output_size << std::endl;

        for (int i = 0; i < _params.hidden_layers.size(); ++i)
        {
            file << _params.hidden_layers[i];
            if (i < _params.hidden_layers.size())
            {
                file << " ";
            }
        }
        file << std::endl;

        int params_offset = 0;
        int prev_size     = _params.input_size;

        _params.hidden_layers.push_back(_params.output_size);

        for (int l = 0; l < _params.hidden_layers.size(); ++l)
        {
            for (int i = 0; i < _params.hidden_layers[l]; ++i)
            {
                for (int j = 0; j < prev_size; ++j)
                {
                    file << _nn_values(params_offset + i * prev_size + j);
                    if (j != prev_size - 1)
                    {
                        file << " ";
                    }
                }
                file << std::endl;
            }

            params_offset += _params.hidden_layers[l] * prev_size;

            for (int i = 0; i < _params.hidden_layers[l]; ++i)
            {
                file << _nn_values(params_offset + i) << std::endl;
            }

            params_offset += _params.hidden_layers[l];
            prev_size      = _params.hidden_layers[l];
        }

        _params.hidden_layers.pop_back();

        file.close();
    }

  protected:
    Params      _params;
    std::string _activation_name;

    Function _out_fn;
    Function _gradient_fn;

    DM  _nn_values;
    int _mask_size;

    void _construct()
    {
        MX X = MX::sym("X", _params.input_size);
        MX Y = MX::sym("Y", _params.output_size);

        int prev_size        = X.size1();
        int total_weights    = 0;
        int total_thresholds = 0;

        for (int i = 0; i < _params.hidden_layers.size(); ++i)
        {
            total_weights    += _params.hidden_layers[i] * prev_size;
            total_thresholds += _params.hidden_layers[i];
            prev_size         = _params.hidden_layers[i];
        }

        _mask_size = total_thresholds;

        total_weights    += _params.output_size * prev_size;
        total_thresholds += _params.output_size;

        MX nn_params = MX::sym("theta", total_weights + total_thresholds);
        MX mask      = MX::sym("r", _mask_size);

        // connect parameters according to NN equations
        MX  prev          = X;
        int params_offset = 0;
        int mask_offset   = 0;
        for (int i = 0; i < _params.hidden_layers.size(); ++i)
        {
            int layer_weights = _params.hidden_layers[i] * prev.size1();

            prev =
              mask(Slice(mask_offset, mask_offset + _params.hidden_layers[i])) *
              activation::activation_to_func[_params.activation](
                mtimes(reshape(nn_params(Slice(params_offset,
                                               params_offset + layer_weights)),
                               _params.hidden_layers[i],
                               prev.size1()),
                       prev) +
                nn_params(Slice(params_offset + layer_weights,
                                params_offset + layer_weights +
                                  _params.hidden_layers[i])));

            params_offset += layer_weights + _params.hidden_layers[i];
            mask_offset   += _params.hidden_layers[i];
        }

        int layer_weights = _params.output_size * prev.size1();
        MX  out =
          mtimes(reshape(nn_params(
                           Slice(params_offset, params_offset + layer_weights)),
                         _params.output_size,
                         prev.size1()),
                 prev) +
          nn_params(Slice(params_offset + layer_weights, nn_params.size1()));
        _out_fn = Function("out", { nn_params, mask, X }, { out });

        MX loss      = sumsqr(out - Y);
        MX gradients = gradient(loss, nn_params);

        _gradient_fn =
          Function("gradient_fn", { nn_params, mask, X, Y }, { gradients });

        if (!_nn_values.size1())
        {
            _nn_values = DM(nn_params.size1(), nn_params.size2());
            reset();
        }
    }

    DM _get_random_mask()
    {
        static std::random_device   rd;
        static std::mt19937         gen(rd());
        std::bernoulli_distribution d(1.0 - _params.dropout_rate);

        DM mask(_mask_size, 1);
        for (int i = 0; i < _mask_size; ++i)
        {
            mask(i) = d(gen);
        }

        return mask;
    }

    void _train_adam(const DM &X, const DM &Y)
    {
        DM  m = DM::zeros(_nn_values.size1(), _nn_values.size2());
        DM  v = DM::zeros(_nn_values.size1(), _nn_values.size2());
        int t = 1;

        for (int epoch = 0; epoch < _params.epochs; ++epoch)
        {
            for (int j = 0; j < X.size2(); ++j)
            {
                std::vector<DM> inputs = { _nn_values,
                                           _get_random_mask(),
                                           X(Slice(), j),
                                           Y(Slice(), j) };

                DM grad_values = _gradient_fn(inputs)[0];

                m = (_params.beta1 * m + (1 - _params.beta1) * grad_values);
                v = (_params.beta2 * v +
                     (1 - _params.beta2) * pow(grad_values, 2));

                _nn_values -=
                  _params.learning_rate * (m / (1 - pow(_params.beta1, t))) /
                  (sqrt((v / (1 - pow(_params.beta2, t++)))) + _params.epsilon);
            }
        }
    }

    void _train_gd(const DM &X, const DM &Y)
    {
        for (int epoch = 0; epoch < _params.epochs; ++epoch)
        {
            DM prev_update = DM::zeros(_nn_values.size1(), _nn_values.size2());
            for (int j = 0; j < X.size2(); ++j)
            {
                std::vector<DM> inputs = { _nn_values,
                                           _get_random_mask(),
                                           X(Slice(), j),
                                           Y(Slice(), j) };

                DM grad_values = _gradient_fn(inputs)[0];

                DM update = _params.learning_rate * grad_values;
                if (_params.max_grad > 0)
                {
                    update = update / norm_2(update) * _params.max_grad;
                }
                _nn_values  -= update + _params.momentum * prev_update;
                prev_update  = update;
            }
        }
    }
};
}

#endif