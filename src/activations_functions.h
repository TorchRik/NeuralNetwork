#include <Eigen/Core>
#include <Eigen/Dense>
#include <vector>

namespace NeuralNetwork::ActivationsFunctions {
using Vector = Eigen::VectorXd;
using Matrix = Eigen::MatrixXd;

enum class ActivationFunctionType { Relu, Sigmoid, Softmax };

class BaseActivationFunction {
 public:
  virtual Vector compute(const Vector& x) = 0;
  virtual Matrix getDerivative(const Vector& x) = 0;
  virtual ActivationFunctionType getName() = 0;
  virtual ~BaseActivationFunction() = default;
};

class Sigmoid : public BaseActivationFunction {
 public:
  Vector compute(const Vector& x) final {
    return 1.0 / (1.0 + exp(-x.array()));
  }
  Matrix getDerivative(const Vector& x) final {
    return (exp(-x.array()) / pow(1.0 + exp(-x.array()), 2))
        .matrix()
        .asDiagonal();
  }
  ActivationFunctionType getName() final {
    return ActivationFunctionType::Sigmoid;
  }
};

class Relu : public BaseActivationFunction {
 public:
  Vector compute(const Vector& x) final { return x.cwiseMax(0.0); }
  Matrix getDerivative(const Vector& x) final {
    return (x.array() > 0.0).cast<double>().matrix().asDiagonal();
  }
  ActivationFunctionType getName() final {
    return ActivationFunctionType::Relu;
  }
};

class Softmax : public BaseActivationFunction {
 public:
  Vector compute(const Vector& x) final {
    auto result = x.array().exp();
    return result / result.sum();
  }
  Matrix getDerivative(const Vector& x) final {
    Vector computeSoftmax = compute(x);
    Matrix diagonal = computeSoftmax.asDiagonal();
    return diagonal - computeSoftmax * computeSoftmax.transpose();
  }
  ActivationFunctionType getName() final {
    return ActivationFunctionType::Softmax;
  }
};

inline std::unique_ptr<BaseActivationFunction> getActivationFunctionByType(
    ActivationFunctionType type) {
  switch (type) {
    case ActivationFunctionType::Relu:
      return std::make_unique<Relu>();
    case ActivationFunctionType::Sigmoid:
      return std::make_unique<Sigmoid>();
    case ActivationFunctionType::Softmax:
      return std::make_unique<Softmax>();
    default:
      throw std::runtime_error("Incorrect type provided");
  }
}
}  // namespace NeuralNetwork::ActivationsFunctions
