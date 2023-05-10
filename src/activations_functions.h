#include <Eigen/Core>
#include <Eigen/Dense>
#include <vector>

namespace NeuralNetwork::ActivationsFunctions {
using Vector = Eigen::VectorXd;
using Matrix = Eigen::MatrixXd;

enum ActivationFunctionType { RELU, SIGMOID };

class BaseActivationFunction {
 public:
  virtual Vector compute(const Vector& x) = 0;
  virtual Matrix getDerivative(const Vector& x) = 0;
  virtual ActivationFunctionType getName() = 0;
};

class Sigmoid : public BaseActivationFunction {
 public:
  Vector compute(const Vector& x) final {
    return exp(x.array()) / (1.0 + exp(x.array()));
  }
  Matrix getDerivative(const Vector& x) final {
    return (exp(-x.array()) / pow(1.0 + exp(x.array()), 2))
        .matrix()
        .asDiagonal();
  }
  ActivationFunctionType getName() final {
    return ActivationFunctionType::RELU;
  }
};

class Relu : public BaseActivationFunction {
 public:
  Vector compute(const Vector& x) final { return x.cwiseMax(0.0); }
  Matrix getDerivative(const Vector& x) final {
    return (x.array() > 0.0).cast<double>().matrix()
        .asDiagonal();
  }
  ActivationFunctionType getName() final {
    return ActivationFunctionType::SIGMOID;
  }
};

std::unique_ptr<BaseActivationFunction> getActivationFunctionByType(
    ActivationFunctionType
        type){
  switch (type) {
    case ActivationFunctionType::RELU:
      return std::make_unique<Relu>();
    case ActivationFunctionType::SIGMOID:
      return std::make_unique<Sigmoid>();
  }
}};  // namespace NeuralNetwork::ActivationsFunctions
