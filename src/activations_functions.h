#include <Eigen/Core>
#include <Eigen/Dense>
#include <vector>

namespace NeuralNetwork::ActivationsFunctions {
using Vector = Eigen::VectorXd;
using Matrix = Eigen::MatrixXd;

class BaseActivationFunction {
 public:
  virtual Vector compute(const Vector& x) = 0;
  virtual Matrix getDerivative(const Vector& x) = 0;
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
};

class Relu : public BaseActivationFunction {
 public:
  Vector compute(const Vector& x) final { return x.cwiseMax(0.0); }
  Matrix getDerivative(const Vector& x) final {
    return (x.array() > 0.0).cast<double>();
  }
};
}  // namespace NeuralNetwork::ActivationsFunctions
