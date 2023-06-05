#include "activations_functions.h"

namespace NN {

Vector Sigmoid::compute(const Vector& x) {
  return 1.0 / (1.0 + exp(-x.array()));
}

Matrix Sigmoid::getDerivative(const Vector& x) {
  return (exp(-x.array()) / pow(1.0 + exp(-x.array()), 2))
      .matrix()
      .asDiagonal();
}

ActivationFunctionType Sigmoid::getName() {
  return ActivationFunctionType::Sigmoid;
}

Vector Relu::compute(const Vector& x) {
  return x.cwiseMax(0.0);
}

Matrix Relu::getDerivative(const Vector& x) {
  return (x.array() > 0.0).cast<double>().matrix().asDiagonal();
}

ActivationFunctionType Relu::getName() {
  return ActivationFunctionType::Relu;
}

Vector Softmax::compute(const Vector& x) {
  auto result = x.array().exp();
  return result / result.sum();
}

Matrix Softmax::getDerivative(const Vector& x) {
  Vector computeSoftmax = compute(x);
  Matrix diagonal = computeSoftmax.asDiagonal();
  return diagonal - computeSoftmax * computeSoftmax.transpose();
}

ActivationFunctionType Softmax::getName() {
  return ActivationFunctionType::Softmax;
}

std::unique_ptr<BaseActivationFunction> getActivationFunctionByType(
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
}  // namespace NN
