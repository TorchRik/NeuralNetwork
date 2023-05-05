#include <Eigen/Core>
#include <Eigen/Dense>
#include <vector>
#include <iostream>
#include "activations_functions.h"

namespace NeuralNetwork {
using Vector = Eigen::VectorXd;
using Matrix = Eigen::MatrixXd;

struct LayerDimension {
  ssize_t n;
  ssize_t m;
};

struct LayerDelta {
  Matrix deltaA_;
  Vector deltaB_;
  const LayerDelta &operator-=(const LayerDelta &&other) {
    deltaA_ -= other.deltaA_;
    deltaB_ -= other.deltaB_;
    return *this;
  }
};

class Layer {
  using ActivationFunction = std::unique_ptr<ActivationsFunctions::BaseActivationFunction>;
 public:

  Layer(LayerDimension dimension, ActivationFunction function) {
    assert(dimension.n > 0 && dimension.m > 0);
    A_ = Matrix::Random(dimension.n, dimension.m);
    b_ = Vector::Random(dimension.n);
    nonLinearFunction_ = std::move(function);
  }

  Vector compute(const Vector &x) const {
    return nonLinearFunction_->compute(A_ * x + b_);
  }
  Matrix getDerivativeA(const Vector &x, const Vector &u) const {
    return nonLinearFunction_->getDerivative(A_ * x + b_) * u * z.transpose();
  }
  Vector getDerivativeB(const Vector &x, const Vector &u) const {
    return nonLinearFunction_->getDerivative(A_ * x + b_) * u;
  }

  LayerDelta getDerivative(const Vector &x, const Vector &u) {
    return {getDerivativeA(x, u), getDerivativeB(x, u)};
  }

  Vector getNextU(const Vector &x, Vector &u) const {
    return (u.transpose() * nonLinearFunction_->getDerivative(A_ * x + b_) * A_).transpose();
  }

  std::pair<Matrix, Vector> getWeights() {
    return {A_, b_};
  }

  LayerDimension getLayerDimension() {
    return {A_.rows(), A_.cols()};
  }

  const Layer &operator+=(const LayerDelta &other) &{
    A_ += other.deltaA_;
    b_ += other.deltaB_;
    return *this;
  }

 private:
  Matrix A_;
  Vector b_;
  ActivationFunction nonLinearFunction_;
};
}
