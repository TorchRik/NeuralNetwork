#include <Eigen/Core>
#include <Eigen/Dense>
#include <vector>
#include "activations_functions.h"

namespace NeuralNetwork {
using Vector = Eigen::VectorXd;
using Matrix = Eigen::MatrixXd;

struct LayerDimension {
  ssize_t n;
  ssize_t m;

  bool operator==(const LayerDimension& other) {
    return other.n == n && other.m == m;
  }
};

struct LayerDelta {
  Matrix deltaA;
  Vector deltaB;

  const LayerDelta& operator-=(const LayerDelta&& other) {
    deltaA -= other.deltaA;
    deltaB -= other.deltaB;
    return *this;
  }
};

class Layer {
  using ActivationFunction =
      std::unique_ptr<ActivationsFunctions::BaseActivationFunction>;

 public:
  Layer(LayerDimension dimension, ActivationFunction function) {
    assert(dimension.n > 0 && dimension.m > 0);
    A_ = Matrix::Random(dimension.n, dimension.m);
    b_ = Vector::Random(dimension.n);
    nonLinearFunction_ = std::move(function);
  }

  Vector compute(const Vector& x) const {
    assert(x.rows() == A_.cols());
    return nonLinearFunction_->compute(A_ * x + b_);
  }

  Matrix getDerivativeA(const Vector& x, const Vector& u) const {
    assert(x.rows() == A_.cols());
    assert(u.rows() == b_.rows());
    return nonLinearFunction_->getDerivative(A_ * x + b_) * u * x.transpose();
  }

  Vector getDerivativeB(const Vector& x, const Vector& u) const {
    assert(x.rows() == A_.cols());
    assert(u.rows() == b_.rows());
    return nonLinearFunction_->getDerivative(A_ * x + b_) * u;
  }

  LayerDelta getDerivative(const Vector& x, const Vector& u) const {
    return {getDerivativeA(x, u), getDerivativeB(x, u)};
  }

  Vector getNextU(const Vector& x, Vector& u) const {
    assert(x.rows() == A_.cols());
    assert(u.rows() == b_.rows());
    return (u.transpose() * nonLinearFunction_->getDerivative(A_ * x + b_) * A_)
        .transpose();
  }

  std::pair<Matrix, Vector> getWeights() const { return {A_, b_}; }

  LayerDimension getLayerDimension() const { return {A_.rows(), A_.cols()}; }

  const Layer& operator+=(const LayerDelta& other) & {
    assert(A_.cols() == other.deltaA.cols() &&
           A_.rows() == other.deltaA.rows());
    assert(b_.rows() == other.deltaB.rows());
    A_ += other.deltaA;
    b_ += other.deltaB;
    return *this;
  }

 private:
  Matrix A_;
  Vector b_;
  ActivationFunction nonLinearFunction_;
};
}  // namespace NeuralNetwork
