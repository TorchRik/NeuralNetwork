#include <Eigen/Core>
#include <Eigen/Dense>
#include <utility>
#include <vector>
#include "activations_functions.h"

namespace NeuralNetwork {
using Vector = Eigen::VectorXd;
using Matrix = Eigen::MatrixXd;

struct LayerDimension {
  ssize_t n;
  ssize_t m;
};

struct LayerDelta {
  Matrix deltaA;
  Vector deltaB;

  const LayerDelta& operator+=(const LayerDelta&& other) {
    deltaA += other.deltaA;
    deltaB += other.deltaB;
    return *this;
  }
  const LayerDelta& operator-=(const LayerDelta&& other) {
    deltaA -= other.deltaA;
    deltaB -= other.deltaB;
    return *this;
  }
  const LayerDelta& operator/=(double num) {
    deltaA /= num;
    deltaB /= num;
    return *this;
  }
  const LayerDelta& operator*=(double num) {
    deltaA *= num;
    deltaB *= num;
    return *this;
  }
  LayerDelta operator/(double num) {
    auto layerDelta = *this;
    layerDelta.deltaA /= num;
    layerDelta.deltaB /= num;
    return layerDelta;
  }
  LayerDelta operator*(double num) {
    auto layerDelta = *this;
    layerDelta.deltaA *= num;
    layerDelta.deltaB *= num;
    return layerDelta;
  }
};

class Layer {
  using ActivationFunctionPtr =
      std::unique_ptr<ActivationsFunctions::BaseActivationFunction>;
  using ActivationFunctionType = ActivationsFunctions::ActivationFunctionType;

 public:
  Layer(LayerDimension dimension, ActivationFunctionType functionType) {
    assert(dimension.n > 0 && dimension.m > 0);
    A_ = Matrix::Random(dimension.n, dimension.m);
    b_ = Vector::Random(dimension.n);
    activationFunction =
        ActivationsFunctions::getActivationFunctionByType(functionType);
  }
  Layer(Matrix&& A, Vector&& b, ActivationFunctionType functionType) {
    assert(A.rows() == b.rows());
    A_ = std::move(A);
    b_ = std::move(b);
    activationFunction =
        ActivationsFunctions::getActivationFunctionByType(functionType);
  }

  [[nodiscard]] Vector compute(const Vector& x) const {
    assert(x.rows() == A_.cols());
    return activationFunction->compute(A_ * x + b_);
  }

  [[nodiscard]] Matrix getDerivativeA(const Vector& x, const Vector& u) const {
    assert(x.rows() == A_.cols());
    assert(u.rows() == b_.rows());
    return (activationFunction->getDerivative(A_ * x + b_)) * u * x.transpose();
  }

  [[nodiscard]] Vector getDerivativeB(const Vector& x, const Vector& u) const {
    assert(x.rows() == A_.cols());
    assert(u.rows() == b_.rows());
    return activationFunction->getDerivative(A_ * x + b_) * u;
  }

  [[nodiscard]] LayerDelta getDerivative(const Vector& x,
                                         const Vector& u) const {
    return {getDerivativeA(x, u), getDerivativeB(x, u)};
  }

  Vector getNextU(const Vector& x, Vector& u) const {
    assert(x.rows() == A_.cols());
    assert(u.rows() == b_.rows());
    return (u.transpose() * activationFunction->getDerivative(A_ * x + b_) * A_)
        .transpose();
  }

  [[nodiscard]] const Matrix& getA() const { return A_; }
  [[nodiscard]] const Vector& getB() const { return b_; }

  [[nodiscard]] ActivationFunctionType getActivationFunctionType() const {
    return activationFunction->getName();
  }

  [[nodiscard]] LayerDimension getLayerDimension() const {
    return {A_.rows(), A_.cols()};
  }

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
  ActivationFunctionPtr activationFunction;
};
}  // namespace NeuralNetwork
