#include <Eigen/Core>
#include <Eigen/Dense>
#include <vector>
#include <iostream>

namespace NeuralNetwork::ActivationsFunctions {
using Vector = Eigen::VectorXd;
using Matrix = Eigen::MatrixXd;

class BaseActivationFunction {
 public:
  virtual Vector compute(const Vector &x) = 0;
  virtual Matrix getDerivative(const Vector &x) = 0;
};

class Sigmoid : public BaseActivationFunction {
 public:
  Vector compute(const Vector &x) final {
    return exp(x.array()) / (1.0 + exp(x.array()));
  }
  Matrix getDerivative(const Vector &x) final {
    return (exp(-x.array()) / pow(1.0 + exp(x.array()), 2)).matrix().asDiagonal();
  }
};

class Relu : public BaseActivationFunction {
 public:
  Vector compute(const Vector &x) final {
    return x.cwiseMax(0.0);
  }
  Matrix getDerivative(const Vector &x) final {
    return (x.array() > 0.0).cast<double>();
  }
};
}

namespace NeuralNetwork::LossFunctions {
using Vector = Eigen::VectorXd;

class BaseLossFunction {
 public:
  virtual double computeLoss(const Vector &expected_y, const Vector &predicted_y) = 0;
  virtual Vector getDerivative(const Vector &expected_y, const Vector &predicted_y) = 0;
};

class SquareLossFunction : BaseLossFunction {
 public:
  double computeLoss(const Vector &expected_y, const Vector &predicted_y) final {
    assert(expected_y.size() == predicted_y.size());
    return (expected_y - predicted_y).squaredNorm() / expected_y.size();
  }
  Vector getDerivative(const Vector &expected_y, const Vector &predicted_y) final {
    return 2 * (expected_y - predicted_y);
  }
};

}

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

  const Layer &operator+=(const LayerDelta &other) & {
    A_ += other.deltaA_;
    b_ += other.deltaB_;
    return *this;
  }

 private:
  Matrix A_;
  Vector b_;
  ActivationFunction nonLinearFunction_;
};


class NeuralNetwork {
  using LossFunction = std::unique_ptr<LossFunctions::BaseLossFunction>;
  using ActivationFunction = std::unique_ptr<ActivationsFunctions::BaseActivationFunction>;
  using LayersDeltas = std::vector<LayerDelta>;
 public:
  NeuralNetwork(
      std::vector<LayerDimension> &dimensions,
      std::vector<ActivationFunction> &activationFunctions,
      LossFunction lossFunction
  ) {
    assert(!dimensions.empty());
    assert(dimensions.size() == activationFunctions.size());
    size_t layers_size = dimensions.size();
    layers_.reserve(layers_size);
    for (size_t index = 0; index < layers_size; ++index) {
      layers_.emplace_back(
          Layer(dimensions[index], std::move(activationFunctions[index]))
      );
    }
    lossFunction_ = std::move(lossFunction);
  }
  Vector predict(const Vector &x) {
    assert (x.size() == layers_[0].getLayerDimension().m);
    Vector computedX = x;
    for (const auto &layer: layers_) {
      computedX = layer.compute(computedX);
    }
    return computedX;
  }

  std::vector<Vector> computeEachLayer(const Vector &x) {
    Vector computed = x;
    std::vector<Vector> computedLayers;
    computedLayers.reserve(layers_.size() + 1);
    computedLayers.emplace_back(computed);
    for (const auto &layer: layers_) {
      computed = layer.compute(computed);
      computedLayers.push_back(computed);
    }
    return computedLayers;
  }

  void addOnIterationsDeltas(const Vector &x, const Vector &y, LayersDeltas &layersDeltas) {
    ssize_t layers_count = layers_.size();
    std::vector<Vector> computedLayers = computeEachLayer(x);

    auto u = lossFunction_->getDerivative(y, computedLayers[layers_count]);
    for (ssize_t index = layers_count - 1; index < 0; --index) {
      layersDeltas[index] -= layers_[index].getDerivative(
          computedLayers[index], u
          );
      u = layers_[index].getNextU(computedLayers[index], u);
    }
  }

  void trainByBatch(std::vector<Vector> batchX, std::vector<Vector> batchY) {
    assert (batchX.size() == batchY.size());
    LayersDeltas layersDeltas(layers_.size());
    for (size_t index = 0; index < batchX.size(); ++index) {
      addOnIterationsDeltas(batchX[index], batchY[index], layersDeltas);
    }
    for (size_t index = 0; index < layers_.size(); ++index) {
      layers_[index] += layersDeltas[index];
    }
  }

 private:
  std::vector<Layer> layers_;
  LossFunction lossFunction_;
};
}

int main() {
}
