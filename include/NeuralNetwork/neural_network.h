#include <Eigen/Core>
#include <Eigen/Dense>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>
#include "../src/layer.h"
#include "../src/lose_functions.h"

namespace NeuralNetwork {
using Vector = Eigen::VectorXd;
using Matrix = Eigen::MatrixXd;

class NeuralNetwork {
  using LossFunctionPtr = std::unique_ptr<LossFunctions::BaseLossFunction>;
  using ActivationFunctionType = ActivationsFunctions::ActivationFunctionType;
  using LossFunctionType = LossFunctions::LossFunctionType;

 public:
  NeuralNetwork(std::vector<LayerDimension>& dimensions,
                std::vector<ActivationFunctionType> activationFunctionsTypes,
                LossFunctionType lossFunctionType) {
    assert(!dimensions.empty());
    assert(dimensions.size() == activationFunctionsTypes.size());

    LayerDimension lastDimension{};
    size_t layers_size = dimensions.size();
    layers_.reserve(layers_size);
    for (size_t index = 0; index < layers_size; ++index) {
      if (index) {
        assert(lastDimension.n == dimensions[index].m);
      }
      lastDimension = dimensions[index];
      layers_.emplace_back(
          Layer(dimensions[index], activationFunctionsTypes[index]));
    }
    lossFunction_ = LossFunctions::getLossFunctionByType(lossFunctionType);
  }

  NeuralNetwork(const std::string& pathToFile) {
    std::ifstream file(pathToFile, std::ios::binary);
    assert(file);
    size_t layers_size;
    file.read(reinterpret_cast<char*>(&layers_size), sizeof(layers_size));
    layers_.reserve(layers_size);
    for (size_t i = 0; i < layers_size; ++i) {
      size_t rows, cols, vec_size;
      ActivationFunctionType type;

      file.read(reinterpret_cast<char*>(&type), sizeof(type));

      file.read(reinterpret_cast<char*>(&rows), sizeof(rows));
      file.read(reinterpret_cast<char*>(&cols), sizeof(cols));
      file.read(reinterpret_cast<char*>(&vec_size), sizeof(vec_size));

      Matrix A(rows, cols);
      Vector b(vec_size);

      file.read(reinterpret_cast<char*>(A.data()), sizeof(double) * A.size());
      file.read(reinterpret_cast<char*>(b.data()), sizeof(double) * vec_size);

      layers_.emplace_back(Layer(std::move(A), std::move(b), type));
    }

    LossFunctionType lossFunctionType;
    file.read(reinterpret_cast<char*>(&lossFunctionType),
              sizeof(lossFunctionType));

    lossFunction_ = LossFunctions::getLossFunctionByType(lossFunctionType);
    file.close();
  }

  [[nodiscard]] Vector predict(const Vector& x) const {
    assert(x.size() == layers_[0].getLayerDimension().m);
    Vector computedX = x;
    for (const auto& layer : layers_) {
      computedX = layer.compute(computedX);
    }
    return computedX;
  }

  [[nodiscard]] std::vector<Vector> computeEachLayer(const Vector& x) const {
    assert(x.size() == layers_[0].getLayerDimension().m);

    Vector computed = x;
    std::vector<Vector> computedLayers;
    computedLayers.emplace_back(computed);
    for (const auto& layer : layers_) {
      computed = layer.compute(computed);
      computedLayers.push_back(computed);
    }
    return computedLayers;
  }

  void addOnIterationsDeltas(const Vector& x, const Vector& y,
                             std::vector<LayerDelta>& layersDeltas) {
    assert(x.size() == layers_[0].getLayerDimension().m);
    assert(y.size() == layers_[layers_.size() - 1].getLayerDimension().n);

    size_t layers_count = layers_.size();
    std::vector<Vector> computedLayers = computeEachLayer(x);

    auto u = lossFunction_->getDerivative(y, computedLayers[layers_count]);
    for (size_t index = layers_count - 1;; --index) {
      layersDeltas[index] -=
          layers_[index].getDerivative(computedLayers[index], u);
      u = layers_[index].getNextU(computedLayers[index], u);
      if (!index) {
        break;
      }
    }
  }

  void trainByBatch(std::vector<Vector> batchX, std::vector<Vector> batchY,
                    std::vector<size_t>& indexes) {
    std::vector<LayerDelta> layersDeltas;
    for (auto const& layer : layers_) {
      auto layerDimension = layer.getLayerDimension();
      layersDeltas.emplace_back(
          LayerDelta{Matrix::Zero(layerDimension.n, layerDimension.m),
                     Vector::Zero(layerDimension.n)});
    }
    for (size_t index : indexes) {
      addOnIterationsDeltas(batchX[index], batchY[index], layersDeltas);
    }
    for (size_t j = 0; j < layers_.size(); ++j) {
      layersDeltas[j] /= indexes.size();
      layers_[j] += layersDeltas[j];
    }
  }

  double getAverageLoss(std::vector<Vector>& X, std::vector<Vector>& Y) const {
    assert(X.size() == Y.size());
    double loss = 0;
    for (size_t index = 0; index < X.size(); ++index) {
      loss +=
          lossFunction_->computeLoss(predict(X[index]), Y[index]) / X.size();
    }
    return loss;
  }

  void train(size_t iterationCount, size_t batchSize, std::vector<Vector>& X,
             std::vector<Vector>& Y, std::vector<Vector>& testX,
             std::vector<Vector>& testY) {
    ssize_t currentIterationNum = 0;
    assert(X.size() == Y.size());

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dist(0, X.size() - 1);
    std::vector<size_t> indexes(batchSize);

    for (size_t i = 0; i < iterationCount; ++i) {
      for (size_t j = 0; j < batchSize; ++j) {
        indexes[j] = dist(gen);
      }
      trainByBatch(X, Y, indexes);
      std::cout << "Iteration " << ++currentIterationNum
                << " average loss for test data "
                << getAverageLoss(testX, testY) << std::endl;
    }
  }

  void saveDataToFile(const std::string& pathToFile) const {
    std::ofstream file(pathToFile, std::ios::binary);
    assert(file);

    const size_t num_layers = layers_.size();
    file.write(reinterpret_cast<const char*>(&num_layers), sizeof(num_layers));

    for (auto const& layer : layers_) {
      const auto& A = layer.getA();
      const auto& b = layer.getB();
      const auto functionType = layer.getActivationFunctionType();
      file.write(reinterpret_cast<const char*>(&functionType),
                 sizeof(functionType));

      ssize_t rows = A.rows();
      ssize_t cols = A.cols();
      ssize_t vec_size = b.size();

      file.write(reinterpret_cast<const char*>(&rows), sizeof(rows));
      file.write(reinterpret_cast<const char*>(&cols), sizeof(cols));
      file.write(reinterpret_cast<const char*>(&vec_size), sizeof(vec_size));

      file.write(reinterpret_cast<const char*>(A.data()),
                 sizeof(double) * A.size());
      file.write(reinterpret_cast<const char*>(b.data()),
                 sizeof(double) * vec_size);
    }

    LossFunctionType lossFunctionType = lossFunction_->getType();
    file.write(reinterpret_cast<const char*>(&lossFunctionType),
               sizeof(lossFunctionType));
    file.close();
  }
 private:
  std::vector<Layer> layers_;
  LossFunctionPtr lossFunction_;
};
}  // namespace NeuralNetwork
