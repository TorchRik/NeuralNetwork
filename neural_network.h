namespace NeuralNetwork {
using Vector = Eigen::VectorXd;
using Matrix = Eigen::MatrixXd;

class NeuralNetwork {
  using LossFunction = std::unique_ptr<LossFunctions::BaseLossFunction>;
  using ActivationFunction = std::unique_ptr<ActivationsFunctions::BaseActivationFunction>;
  using LayersDeltas = std::vector<LayerDelta>;
 public:
  NeuralNetwork(
      std::vector<LayerDimension> &dimensions,
      std::vector<ActivationFunction> &&activationFunctions,
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
    size_t layers_count = layers_.size();
    std::vector<Vector> computedLayers = computeEachLayer(x);

    auto u = lossFunction_->getDerivative(y, computedLayers[layers_count]);
    for (size_t index = layers_count - 1; ; --index) {
      layersDeltas[index] -= layers_[index].getDerivative(
          computedLayers[index], u
      );
      u = layers_[index].getNextU(computedLayers[index], u);
      if (!index) {
        break;
      }
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