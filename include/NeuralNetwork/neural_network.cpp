#include "neural_network.h"

namespace NN {

NeuralNetwork::NeuralNetwork(
    const std::initializer_list<ssize_t>& dimensions,
    const std::initializer_list<ActivationFunctionType>&
        activationFunctionsTypes,
    LossFunctionType lossFunctionType, unsigned int randomSeed)
    : random_(randomSeed) {
  assert(activationFunctionsTypes.size() > 0);
  assert(dimensions.size() == activationFunctionsTypes.size() + 1);

  layers_.reserve(activationFunctionsTypes.size());
  auto dimensionIt = dimensions.begin();
  auto functionIt = activationFunctionsTypes.begin();
  assert(*(dimensionIt++) > 0);
  for (; dimensionIt != dimensions.end() &&
         functionIt != activationFunctionsTypes.end();
       ++dimensionIt, ++functionIt) {
    assert(*dimensionIt > 0);
    layers_.emplace_back(*(dimensionIt - 1), *dimensionIt, *functionIt, random_);
  }
  lossFunction_ = LossFunctions::getLossFunctionByType(lossFunctionType);
}

NeuralNetwork::NeuralNetwork(unsigned int randomSeed): random_(randomSeed) {};

[[nodiscard]] Vector NeuralNetwork::predict(const Vector& x) const {
  assert(x.size() == layers_[0].getStartDimension());
  Vector computedX = x;
  for (const auto& layer : layers_) {
    computedX = layer.compute(std::move(computedX));
  }
  return computedX;
}
[[nodiscard]] std::vector<Vector> NeuralNetwork::computeEachLayer(
    const Vector& x) const {
  assert(x.size() == layers_[0].getStartDimension());

  std::vector<Vector> computedLayers;
  computedLayers.reserve(layers_.size() + 1);
  computedLayers.push_back(x);
  for (const auto& layer : layers_) {
    computedLayers.push_back(layer.compute(computedLayers.back()));
  }
  return computedLayers;
}
void NeuralNetwork::trainByOneExample(const Vector& x, const Vector& y,
                                      std::vector<LayerDelta>* layersDeltas) {
  assert(x.size() == layers_[0].getStartDimension());
  assert(y.size() == layers_[layers_.size() - 1].getEndDimension());

  ssize_t layers_count = layers_.size();
  std::vector<Vector> computedLayers = computeEachLayer(x);

  auto u = lossFunction_->getDerivative(y, computedLayers[layers_count]);
  for (ssize_t index = layers_count - 1; index >= 0; --index) {
    (*layersDeltas)[index] -=
        layers_[index].getDerivative(computedLayers[index], u);
    u = layers_[index].getNextU(computedLayers[index], u);
  }
}
void NeuralNetwork::trainByBatch(const std::vector<Vector>& batchX,
                                 const std::vector<Vector>& batchY,
                                 const std::vector<ssize_t>& indexes) {
  std::vector<LayerDelta> layersDeltas;
  for (auto const& layer : layers_) {
    layersDeltas.push_back(
        {Matrix::Zero(layer.getEndDimension(), layer.getStartDimension()),
         Vector::Zero(layer.getEndDimension())});
  }
  for (ssize_t index : indexes) {
    trainByOneExample(batchX[index], batchY[index], &layersDeltas);
  }
  for (ssize_t j = 0; j < layers_.size(); ++j) {
    layersDeltas[j] /= indexes.size();
    layers_[j] += layersDeltas[j];
  }
}

double NeuralNetwork::getAverageLoss(const std::vector<Vector>& X,
                                     const std::vector<Vector>& Y) const {
  assert(X.size() == Y.size());
  double loss = 0;
  for (auto x = X.begin(), y = Y.begin(); x != X.end() && y != Y.end();
       ++x, ++y) {
    loss += lossFunction_->computeLoss(predict(*x), *y);
  }
  return loss / X.size();
}

void NeuralNetwork::train(ssize_t iterationCount, ssize_t batchSize,
                          double expected_loss, const std::vector<Vector>& X,
                          const std::vector<Vector>& Y,
                          const std::vector<Vector>& testX,
                          const std::vector<Vector>& testY) {
  ssize_t currentIterationNum = 0;
  assert(X.size() == Y.size());

  std::vector<ssize_t> indexes(batchSize);

  for (ssize_t i = 0; i < iterationCount; ++i) {
    random_.generateIndexes(X.size(), batchSize, &indexes);
    trainByBatch(X, Y, indexes);
    double loss = getAverageLoss(testX, testY);
    std::cout << "Iteration " << ++currentIterationNum
              << " average loss for test data " << loss << std::endl;
    if (loss < expected_loss) {
      break;
    }
  }
}

std::ostream& operator<<(std::ostream& os, const NeuralNetwork& model) {
  os << model.layers_;
  LossFunctionType lossFunctionType = model.lossFunction_->getType();
  os.write(reinterpret_cast<const char*>(&lossFunctionType),
           sizeof(lossFunctionType));
  return os;
}

std::istream& operator>>(std::istream& is, NeuralNetwork& model) {
  is >> model.layers_;
  LossFunctionType lossFunctionType;
  is.read(reinterpret_cast<char*>(&lossFunctionType), sizeof(lossFunctionType));
  model.lossFunction_ = LossFunctions::getLossFunctionByType(lossFunctionType);
  return is;
}

}  // namespace NN
