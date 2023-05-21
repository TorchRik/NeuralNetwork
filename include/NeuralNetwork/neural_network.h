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
using LossFunctionType = LossFunctions::LossFunctionType;
using ActivationFunctionType = ActivationsFunctions::ActivationFunctionType;

class NeuralNetwork {
  using LossFunctionPtr = std::unique_ptr<LossFunctions::BaseLossFunction>;

 public:
  NeuralNetwork(const std::initializer_list<ssize_t>& dimensions,
                const std::initializer_list<ActivationFunctionType>&
                    activationFunctionsTypes,
                LossFunctionType lossFunctionType);

  NeuralNetwork() = default;

  [[nodiscard]] Vector predict(const Vector& x) const;

  double getAverageLoss(const std::vector<Vector>& X,
                        const std::vector<Vector>& Y) const;

  void train(ssize_t iterationCount, ssize_t batchSize, double expected_loss,
             const std::vector<Vector>& X, const std::vector<Vector>& Y,
             const std::vector<Vector>& testX,
             const std::vector<Vector>& testY);

  friend std::istream& operator>>(std::istream& is, NeuralNetwork& model);
  friend std::ostream& operator<<(std::ostream& os, const NeuralNetwork& model);

 private:
  [[nodiscard]] std::vector<Vector> computeEachLayer(const Vector& x) const;

  void trainByBatch(const std::vector<Vector>& batchX,
                    const std::vector<Vector>& batchY,
                    const std::vector<ssize_t>& indexes);

  void trainByOneExample(const Vector& x, const Vector& y,
                         std::vector<LayerDelta>* layersDeltas);
  std::vector<Layer> layers_;
  LossFunctionPtr lossFunction_;
};
}  // namespace NeuralNetwork
