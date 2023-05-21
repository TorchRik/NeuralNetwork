#include <NeuralNetwork/neural_network.h>

#include <gtest/gtest.h>

TEST(layerTest, randomInit) {
  auto layer = NeuralNetwork::Layer(
      {5, 10},
      NeuralNetwork::ActivationsFunctions::ActivationFunctionType::Relu);
  NeuralNetwork::LayerDimension expectedDimension{5, 10};
  assert(layer.getLayerDimension().n == expectedDimension.n);
}

TEST(sigmoidTest, derivative) {
  auto function =
      NeuralNetwork::ActivationsFunctions::getActivationFunctionByType(
          NeuralNetwork::ActivationsFunctions::ActivationFunctionType::Relu);
  NeuralNetwork::Vector x(10);
  std::cout << function->getDerivative(x);
}
