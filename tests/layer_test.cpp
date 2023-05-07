#include <NeuralNetwork/neural_network.h>

#include <gtest/gtest.h>


TEST(layer_test, random_init) {
  Eigen::VectorXd a(5);
  std::cout << a.rows();
//  auto layer = NeuralNetwork::Layer(
//      {5, 10},
//      std::make_unique<NeuralNetwork::ActivationsFunctions::Relu>()
//      );
//  NeuralNetwork::LayerDimension expectedDimension{5, 10};
//  assert(layer.getLayerDimension().n == expectedDimension.n);
}
