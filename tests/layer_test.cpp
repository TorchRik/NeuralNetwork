#include <NeuralNetwork/neural_network.h>
#include <math.h>

#include <gtest/gtest.h>

TEST(squareLossFunctionTest, derivative) {
  auto function = NN::getLossFunctionByType(NN::LossFunctionType::Square);
  NN::Vector x(3);
  NN::Vector y(3);
  x << 1, 2, 3;
  NN::Vector expected(3);
  expected << -2, -4, -6;
  assert(function->getDerivative(x, y) == expected);
}

TEST(squareLossFunctionTest, computeLoss) {
  auto function = NN::getLossFunctionByType(NN::LossFunctionType::Square);
  NN::Vector x(3);
  NN::Vector y(3);
  x << 1, 2, 3;
  assert(function->computeLoss(x, y) == sqrt(1 + 4 + 9));
}
