#include <NeuralNetwork/neural_network.h>
#include "src/mnist_parser.h"

#include <iostream>
#include <vector>

using namespace NN;

double getCountPredicted(const NN::NeuralNetwork& model, std::vector<Vector>& X,
                         std::vector<Vector>& Y) {
  double count = 0;
  for (size_t i = 0; i < X.size(); ++i) {
    auto p = model.predict(X[i]);
    int max_index = 0;
    for (auto j = 0; j < p.size(); ++j) {
      if (p[j] > p[max_index]) {
        max_index = j;
      }
    }
    count += Y[i][max_index];
  }
  return count;
}

int main(int, char*[]) {
  auto images = MNIST::readMnistImages(
      "/Users/torchrik/stash/cpp-lib-template/examples/mnist/data/"
      "t10k-images-idx3-ubyte");
  auto labels = MNIST::readMnistLabels(
      "/Users/torchrik/stash/cpp-lib-template/examples/mnist/data/"
      "t10k-labels-idx1-ubyte");
  auto imagesTest = MNIST::readMnistImages(
      "/Users/torchrik/stash/cpp-lib-template/examples/mnist/data/"
      "t10k-images-idx3-ubyte");
  auto labelsTest = MNIST::readMnistLabels(
      "/Users/torchrik/stash/cpp-lib-template/examples/mnist/data/"
      "t10k-labels-idx1-ubyte");
  ssize_t hidden_size = 16;
  ssize_t output_size = 10;
  ssize_t iteration_count = 1000;
  ssize_t batch_size = 2000;
  double expected_loss = 0.01;

  auto model = NN::NeuralNetwork(
      {images[0].size(), hidden_size, hidden_size, output_size},
      {NN::ActivationFunctionType::Relu, NN::ActivationFunctionType::Relu,
       NN::ActivationFunctionType::Softmax},
      NN::LossFunctionType::Square);
  model.train(iteration_count, batch_size, expected_loss, images, labels,
              imagesTest, labelsTest);

  std::ofstream file(
      "/Users/torchrik/stash/cpp-lib-template/examples/mnist/saved_models/1",
      std::ios::binary);
  file << model;
  file.close();
  NeuralNetwork new_model;
  std::ifstream file_is(
      "/Users/torchrik/stash/cpp-lib-template/examples/mnist/saved_models/1",
      std::ios::binary);
  file_is >> new_model;
  std::cout << getCountPredicted(model, imagesTest, labelsTest) << std::endl;
  std::cout << getCountPredicted(new_model, imagesTest, labelsTest);
}