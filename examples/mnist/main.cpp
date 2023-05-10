#include <NeuralNetwork/neural_network.h>

#include <fstream>
#include <iostream>
#include <vector>

const int kImageSize = 28 * 28;

using Vector = NeuralNetwork::Vector;
using ActivationFunctionType =
    NeuralNetwork::ActivationsFunctions::ActivationFunctionType;
using LossFunctionType = NeuralNetwork::LossFunctions::LossFunctionType;

Vector imageToVector(const unsigned char* image_data) {
  Vector image_vector(kImageSize);

  for (int i = 0; i < kImageSize; i++) {
    image_vector[i] = static_cast<double>(image_data[i]) / 255.0;
  }

  return image_vector;
}

std::vector<Vector> readMnistImages(const std::string& file_path) {
  std::ifstream file(file_path, std::ios::binary);

  int magic_number, num_images, num_rows, num_cols;
  file.read(reinterpret_cast<char*>(&magic_number), sizeof(magic_number));
  file.read(reinterpret_cast<char*>(&num_images), sizeof(num_images));
  file.read(reinterpret_cast<char*>(&num_rows), sizeof(num_rows));
  file.read(reinterpret_cast<char*>(&num_cols), sizeof(num_cols));

  magic_number = __builtin_bswap32(magic_number);
  num_images = __builtin_bswap32(num_images);
  num_rows = __builtin_bswap32(num_rows);
  num_cols = __builtin_bswap32(num_cols);

  if (magic_number != 2051) {
    std::cerr << "Invalid magic number: " << magic_number << std::endl;
    return {};
  }

  if (num_rows != 28 || num_cols != 28) {
    std::cerr << "Invalid image dimensions: " << num_rows << "x" << num_cols
              << std::endl;
    return {};
  }

  std::vector<Vector> images(num_images);

  for (int i = 0; i < num_images; i++) {
    std::vector<unsigned char> image_data(kImageSize);
    file.read(reinterpret_cast<char*>(image_data.data()), kImageSize);
    images[i] = imageToVector(image_data.data());
  }

  return images;
}

Vector performNumToProbabilityVector(int num) {
  Vector v = Eigen::VectorXd::Zero(10);
  v[num] = 1;
  return v;
}

// Read the MNIST labels from a ubyte file
std::vector<Vector> readMnistLabels(const std::string& file_path) {
  std::ifstream file(file_path, std::ios::binary);

  if (!file) {
    std::cerr << "Failed to open file: " << file_path << std::endl;
    return {};
  }

  int magic_number, num_labels;
  file.read(reinterpret_cast<char*>(&magic_number), sizeof(magic_number));
  file.read(reinterpret_cast<char*>(&num_labels), sizeof(num_labels));

  magic_number = __builtin_bswap32(magic_number);
  num_labels = __builtin_bswap32(num_labels);

  if (magic_number != 2049) {
    std::cerr << "Invalid magic number: " << magic_number << std::endl;
    return {};
  }

  std::vector<Vector> labels(num_labels);

  for (int i = 0; i < num_labels; i++) {
    unsigned char label;
    file.read(reinterpret_cast<char*>(&label), sizeof(label));
    labels[i] = performNumToProbabilityVector(static_cast<int>(label));
  }

  return labels;
}

int main(int, char*[]) {
  auto images = readMnistImages(
      "/Users/torchrik/stash/cpp-lib-template/examples/mnist/data/"
      "t10k-images-idx3-ubyte");
  auto labels = readMnistLabels(
      "/Users/torchrik/stash/cpp-lib-template/examples/mnist/data/"
      "t10k-labels-idx1-ubyte");
  auto imagesTest = readMnistImages(
      "/Users/torchrik/stash/cpp-lib-template/examples/mnist/data/"
      "t10k-images-idx3-ubyte");
  auto labelsTest = readMnistLabels(
      "/Users/torchrik/stash/cpp-lib-template/examples/mnist/data/"
      "t10k-labels-idx1-ubyte");

  size_t layersCount = 3;
  ssize_t imageSize = images[0].size();
  ssize_t predictionSize = 10;
  std::cout << imageSize << "\n";
  std::vector<NeuralNetwork::LayerDimension> dimensions(layersCount);
  dimensions[0] = {100, imageSize};
  dimensions[1] = {100, 100};
  dimensions[2] = {10, 100};
  std::vector<ActivationFunctionType> functions(layersCount,
                                                ActivationFunctionType::SIGMOID);
  auto model = NeuralNetwork::NeuralNetwork(dimensions, functions,
                                            LossFunctionType::SQUARE);
  model.train(5, 500, images, labels, imagesTest, labelsTest);
  model.saveDataToFile(
      "/Users/torchrik/stash/cpp-lib-template/examples/mnist/saved_model");
}
