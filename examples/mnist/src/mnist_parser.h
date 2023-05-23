#include <Eigen/Core>
#include <vector>
#include <fstream>
#include <iostream>


namespace MNIST {
const int kImageSize = 28 * 28;

Eigen::VectorXd imageToVector(const unsigned char* image_data);

std::vector<Eigen::VectorXd> readMnistImages(const std::string& file_path);
Eigen::VectorXd performNumToProbabilityVector(int num);
std::vector<Eigen::VectorXd> readMnistLabels(const std::string& file_path);

}  // namespace MNIST