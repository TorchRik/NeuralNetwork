#include <Eigen/Core>
#include <EigenRand/EigenRand>
#include <random>

namespace NN::Random {
using Matrix = Eigen::MatrixXd;
using Vector = Eigen::VectorXd;
class Random {
 public:
  Random(unsigned int seed);

  void generateIndexes(ssize_t dataSize, ssize_t batchSize,
                       std::vector<ssize_t>* indexes);
  void generateRandomWeights(ssize_t startDimension, ssize_t endDimension,
                             Matrix* A, Vector* b);

 private:
  std::mt19937 generator_;
};
}  // namespace NN::Random