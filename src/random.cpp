#include "random.h"

namespace NN::Random {
Random::Random(unsigned int seed) : generator_(seed){};

void Random::generateIndexes(ssize_t dataSize, ssize_t batchSize,
                             std::vector<ssize_t>* indexes) {
  std::uniform_int_distribution<> dist(0, dataSize - 1);
  indexes->clear();
  indexes->reserve(batchSize);
  for (ssize_t j = 0; j < batchSize; ++j) {
    (*indexes).push_back(dist(generator_));
  }
}
void Random::generateRandomWeights(ssize_t startDimension, ssize_t endDimension,
                                   Matrix* A, Vector* b) {
  (*A) = Eigen::Rand::normal<Matrix>(endDimension, startDimension, generator_);
  (*b) = Eigen::Rand::normal<Matrix>(endDimension, 1, generator_);
}
}  // namespace NN::Random