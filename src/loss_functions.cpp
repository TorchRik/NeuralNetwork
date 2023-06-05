#include "loss_functions.h"

namespace NN {

double SquareLossFunction::computeLoss(const Vector& expected_y,
                                       const Vector& predicted_y) {
  assert(expected_y.size() == predicted_y.size());
  return (expected_y - predicted_y).squaredNorm() / expected_y.size();
}

Vector SquareLossFunction::getDerivative(const Vector& expected_y,
                                         const Vector& predicted_y) {
  return 2 * (predicted_y - expected_y);
}

LossFunctionType SquareLossFunction::getType() {
  return LossFunctionType::Square;
}

std::unique_ptr<BaseLossFunction> getLossFunctionByType(LossFunctionType type) {
  switch (type) {
    case LossFunctionType::Square:
      return std::make_unique<SquareLossFunction>();
    default:
      throw std::runtime_error("Incorrect type provided");
  }
}

}  // namespace NN
