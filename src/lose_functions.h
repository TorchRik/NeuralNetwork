#include <Eigen/Core>
#include <Eigen/Dense>
#include <vector>
#include <iostream>

namespace NeuralNetwork::LossFunctions {
using Vector = Eigen::VectorXd;

class BaseLossFunction {
 public:
  virtual double computeLoss(const Vector &expected_y, const Vector &predicted_y) = 0;
  virtual Vector getDerivative(const Vector &expected_y, const Vector &predicted_y) = 0;
};

class SquareLossFunction : BaseLossFunction {
 public:
  double computeLoss(const Vector &expected_y, const Vector &predicted_y) final {
    assert(expected_y.size() == predicted_y.size());
    return (expected_y - predicted_y).squaredNorm() / expected_y.size();
  }
  Vector getDerivative(const Vector &expected_y, const Vector &predicted_y) final {
    return 2 * (expected_y - predicted_y);
  }
};

}
