#include <Eigen/Core>
#include <Eigen/Dense>
#include <vector>

namespace NN::LossFunctions {
using Vector = Eigen::VectorXd;

enum class LossFunctionType {
  Square,
};

class BaseLossFunction {
 public:
  virtual double computeLoss(const Vector& expected_y,
                             const Vector& predicted_y) = 0;
  virtual Vector getDerivative(const Vector& expected_y,
                               const Vector& predicted_y) = 0;
  virtual LossFunctionType getType() = 0;
  virtual ~BaseLossFunction() = default;
};

class SquareLossFunction : public BaseLossFunction {
 public:
  double computeLoss(const Vector& expected_y,
                     const Vector& predicted_y) final {
    assert(expected_y.size() == predicted_y.size());
    return (expected_y - predicted_y).squaredNorm() / expected_y.size();
  }
  Vector getDerivative(const Vector& expected_y,
                       const Vector& predicted_y) final {
    return 2 * (predicted_y - expected_y);
  }
  LossFunctionType getType() final { return LossFunctionType::Square; }
};

inline std::unique_ptr<BaseLossFunction> getLossFunctionByType(
    LossFunctionType type) {
  switch (type) {
    case LossFunctionType::Square:
      return std::make_unique<SquareLossFunction>();
    default:
      throw std::runtime_error("Incorrect type provided");
  }
}

}  // namespace NN::LossFunctions
