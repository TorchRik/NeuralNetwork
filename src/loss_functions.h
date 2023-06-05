#include <Eigen/Core>
#include <Eigen/Dense>
#include <vector>
#include "definitions.h"

namespace NN {
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
  double computeLoss(const Vector& expected_y, const Vector& predicted_y) final;
  Vector getDerivative(const Vector& expected_y,
                       const Vector& predicted_y) final;
  LossFunctionType getType() final;
};

std::unique_ptr<BaseLossFunction> getLossFunctionByType(LossFunctionType type);

}  // namespace NN
