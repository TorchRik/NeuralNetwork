#include <Eigen/Core>
#include <Eigen/Dense>
#include <vector>
#include "definitions.h"

namespace NN {
enum class ActivationFunctionType { Relu, Sigmoid, Softmax };

class BaseActivationFunction {
 public:
  virtual Vector compute(const Vector& x) = 0;
  virtual Matrix getDerivative(const Vector& x) = 0;
  virtual ActivationFunctionType getName() = 0;
  virtual ~BaseActivationFunction() = default;
};

class Sigmoid : public BaseActivationFunction {
 public:
  Vector compute(const Vector& x) final;
  Matrix getDerivative(const Vector& x) final;
  ActivationFunctionType getName() final;
};

class Relu : public BaseActivationFunction {
 public:
  Vector compute(const Vector& x) final;
  Matrix getDerivative(const Vector& x) final;
  ActivationFunctionType getName() final;
};

class Softmax : public BaseActivationFunction {
 public:
  Vector compute(const Vector& x) final;
  Matrix getDerivative(const Vector& x) final;
  ActivationFunctionType getName() final;
};

std::unique_ptr<BaseActivationFunction> getActivationFunctionByType(
    ActivationFunctionType type);
}  // namespace NN
