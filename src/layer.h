#include <Eigen/Core>
#include <Eigen/Dense>
#include <utility>
#include <vector>
#include "activations_functions.h"
#include "random.h"
#include "definitions.h"

namespace NN {
struct LayerDelta {
  Matrix deltaA;
  Vector deltaB;

  LayerDelta& operator+=(const LayerDelta& other);
  LayerDelta& operator-=(const LayerDelta& other);
  LayerDelta& operator/=(double num);
  LayerDelta& operator*=(double num);
  LayerDelta operator/(double num);
  LayerDelta operator*(double num);
};

class Layer {
  using ActivationFunctionPtr = std::unique_ptr<BaseActivationFunction>;
  using ActivationFunctionType = ActivationFunctionType;

 public:
  Layer(ssize_t startDimension, ssize_t endDimension,
        ActivationFunctionType functionType, Random& random);
  Layer(Matrix&& A, Vector&& b, ActivationFunctionType functionType);
  Layer() = default;

  [[nodiscard]] Vector compute(Vector x) const;

  [[nodiscard]] Matrix getDerivativeA(const Vector& x, const Vector& u) const;

  [[nodiscard]] Vector getDerivativeB(const Vector& x, const Vector& u) const;

  [[nodiscard]] LayerDelta getDerivative(const Vector& x,
                                         const Vector& u) const;

  Vector getNextU(const Vector& x, Vector& u) const;

  [[nodiscard]] const Matrix& getA() const;
  [[nodiscard]] const Vector& getB() const;

  [[nodiscard]] ActivationFunctionType getActivationFunctionType() const;

  [[nodiscard]] ssize_t getStartDimension() const;
  [[nodiscard]] ssize_t getEndDimension() const;

  Layer& operator+=(const LayerDelta& delta);

  friend std::ostream& operator<<(std::ostream& os, const Layer& person);
  friend std::istream& operator>>(std::istream& is, Layer& layer);

 private:
  Matrix A_;
  Vector b_;
  ActivationFunctionPtr activationFunction;
};
std::ostream& operator<<(std::ostream& os, const std::vector<Layer>& layers);
std::istream& operator>>(std::istream& is, std::vector<Layer>& layers);
}  // namespace NN
