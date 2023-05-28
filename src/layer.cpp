#include "layer.h"

namespace NN {

LayerDelta& LayerDelta::operator+=(const LayerDelta& other) {
  deltaA += other.deltaA;
  deltaB += other.deltaB;
  return *this;
}
LayerDelta& LayerDelta::operator-=(const LayerDelta& other) {
  deltaA -= other.deltaA;
  deltaB -= other.deltaB;
  return *this;
}
LayerDelta& LayerDelta::operator/=(double num) {
  deltaA /= num;
  deltaB /= num;
  return *this;
}
LayerDelta& LayerDelta::operator*=(double num) {
  deltaA *= num;
  deltaB *= num;
  return *this;
}
LayerDelta LayerDelta::operator/(double num) {
  auto result = *this;
  result /= num;
  return result;
}
LayerDelta LayerDelta::operator*(double num) {
  auto result = *this;
  result *= num;
  return result;
}

Layer::Layer(ssize_t startDimension, ssize_t endDimension,
             ActivationFunctionType functionType, Random::Random& random)
    : activationFunction(
          ActivationsFunctions::getActivationFunctionByType(functionType)) {
  assert(startDimension > 0 && endDimension > 0);
  random.generateRandomWeights(startDimension, endDimension, &A_, &b_);
}
Layer::Layer(Matrix&& A, Vector&& b, ActivationFunctionType functionType)
    : A_(std::move(A)),
      b_(std::move(b)),
      activationFunction(
          ActivationsFunctions::getActivationFunctionByType(functionType)) {
  assert(A_.rows() == b_.rows());
}

[[nodiscard]] Vector Layer::compute(Vector x) const {
  assert(x.rows() == A_.cols());
  return activationFunction->compute(A_ * x + b_);
}

[[nodiscard]] Matrix Layer::getDerivativeA(const Vector& x,
                                           const Vector& u) const {
  assert(x.rows() == A_.cols());
  assert(u.rows() == b_.rows());
  return (activationFunction->getDerivative(A_ * x + b_)) * u * x.transpose();
}

[[nodiscard]] Vector Layer::getDerivativeB(const Vector& x,
                                           const Vector& u) const {
  assert(x.rows() == A_.cols());
  assert(u.rows() == b_.rows());
  return activationFunction->getDerivative(A_ * x + b_) * u;
}

[[nodiscard]] LayerDelta Layer::getDerivative(const Vector& x,
                                              const Vector& u) const {
  return {getDerivativeA(x, u), getDerivativeB(x, u)};
}

Vector Layer::getNextU(const Vector& x, Vector& u) const {
  assert(x.rows() == A_.cols());
  assert(u.rows() == b_.rows());
  return (u.transpose() * activationFunction->getDerivative(A_ * x + b_) * A_)
      .transpose();
}

[[nodiscard]] const Matrix& Layer::getA() const {
  return A_;
}
[[nodiscard]] const Vector& Layer::getB() const {
  return b_;
}

[[nodiscard]] Layer::ActivationFunctionType Layer::getActivationFunctionType()
    const {
  return activationFunction->getName();
}

[[nodiscard]] ssize_t Layer::getStartDimension() const {
  return A_.cols();
}

[[nodiscard]] ssize_t Layer::getEndDimension() const {
  return A_.rows();
}

Layer& Layer::operator+=(const LayerDelta& delta) {
  assert(A_.cols() == delta.deltaA.cols() && A_.rows() == delta.deltaA.rows());
  assert(b_.rows() == delta.deltaB.rows());
  A_ += delta.deltaA;
  b_ += delta.deltaB;
  return *this;
}

std::istream& operator>>(std::istream& is, Layer& layer) {
  ssize_t rows, cols, vec_size;
  ActivationsFunctions::ActivationFunctionType type;

  is.read(reinterpret_cast<char*>(&type), sizeof(type));

  is.read(reinterpret_cast<char*>(&rows), sizeof(rows));
  is.read(reinterpret_cast<char*>(&cols), sizeof(cols));
  is.read(reinterpret_cast<char*>(&vec_size), sizeof(vec_size));
  assert(cols > 0 && rows > 0);

  layer.A_ = Matrix(rows, cols);
  layer.b_ = Vector(vec_size);

  is.read(reinterpret_cast<char*>(layer.A_.data()),
          sizeof(double) * layer.A_.size());
  is.read(reinterpret_cast<char*>(layer.b_.data()), sizeof(double) * vec_size);
  layer.activationFunction =
      ActivationsFunctions::getActivationFunctionByType(type);
  return is;
}


std::ostream& operator<<(std::ostream& os, const Layer& layer) {
  const auto functionType = layer.getActivationFunctionType();
  os.write(reinterpret_cast<const char*>(&functionType), sizeof(functionType));

  ssize_t rows = layer.A_.rows();
  ssize_t cols = layer.A_.cols();
  size_t vec_size = layer.b_.size();

  os.write(reinterpret_cast<const char*>(&rows), sizeof(rows));
  os.write(reinterpret_cast<const char*>(&cols), sizeof(cols));
  os.write(reinterpret_cast<const char*>(&vec_size), sizeof(vec_size));

  os.write(reinterpret_cast<const char*>(layer.A_.data()),
           sizeof(double) * layer.A_.size());
  os.write(reinterpret_cast<const char*>(layer.b_.data()),
           sizeof(double) * vec_size);
  return os;
}

std::ostream& operator<<(std::ostream& os, const std::vector<Layer>& layers) {
  ssize_t num_layers = layers.size();
  os.write(reinterpret_cast<const char*>(&num_layers), sizeof(num_layers));
  for (auto const& layer : layers) {
    os << layer;
  }
  return os;
}

std::istream& operator>>(std::istream& is, std::vector<Layer>& layers) {
  ssize_t layers_size;
  is.read(reinterpret_cast<char*>(&layers_size), sizeof(layers_size));
  assert(layers_size > 0);

  layers.clear();
  layers.reserve(layers_size);
  for (ssize_t i = 0; i < layers_size; ++i) {
    layers.emplace_back();
    is >> layers[i];
  }
  return is;
}

}  // namespace NN
