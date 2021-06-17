#include "oneflow/core/framework/attr_map.h"
#include "oneflow/core/framework/op_arg_util.h"
#include "oneflow/core/framework/op_expr.h"
#include "oneflow/core/framework/op_interpreter.h"
#include "oneflow/core/framework/session_util.h"
#include "oneflow/core/framework/tensor.h"
#include "oneflow/core/framework/tensor_tuple.h"

namespace oneflow {
namespace one {

static Maybe<AutogradInterpreter> GetInterpreter();

template<typename T>
static Maybe<T> Dispatch(const OpExpr& op_expr, const TensorTuple& inputs, const AttrMap& attrs);

template<typename T>
static Maybe<T> Dispatch(const OpExpr& op_expr, const TensorTuple& inputs) {
  return Dispatch<T>(op_expr, inputs, AttrMap{});
}

}  // namespace one
}  // namespace oneflow
