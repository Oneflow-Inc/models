#include "op_dispatch.h"

#include "oneflow/core/framework/device.h"
#include "oneflow/core/framework/dtype.h"
#include "oneflow/core/framework/py_distribute.h"
#include "oneflow/core/framework/tensor_impl.h"

namespace oneflow {
namespace one {

namespace {

std::shared_ptr<AutogradInterpreter> BuildEagerInterpreter(const bool& is_mirrored) {
  std::shared_ptr<OpExprInterpreter> internal;
  if (is_mirrored) {
    internal = std::make_shared<EagerMirroredInterpreter>();
  } else {
    internal = std::make_shared<EagerConsistentInterpreter>();
  }
  return std::make_shared<AutogradInterpreter>(internal);
}

std::shared_ptr<AutogradInterpreter> BuildLazyInterpreter() {
  auto internal = std::make_shared<LazyInterpreter>();
  return std::make_shared<AutogradInterpreter>(internal);
}

}  // namespace

/*static*/ Maybe<AutogradInterpreter> GetInterpreter() {
  static const auto& g_lazy_interpreter = BuildLazyInterpreter();
  static const auto& g_eager_consistent_interpreter = BuildEagerInterpreter(/*is_mirrored=*/false);
  static const auto& g_eager_mirrored_interpreter = BuildEagerInterpreter(/*is_mirrored=*/true);
  if (EagerExecutionEnabled()) {
    const auto& session = JUST(GetDefaultSession());
    bool is_mirrored_strategy_enabled = session->is_mirrored_strategy_enabled_stack()->empty()
                                        || JUST(session->IsMirroredStrategyEnabled());
    if (is_mirrored_strategy_enabled) {
      return g_eager_mirrored_interpreter;
    } else {
      return g_eager_consistent_interpreter;
    }
  }
  return g_lazy_interpreter;
}

template<>
/*static*/ Maybe<TensorTuple> Dispatch<TensorTuple>(const OpExpr& op_expr,
                                                    const TensorTuple& inputs,
                                                    const AttrMap& attrs) {
  auto outputs = std::make_shared<TensorTuple>(op_expr.output_size());
  JUST(JUST(GetInterpreter())->Apply(op_expr, inputs, outputs.get(), attrs));
  return outputs;
}

template<>
/*static*/ Maybe<Tensor> Dispatch<Tensor>(const OpExpr& op_expr, const TensorTuple& inputs,
                                          const AttrMap& attrs) {
  return JUST(Dispatch<TensorTuple>(op_expr, inputs, attrs))->at(0);
}

}  // namespace one
}  // namespace oneflow
