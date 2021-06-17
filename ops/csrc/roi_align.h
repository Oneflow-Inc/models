#include "oneflow/core/framework/op_expr.h"

namespace oneflow {
namespace op_expr_helper {

Maybe<one::UserOpExpr> RoIAlignGradOp(const int32_t& pooled_h, const int32_t& pooled_w,
                                      const float& spatial_scale, const int32_t& sampling_ratio,
                                      const bool& aligned);
Maybe<one::UserOpExpr> RoIAlignGradOp(const int32_t& pooled_h, const int32_t& pooled_w,
                                      const float& spatial_scale, const int32_t& sampling_ratio,
                                      const bool& aligned, const std::string& name);

}  // namespace op_expr_helper
}  // namespace oneflow
