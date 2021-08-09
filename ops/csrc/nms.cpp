#include "oneflow/core/framework/framework.h"

namespace oneflow {

REGISTER_USER_OP("nms")
    .Input("in")
    .Output("out")
    .Attr<float>("iou_threshold")
    .Attr<int32_t>("keep_n")
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      Shape* in_shape = ctx->Shape4ArgNameAndIndex("in", 0);
      Shape* out_shape = ctx->Shape4ArgNameAndIndex("out", 0);
      *out_shape = Shape({in_shape->At(0)});
      return Maybe<void>::Ok();
    })
    .SetDataTypeInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      *ctx->Dtype4ArgNameAndIndex("out", 0) = DataType::kInt8;
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn(user_op::GetSbpFnUtil::DefaultBroadcastToBroadcast);
;

}  // namespace oneflow
