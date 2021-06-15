#include "oneflow/core/framework/framework.h"

namespace oneflow {

REGISTER_USER_OP("roi_align")
    .Input("x")
    .Input("rois")
    .Output("y")
    .Attr<int32_t>("pooled_h")
    .Attr<int32_t>("pooled_w")
    .Attr<float>("spatial_scale")
    .Attr<int32_t>("sampling_ratio")
    .Attr<bool>("aligned")
    .SetTensorDescInferFn([](user_op::InferContext *ctx) -> Maybe<void> {
      const Shape *x_shape = ctx->Shape4ArgNameAndIndex("x", 0);
      const Shape *rois_shape = ctx->Shape4ArgNameAndIndex("rois", 0);
      Shape *y_shape = ctx->Shape4ArgNameAndIndex("y", 0);
      const int32_t pooled_h = ctx->Attr<int32_t>("pooled_h");
      const int32_t pooled_w = ctx->Attr<int32_t>("pooled_w");
      // x: feature map (N, C, H, W)
      CHECK_EQ(x_shape->NumAxes(), 4);
      // rois: (R, 5)
      CHECK_EQ(rois_shape->NumAxes(), 2);
      CHECK_EQ(rois_shape->At(1), 5);
      // y: (R, C, pool_h, pool_w)
      *y_shape = Shape({rois_shape->At(0), x_shape->At(1), pooled_h, pooled_w});
      return Maybe<void>::Ok();
    })
    .SetInputArgModifyFn([](user_op::GetInputArgModifier GetInputArgModifierFn,
                            const user_op::UserOpConfWrapper &) {
      user_op::InputArgModifier *roi_modifier = GetInputArgModifierFn("rois", 0);
      CHECK(roi_modifier != nullptr);
      roi_modifier->set_requires_grad(false);
      user_op::InputArgModifier *feat_modifier = GetInputArgModifierFn("x", 0);
      CHECK(feat_modifier != nullptr);
      feat_modifier->set_requires_grad(true);
    })
    .SetGetSbpFn([](user_op::SbpContext *ctx) -> Maybe<void> {
      ctx->NewBuilder()
          .Split(user_op::OpArg("x", 0), 0)
          .Split(user_op::OpArg("rois", 0), 0)
          .Split(user_op::OpArg("y", 0), 0)
          .Build();
      return Maybe<void>::Ok();
    })
    .SetDataTypeInferFn([](user_op::InferContext *ctx) -> Maybe<void> {
      *ctx->Dtype4ArgNameAndIndex("y", 0) = *ctx->Dtype4ArgNameAndIndex("x", 0);
      return Maybe<void>::Ok();
    });

REGISTER_USER_OP("roi_align_grad")
    .Input("dy")
    .Input("x_like")
    .Input("rois")
    .Output("dx")
    .Attr<int32_t>("pooled_h")
    .Attr<int32_t>("pooled_w")
    .Attr<float>("spatial_scale")
    .Attr<int32_t>("sampling_ratio")
    .Attr<bool>("aligned")
    .SetTensorDescInferFn([](user_op::InferContext *ctx) -> Maybe<void> {
      const Shape *dy_shape = ctx->Shape4ArgNameAndIndex("dy", 0);
      const Shape *x_like_shape = ctx->Shape4ArgNameAndIndex("x_like", 0);
      const Shape *rois_shape = ctx->Shape4ArgNameAndIndex("rois", 0);
      const int32_t pooled_h = ctx->Attr<int32_t>("pooled_h");
      const int32_t pooled_w = ctx->Attr<int32_t>("pooled_w");
      // x: feature map (N, C, H, W)
      CHECK_EQ_OR_RETURN(x_like_shape->NumAxes(), 4);
      // rois: (R, 5)
      CHECK_EQ_OR_RETURN(rois_shape->NumAxes(), 2);
      CHECK_EQ_OR_RETURN(rois_shape->At(1), 5);
      // y: (R, C, pool_h, pool_w)
      const Shape &y_shape = Shape({rois_shape->At(0), x_like_shape->At(1), pooled_h, pooled_w});
      CHECK_EQ_OR_RETURN(y_shape, *dy_shape);
      *(ctx->Shape4ArgNameAndIndex("dx", 0)) = *x_like_shape;
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext *ctx) -> Maybe<void> {
      ctx->NewBuilder()
          .Split(user_op::OpArg("dy", 0), 0)
          .Split(user_op::OpArg("x_like", 0), 0)
          .Split(user_op::OpArg("rois", 0), 0)
          .Split(user_op::OpArg("dx", 0), 0)
          .Build();
      return Maybe<void>::Ok();
    })
    .SetDataTypeInferFn([](user_op::InferContext *ctx) -> Maybe<void> {
      const DataType *dy_like_dtype = ctx->Dtype4ArgNameAndIndex("dy", 0);
      const DataType *x_like_dtype = ctx->Dtype4ArgNameAndIndex("x_like", 0);
      CHECK_EQ_OR_RETURN(*dy_like_dtype, *x_like_dtype);
      *ctx->Dtype4ArgNameAndIndex("dx", 0) = *x_like_dtype;
    });

REGISTER_USER_OP_GRAD("roi_align")
    .SetGenBackwardOpConfFn([](const user_op::UserOpWrapper &op, user_op::AddOpFn AddOp) {
      if (op.NeedGenGradTensor4OpInput("x", 0)) {
        user_op::UserOpConfWrapperBuilder builder(op.op_name() + "_grad");
        user_op::UserOpConfWrapper grad_op =
            builder.Op("roi_align_grad")
                .Input("dy", op.GetGradTensorWithOpOutput("y", 0))
                .Input("x_like", op.input("x", 0))
                .Input("rois", op.input("rois", 0))
                .Attr("pooled_h", op.attr<int32_t>("pooled_h"))
                .Attr("pooled_w", op.attr<int32_t>("pooled_w"))
                .Attr("spatial_scale", op.attr<float>("spatial_scale"))
                .Attr("sampling_ratio", op.attr<int32_t>("sampling_ratio"))
                .Attr("aligned", op.attr<bool>("aligned"))
                .Output("dx")
                .Build();
        op.BindGradTensorWithOpInput(grad_op.output("dx", 0), "x", 0);
        AddOp(grad_op);
      }
    });

}  // namespace oneflow
