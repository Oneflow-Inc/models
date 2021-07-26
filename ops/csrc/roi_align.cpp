#include "oneflow/core/framework/framework.h"
#include "oneflow/core/framework/op_expr_grad_function.h"
#include "oneflow/core/framework/op_expr.h"
#include "oneflow/core/framework/op_expr_helper.h"
#include "oneflow/core/framework/op_builder.h"
#include "oneflow/core/framework/user_op_conf_trait.h"
#include "oneflow/core/framework/id_util.h"
// #include "oneflow/core/framework/op_dispatch.h"
#include "roi_align.h"

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
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      const Shape* x_shape = ctx->Shape4ArgNameAndIndex("x", 0);
      const Shape* rois_shape = ctx->Shape4ArgNameAndIndex("rois", 0);
      Shape* y_shape = ctx->Shape4ArgNameAndIndex("y", 0);
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
                            const user_op::UserOpConfWrapper&) -> Maybe<void> {
      user_op::InputArgModifier* roi_modifier = GetInputArgModifierFn("rois", 0);
      CHECK(roi_modifier != nullptr);
      roi_modifier->set_requires_grad(false);
      user_op::InputArgModifier* feat_modifier = GetInputArgModifierFn("x", 0);
      CHECK(feat_modifier != nullptr);
      feat_modifier->set_requires_grad(true);
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      ctx->NewBuilder()
          .Split(user_op::OpArg("x", 0), 0)
          .Split(user_op::OpArg("rois", 0), 0)
          .Split(user_op::OpArg("y", 0), 0)
          .Build();
      return Maybe<void>::Ok();
    })
    .SetDataTypeInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      *ctx->Dtype4ArgNameAndIndex("y", 0) = *ctx->Dtype4ArgNameAndIndex("x", 0);
      return Maybe<void>::Ok();
    });

/*namespace op_expr_helper {

Maybe<one::UserOpExpr> RoIAlignGradOp(const int32_t& pooled_h, const int32_t& pooled_w,
                                      const float& spatial_scale, const int32_t& sampling_ratio,
                                      const bool& aligned) {
  return RoIAlignGradOp(pooled_h, pooled_w, spatial_scale, sampling_ratio, aligned,
                        *CHECK_JUST(UniqueStr("roi_align_grad")));
}
Maybe<one::UserOpExpr> RoIAlignGradOp(const int32_t& pooled_h, const int32_t& pooled_w,
                                      const float& spatial_scale, const int32_t& sampling_ratio,
                                      const bool& aligned, const std::string& name) {
  return one::OpBuilder("roi_align_grad", name)
      .Input("dy")
      .Input("x_like")
      .Input("rois")
      .Output("dx")
      .Attr<int32_t>("pooled_h", pooled_h)
      .Attr<int32_t>("pooled_w", pooled_w)
      .Attr<float>("spatial_scale", spatial_scale)
      .Attr<int32_t>("sampling_ratio", sampling_ratio)
      .Attr<bool>("aligned", aligned)
      .Build();
}

}  // namespace op_expr_helper

namespace one {
namespace {

struct RoIAlignInterpState : public OpExprInterpState {
  bool requires_grad = true;
};

class RoIAlignGrad : public OpExprGradFunction<RoIAlignInterpState> {
 public:
  Maybe<void> Init(const OpExpr& op) override;
  Maybe<void> Capture(RoIAlignInterpState* ctx, const TensorTuple& inputs,
                      const TensorTuple& outputs, const AttrMap& attrs) const override;
  Maybe<void> Apply(const RoIAlignInterpState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override;

 private:
  std::shared_ptr<user_op::UserOpConfTrait> op_trait_;
  int32_t pooled_h_;
  int32_t pooled_w_;
  float spatial_scale_;
  int32_t sampling_ratio_;
  bool aligned_;

  std::shared_ptr<OpExpr> input_grad_op_;
};

Maybe<void> RoIAlignGrad::Init(const OpExpr& op) {
  const auto* fw_op_expr = dynamic_cast<const UserOpExpr*>(&op);
  CHECK_NOTNULL_OR_RETURN(fw_op_expr);
  const std::string& op_name = fw_op_expr->op_name();
  op_trait_ = std::make_shared<user_op::UserOpConfTrait>(op_name, fw_op_expr->proto());

  pooled_h_ = JUST(op_trait_->GetAttr<int32_t>("pooled_h"));
  pooled_w_ = JUST(op_trait_->GetAttr<int32_t>("pooled_w"));
  spatial_scale_ = JUST(op_trait_->GetAttr<float>("spatial_scale"));
  sampling_ratio_ = JUST(op_trait_->GetAttr<int32_t>("sampling_ratio"));
  aligned_ = JUST(op_trait_->GetAttr<bool>("aligned"));
  input_grad_op_ = JUST(op_expr_helper::RoIAlignGradOp(pooled_h_, pooled_w_, spatial_scale_,
                                                       sampling_ratio_, aligned_));

  return Maybe<void>::Ok();
}

Maybe<void> RoIAlignGrad::Capture(RoIAlignInterpState* ctx, const TensorTuple& inputs,
                                  const TensorTuple& outputs, const AttrMap& attrs) const {
  ctx->requires_grad = inputs.at(0)->requires_grad();
  if (!ctx->requires_grad) { return Maybe<void>::Ok(); }

  ctx->SaveTensorForBackward(inputs.at(0));
  ctx->SaveTensorForBackward(inputs.at(1));
  return Maybe<void>::Ok();
}

Maybe<void> RoIAlignGrad::Apply(const RoIAlignInterpState* ctx, const TensorTuple& out_grads,
                                TensorTuple* in_grads) const {
  if (!ctx->requires_grad) { return Maybe<void>::Ok(); }

  const auto& saved_tensors = ctx->SavedTensors();
  in_grads->at(0) = JUST(Dispatch<Tensor>(*/
//      *input_grad_op_, {out_grads.at(0), saved_tensors.at(0), saved_tensors.at(1),
//      /*attrs=*/{}}));
//  return Maybe<void>::Ok();
//}

//}  // namespace

// REGISTER_OP_EXPR_GRAD_FUNCTION("roi_align", RoIAlignGrad);

//}  // namespace one
}  // namespace oneflow
