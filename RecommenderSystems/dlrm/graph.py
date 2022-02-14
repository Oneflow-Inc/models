import oneflow as flow


class DLRMValGraph(flow.nn.Graph):
    def __init__(self, wdl_module, use_fp16=False):
        super(DLRMValGraph, self).__init__()
        self.module = wdl_module
        if use_fp16:
            self.config.enable_amp(True)

    def build(self, labels, dense_fields, sparse_fields):
        labels = labels.to("cuda").to(dtype=flow.float32)
        dense_fields = dense_fields.to("cuda")
        sparse_fields = sparse_fields.to("cuda")

        predicts = self.module(dense_fields, sparse_fields)
        return predicts, labels


class DLRMTrainGraph(flow.nn.Graph):
    def __init__(self, wdl_module, bce_loss, optimizer, lr_scheduler=None, grad_scaler=None, use_fp16=False):
        super(DLRMTrainGraph, self).__init__()
        self.module = wdl_module
        self.bce_loss = bce_loss
        self.add_optimizer(optimizer, lr_sch=lr_scheduler)
        self.config.allow_fuse_model_update_ops(True)
        self.config.allow_fuse_add_to_output(True)
        self.config.allow_fuse_cast_scale(True)
        if use_fp16:
            self.config.enable_amp(True)
            self.set_grad_scaler(grad_scaler)

    def build(self, labels, dense_fields, sparse_fields):
        labels = labels.to("cuda").to(dtype=flow.float32)
        dense_fields = dense_fields.to("cuda").to(dtype=flow.float32)
        sparse_fields = sparse_fields.to("cuda").to(dtype=flow.int64)

        logits = self.module(dense_fields, sparse_fields)
        loss = self.bce_loss(logits, labels)
        reduce_loss = flow.mean(loss)
        reduce_loss.backward()
        return reduce_loss


class DLRMValGraphWithDataloader(DLRMValGraph):
    def __init__(self, wdl_module, dataloader, use_fp16=False):
        super(DLRMValGraphWithDataloader, self).__init__(wdl_module, dataloader, use_fp16)
        self.dataloader = dataloader

    def build(self):
        labels, dense_fields, sparse_fields = self.dataloader()
        return super(DLRMValGraphWithDataloader, self).build(labels, dense_fields, sparse_fields)


class DLRMTrainGraphWithDataloader(DLRMTrainGraph):
    def __init__(self, wdl_module, dataloader, bce_loss, optimizer, lr_scheduler=None, grad_scaler=None, use_fp16=False):
        super(DLRMTrainGraphWithDataloader, self).__init__(wdl_module, bce_loss, optimizer, lr_scheduler, grad_scaler, use_fp16)
        self.dataloader = dataloader

    def build(self):
        labels, dense_fields, sparse_fields = self.dataloader()
        return super(DLRMTrainGraphWithDataloader, self).build(labels, dense_fields, sparse_fields)
