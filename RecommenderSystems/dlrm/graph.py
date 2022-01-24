import oneflow as flow
import os


class DLRMValGraph(flow.nn.Graph):
    def __init__(self, wdl_module, dataloader, slotloader):
        super(DLRMValGraph, self).__init__()
        self.module = wdl_module
        self.dataloader = dataloader
        self.slotloader = slotloader

    def build(self):
        (
            labels,
            dense_fields,
            sparse_fields,
        ) = self.dataloader()
        sparse_slots = self.slotloader(False)
        labels = labels.to("cuda").to(dtype=flow.float32)
        dense_fields = dense_fields.to("cuda")
        sparse_fields = sparse_fields.to("cuda")
        sparse_slots = sparse_slots.to("cuda")

        predicts = self.module(dense_fields, sparse_fields, sparse_slots)
        return predicts, labels


class DLRMTrainGraph(flow.nn.Graph):
    def __init__(self, wdl_module, dataloader, slotloader, bce_loss, optimizer, lr_scheduler=None, grad_scaler=None):
        super(DLRMTrainGraph, self).__init__()
        self.module = wdl_module
        self.dataloader = dataloader
        self.slotloader = slotloader
        self.bce_loss = bce_loss
        self.add_optimizer(optimizer, lr_sch=lr_scheduler)
        if int(os.environ.get("USE_FP16", 0))==1:
            print("use fp16")
            self.config.enable_amp(True)
            self.set_grad_scaler(grad_scaler)


    def build(self):
        (
            labels,
            dense_fields,
            sparse_fields,
        ) = self.dataloader()
        sparse_slots = self.slotloader(True)
        labels = labels.to("cuda").to(dtype=flow.float32)
        dense_fields = dense_fields.to("cuda")
        sparse_fields = sparse_fields.to("cuda")
        sparse_slots = sparse_slots.to("cuda")

        logits = self.module(dense_fields, sparse_fields, sparse_slots)
        loss = self.bce_loss(logits, labels)
        reduce_loss = flow.mean(loss)
        reduce_loss.backward()
        return reduce_loss
