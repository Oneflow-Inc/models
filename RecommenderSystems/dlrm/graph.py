import oneflow as flow


class DLRMValGraph(flow.nn.Graph):
    def __init__(self, wdl_module, dataloader):
        super(DLRMValGraph, self).__init__()
        self.module = wdl_module
        self.dataloader = dataloader

    def build(self):
        (
            labels,
            dense_fields,
            sparse_fields,
        ) = self.dataloader()
        labels = labels.to("cuda").to(dtype=flow.float32)
        dense_fields = dense_fields.to("cuda")
        sparse_fields = sparse_fields.to("cuda")

        predicts = self.module(dense_fields, sparse_fields)
        return predicts, labels


class DLRMTrainGraph(flow.nn.Graph):
    def __init__(self, wdl_module, dataloader, bce_loss, optimizer, lr_scheduler=None):
        super(DLRMTrainGraph, self).__init__()
        self.module = wdl_module
        self.dataloader = dataloader
        self.bce_loss = bce_loss
        self.add_optimizer(optimizer, lr_sch=lr_scheduler)

    def build(self):
        (
            labels,
            dense_fields,
            sparse_fields,
        ) = self.dataloader()
        labels = labels.to("cuda").to(dtype=flow.float32)
        dense_fields = dense_fields.to("cuda")
        sparse_fields = sparse_fields.to("cuda")

        logits = self.module(dense_fields, sparse_fields)
        loss = self.bce_loss(logits, labels)
        reduce_loss = flow.mean(loss)
        reduce_loss.backward()
        return reduce_loss
