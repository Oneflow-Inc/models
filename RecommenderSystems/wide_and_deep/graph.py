import oneflow as flow


class WideAndDeepValGraph(flow.nn.Graph):
    def __init__(self, wdl_module, dataloader):
        super(WideAndDeepValGraph, self).__init__()
        self.module = wdl_module
        self.dataloader = dataloader

    def build(self):
        (
            labels,
            dense_fields,
            wide_sparse_fields,
            deep_sparse_fields,
        ) = self.dataloader()
        labels = labels.to("cuda").to(dtype=flow.float32)
        dense_fields = dense_fields.to("cuda")
        wide_sparse_fields = wide_sparse_fields.to("cuda")
        deep_sparse_fields = deep_sparse_fields.to("cuda")

        predicts = self.module(dense_fields, wide_sparse_fields, deep_sparse_fields)
        return predicts, labels


class WideAndDeepTrainGraph(flow.nn.Graph):
    def __init__(self, wdl_module, dataloader, bce_loss, optimizer, sparse_opt):
        super(WideAndDeepTrainGraph, self).__init__()
        self.module = wdl_module
        self.dataloader = dataloader
        self.bce_loss = bce_loss
        self.add_optimizer(optimizer)
        self.add_optimizer(sparse_opt, is_sparse=True)

    def build(self):
        (
            labels,
            dense_fields,
            wide_sparse_fields,
            deep_sparse_fields,
        ) = self.dataloader()
        labels = labels.to("cuda").to(dtype=flow.float32)
        dense_fields = dense_fields.to("cuda")
        wide_sparse_fields = wide_sparse_fields.to("cuda")
        deep_sparse_fields = deep_sparse_fields.to("cuda")

        logits = self.module(dense_fields, wide_sparse_fields, deep_sparse_fields)
        loss = self.bce_loss(logits, labels)
        reduce_loss = flow.mean(loss)
        reduce_loss.backward()
        return reduce_loss
