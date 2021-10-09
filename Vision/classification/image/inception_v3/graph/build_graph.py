import oneflow as flow


def build_train_graph(model, criterion, data_loader, optimizer, lr_scheduler=None, *args, **kwargs):
    return TrainGraph(model, criterion, data_loader, optimizer, lr_scheduler, *args, **kwargs)

def build_eval_graph(model, data_loader):
    return EvalGraph(model, data_loader)


class TrainGraph(flow.nn.Graph):
    def __init__(self, model, criterion, data_loader, optimizer, lr_scheduler=None, return_pred_and_label=True):
        super().__init__()
        self.return_pred_and_label = return_pred_and_label

        self.config.allow_fuse_add_to_output(True)
        self.config.allow_fuse_model_update_ops(True)

        self.model = model
        self.criterion = criterion
        self.data_loader = data_loader
        self.add_optimizer(optimizer, lr_sch=lr_scheduler)

    def build(self):
        image, label = self.data_loader()
        image = image.to("cuda")
        label = label.to("cuda")
        logits, aux = self.model(image)
        loss = self.criterion(logits, label) + self.criterion(aux, label)
        if self.return_pred_and_label:
            pred = logits.softmax()
        else:
            pred = None
            label = None
        loss.backward()
        return loss, pred, label


class EvalGraph(flow.nn.Graph):
    def __init__(self, model, data_loader):
        super().__init__()

        # self.config.allow_fuse_add_to_output(True)

        self.data_loader = data_loader
        self.model = model
    
    def build(self):
        image, label = self.data_loader()
        image = image.to("cuda")
        label = label.to("cuda")
        logits, aux = self.model(image)
        pred = logits.softmax()
        return pred, label