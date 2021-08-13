import oneflow as flow

def main():

    rank = flow.distributed.get_rank()

    resnet50_module = flow.nn.Linear(512, 1000)
   
    resnet50_module.to_consistent(placement=flow.placement("cuda", {0: [0, 1]}), sbp=[flow.sbp.broadcast])

    class Resnet50EvalGraph(flow.nn.Graph):
        def __init__(self):
            super().__init__()
            self.resnet50 = resnet50_module
        
        def build(self, input):
            input = input.to("cuda")
            with flow.no_grad():
                logits = self.resnet50(input)
            return logits

    resnet50_eval_graph = Resnet50EvalGraph()

    feat = flow.ones(8,512).to_consistent(placement=flow.placement("cuda", {0: [0, 1]}), sbp=[flow.sbp.split(0)])

    out = resnet50_eval_graph(feat)

    print(out.to_local().numpy())

if __name__ == "__main__":
    main()
