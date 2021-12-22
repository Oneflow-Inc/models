from roberta import (
    RobertaTransform, 
    RobertaForMaskedLMTransform,
    RobertaForSequenceClassificationTransform
)

def transform_roberta_only():

    pretrained_models = ["roberta-base", "roberta-large", "roberta-large-mnli", \
                            "roberta-base-openai-detector", "roberta-large-openai-detector",\
                            "distilroberta-base"]
    test_args = {"bs": 8, "seq_len": 16}

    for pretrained in pretrained_models:
        trans_class = RobertaTransform(
            pretrained_model=pretrained,
            save_dir="/remote-home/share/shxing/robertaonly_pretrain_oneflow",
            model_dir=pretrained,
        )
        trans_class.run(test_args)

def transform_RobertaForMaskedLM():

    pretrained_models = ["roberta-base", "roberta-large", "distilroberta-base"]
    test_args = {"bs": 8, "seq_len": 16}

    for pretrained in pretrained_models:
        trans_class = RobertaForMaskedLMTransform(
            pretrained_model=pretrained,
            save_dir="/remote-home/share/shxing/roberta_pretrain_oneflow",
            model_dir=pretrained,
        )
        trans_class.run(test_args)

def transform_RobertaForSequenceClassification():

    pretrained_models = ["roberta-base-openai-detector", "roberta-large-openai-detector",\
                            "roberta-large-mnli"]
    test_args = {"bs": 8, "seq_len": 16}

    for pretrained in pretrained_models:
        trans_class = RobertaForSequenceClassificationTransform(
            pretrained_model=pretrained,
            save_dir="/remote-home/share/shxing/roberta_pretrain_oneflow",
            model_dir=pretrained,
        )
        trans_class.run(test_args)

if __name__ == '__main__':

    # transform_roberta_only()
    # transform_RobertaForMaskedLM()
    transform_RobertaForSequenceClassification()
