# PNN
[PNN](https://arxiv.org/abs/1703.04247) is a Factorization-Machine based Neural Network for CTR prediction. Its model structure is as follows. Based on this structure, this project uses OneFlow distributed deep learning framework to realize training the model in graph mode on the Criteo data set.
<img width="539" alt="Screen Shot 2022-04-01 at 4 45 22 PM" src="https://user-images.githubusercontent.com/46690197/161228714-ae9410bb-56db-46b0-8f0b-cb8becb6ee03.png">

## Directory description

## Arguments description
|Argument Name|Argument Explanation|Default Value|
|-----|---|------|
|data_dir|the data file directory|*Required Argument*|
|persistent_path|path for OneEmbeddig persistent kv store|*Required Argument*|
|table_size_array|table size array for sparse fields|*Required Argument*|
|store_type|OneEmbeddig persistent kv store type: `device_mem`, `cached_host_mem` or `cached_ssd` |cached_ssd|
|cache_memory_budget_mb|size of cache memory budget on each device in megabytes when `store_type` is `cached_host_mem` or `cached_ssd`|8192|
|embedding_vec_size|embedding vector dimention size|128|
|bottom_mlp|bottom MLPs hidden units number|512,256,128|
|top_mlp|top MLPs hidden units number|1024,1024,512,256|
|disable_interaction_padding|disable interaction output padding or not|False|
|interaction_itself|interaction itself or not|False|
|disable_fusedmlp|disable fused MLP or not|False|
|train_batch_size|training batch size|55296|
|train_batches|number of minibatch training interations|75000|
|learning_rate|basic learning rate for training|24|
|warmup_batches|learning rate warmup batches|2750|
|decay_start|learning rate decay start iteration|49315|
|decay_batches|number of learning rate decay iterations|27772|
|loss_print_interval|training loss print interval|1000|
|eval_interval|evaluation interval|10000|
|eval_batches|number of evaluation batches|1612|
|eval_batch_size|evaluation batch size|55296|
|model_load_dir|model loading directory|None|
|model_save_dir|model saving directory|None|
|save_model_after_each_eval|save model or not after each evaluation|False|
|save_initial_model|save initial model parameters or not|False|
|amp|enable Automatic Mixed Precision(AMP) training|False|
|loss_scale_policy|loss scale policy for AMP training: `static` or `dynamic`|static|

## Getting Started

