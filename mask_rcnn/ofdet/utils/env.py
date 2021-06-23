import oneflow as flow
import ofdet


def setup_env(args, cfg):
    flow.env.ctrl_port(args.ctrl_port)
    # flow.config.load_library(ofdet.lib_path())
    # flow.config.gpu_device_num(cfg.ENV.NUM_GPUS)
    flow.config.gpu_device_num(1)


def get_flow_dtype(dtype_str):
    if dtype_str == "float32":
        return flow.float32
    else:
        raise NotImplementedError


def get_train_config(cfg):
    train_config = flow.FunctionConfig()
    train_config.default_logical_view(flow.scope.consistent_view())
    train_config.enable_inplace(cfg.ENV.ENABLE_INPLACE)
    train_config.default_data_type(get_flow_dtype(cfg.DTYPE))

    train_config.cudnn_buf_limit_mbyte(cfg.ENV.CUDNN_BUFFER_SIZE_LIMIT)
    train_config.cudnn_conv_heuristic_search_algo(
        cfg.ENV.CUDNN_CONV_HEURISTIC_SEARCH_ALGO
    )
    train_config.cudnn_conv_use_deterministic_algo_only(
        cfg.ENV.CUDNN_CONV_USE_DETERMINISTIC_ALGO_ONLY
    )

    return train_config


def get_test_config(cfg):
    flow.enable_eager_execution(False)
    test_config = flow.FunctionConfig()
    test_config.default_logical_view(flow.scope.consistent_view())
    test_config.default_data_type(get_flow_dtype(cfg.DTYPE))
    test_config.cudnn_buf_limit_mbyte(cfg.ENV.CUDNN_BUFFER_SIZE_LIMIT)
    test_config.enable_cudnn(True)
    test_config.cudnn_conv_heuristic_search_algo(True)
    test_config.enable_cudnn_fused_normalization_add_relu(True)
    test_config.enable_fuse_add_to_output(True)
    test_config.enable_fuse_cast_scale(True)

    return test_config
