from functools import partial
import os
import time
import logging

import numpy as np

import oneflow.experimental as flow
import oneflow.typing as tp

import dataset
from trainer import Trainer
from utils import get_config

if __name__ == "__main__":
    flow.enable_eager_execution()

    localtime = time.localtime()
    log_filename = f"/home/wurihui/data/log/{localtime.tm_mon}_{localtime.tm_mday}_{localtime.tm_hour}_{localtime.tm_min}.log"
    logging.basicConfig(level=logging.DEBUG, filemode='a', filename=log_filename)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger('').addHandler(console)
    logging.info(f"log saved to {log_filename}")

    flow.env.init()
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)
    func_config.default_logical_view(flow.scope.consistent_view())
    # func_config.default_placement_scope(flow.scope.placement("gpu", "0:0-1"))

    config = get_config('./funit_animals.yaml')

    num_gpus = config['num_gpus']
    flow.config.gpu_device_num(num_gpus)
    device = flow.device("cuda:0")

    N = int(config['batch_size'])
    C = 3
    H = int(config['image_size'])
    W = int(config['image_size'])

    trainer = Trainer(config, device)
    G = trainer.model.gen
    D = trainer.model.dis

    # @flow.global_function("train", func_config)
    # def Train(
    #     images_content: tp.Numpy.Placeholder((N, C, H, W), dtype=flow.float32), 
    #     labels_content: tp.Numpy.Placeholder((N,), dtype=flow.int32), 
    #     images_style: tp.Numpy.Placeholder((N, C, H, W), dtype=flow.float32), 
    #     labels_style: tp.Numpy.Placeholder((N,), dtype=flow.int32)
    # ):
    #     co_data = images_content, labels_content
    #     cl_data = images_style, labels_style
    #     d_acc = trainer.dis_update(co_data, cl_data, config)
    #     g_acc = trainer.gen_update(co_data, cl_data, config)
    #     return d_acc, g_acc

    # @flow.global_function("train", func_config)
    # def TrainGenerator(
    #     images_content: tp.Numpy.Placeholder((N, C, H, W), dtype=flow.float32), 
    #     images_style: tp.Numpy.Placeholder((N, C, H, W), dtype=flow.float32), 
    #     labels_content: tp.Numpy.Placeholder((N,), dtype=flow.int32), 
    #     labels_style: tp.Numpy.Placeholder((N,), dtype=flow.int32)
    # ):
    #     d_acc = trainer.dis_update(
    #         (images_content, labels_content), 
    #         (images_style, labels_style), 
    #         config
    #     )
    #     content_a = G.ContentEncoder(images_content)
    #     style_a = G.StyleEncoder(images_content)
    #     style_b = G.StyleEncoder(images_style)
    #     style_b = flow.squeeze(style_b, axis=[2, 3])
    #     style_b = G.MLP(style_b)
    #     images_trans = G.Decoder(content_a, style_b)
    #     style_a = flow.squeeze(style_a, axis=[2, 3])
    #     style_a = G.MLP(style_a)
    #     images_recon = G.Decoder(content_a, style_a)

    #     fake_out_trans, fake_features_trans = D.ResDiscriminator(images_trans, labels_style)
    #     real_out_style, real_features_style = D.ResDiscriminator(images_style, labels_style)
    #     fake_out_recon, fake_features_recon = D.ResDiscriminator(images_recon, labels_content)


    # @flow.global_function("train", func_config)
    # def TrainDiscriminator(
    #     images_trans: tp.Numpy.Placeholder((N, C, H, W), dtype=flow.float32), 
    #     images_style: tp.Numpy.Placeholder((N, C, H, W), dtype=flow.float32), 
    #     labels_style: tp.Numpy.Placeholder((N,), dtype=flow.int32)
    # ):
    #     pass

    logging.info("Job function configured")

    augment = partial(
        dataset.augment, 
        random_scale_limit=config['random_scale_limit'], 
        resize_smallest_side=config['resize_smallest_side'], 
        random_crop_h_w=config['image_size']
    )
    train_dataset = dataset.Dataset(os.path.join(config['dataset_dir'], "train"), augment=augment)
    test_dataset = dataset.Dataset(os.path.join(config['dataset_dir'], "val"), augment=augment)
    logging.info("Dataset loaded")

    for epoch in range(config['epoch']):
        data_iter = train_dataset.data_iterator(N)
        iteration = 0

        while True:
            print(f"[iter] {iteration}")
            try:
                images_content, labels_content = next(data_iter)
                images_style, labels_style = next(data_iter)
            except StopIteration:
                break

            # images_trans, loss_G, images_recon = TrainGenerator(
            #     images_content, images_style, 
            #     labels_content, labels_style
            # ).get()

            # loss_D = TrainDiscriminator(
            #     images_trans.numpy(), images_style, labels_style
            # ).get()
            # loss_D, loss_G = Train(images_content, labels_content, images_style, labels_style)

            co_data = flow.Tensor(images_content, device=device), flow.Tensor(labels_content, dtype=flow.int, device=device)
            cl_data = flow.Tensor(images_style, device=device), flow.Tensor(labels_style, dtype=flow.int, device=device)
            d_acc = trainer.dis_update(co_data, cl_data, config)
            g_acc = trainer.gen_update(co_data, cl_data, config)

            if iteration % 20 == 0:
                loss_G_data = f"{g_acc.numpy()[0]}"
                loss_D_data = f"{d_acc.numpy()[0]}"

                logging.info(
                    f"[Epoch {epoch:4} / iter: {iteration:6}] loss_G: {loss_G_data}, loss_D: {loss_D_data}"
                )

            iteration += 1

        filename = (
            f"{config['checkpoints_dir']}/"
            f"epoch_{epoch}_"
            f"Gloss_{loss_G_data}_Dloss_{loss_D_data}"
        )

        flow.checkpoint.save(filename)
        logging.info(f"checkpoint saved to {filename}")

        # test_image_recon = images_recon.numpy()[0, :, :, :]
        # test_image_recon_max = test_image_recon.max()
        # test_image_recon_min = test_image_recon.min()
        # test_image_recon = (test_image_recon - test_image_recon_min) / (test_image_recon_max - test_image_recon_min)
        # test_image_recon = (test_image_recon * 255).astype(np.uint8)
        # test_image_recon = np.transpose(test_image_recon, (1, 2, 0))

        # import cv2
        # save_path = f"/home/wurihui/data/Imgs/{epoch}_{iteration}.png"
        # cv2.imwrite(save_path, test_image_recon)
        # logging.info(f"Image saved to {save_path}")

        train_dataset.shuffle()
