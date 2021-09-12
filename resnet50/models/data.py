import numpy as np
import os
import oneflow as flow


def make_data_loader(args, mode, is_consistent=False):
    # TODO(zwx): support synthetic data
    assert mode in ("train", "validation")

    if mode == "train":
        total_batch_size = args.train_global_batch_size
        batch_size = args.train_batch_size
        num_samples = args.samples_per_epoch
    else:
        total_batch_size = args.val_global_batch_size
        batch_size = args.val_batch_size
        num_samples = args.val_samples_per_epoch

    placement = None
    sbp = None

    if is_consistent:
        world_size = flow.env.get_world_size()
        placement = flow.placement("cpu", {0: range(world_size)})
        sbp = flow.sbp.split(0)
        # NOTE(zwx): consistent view, only consider logical batch size
        batch_size = total_batch_size

    ofrecord_data_loader = OFRecordDataLoader(
        ofrecord_dir=args.ofrecord_path,
        ofrecord_part_num=args.ofrecord_part_num,
        dataset_size=num_samples,
        mode=mode,
        batch_size=batch_size,
        total_batch_size=total_batch_size,
        channel_last=args.channels_last,
        placement=placement,
        sbp=sbp,
        use_gpu_decode=args.use_gpu_decode
    )
    return ofrecord_data_loader


class OFRecordDataLoader(flow.nn.Module):
    def __init__(
        self,
        ofrecord_dir="./ofrecord",
        ofrecord_part_num=1,
        dataset_size=9469,
        mode="train",
        batch_size=1,
        total_batch_size=1,
        channel_last=False,
        placement=None,
        sbp=None,
        use_gpu_decode=False
    ):
        super().__init__()

        assert mode in ("train", "validation")

        self.batch_size = batch_size
        self.total_batch_size = total_batch_size
        self.dataset_size = dataset_size
        self.mode = mode

        random_shuffle = True if mode == "train" else False
        shuffle_after_epoch = True if mode == "train" else False

        ofrecord_path = os.path.join(ofrecord_dir, self.mode)

        self.ofrecord_reader = flow.nn.OfrecordReader(
            ofrecord_path,
            batch_size=batch_size,
            data_part_num=ofrecord_part_num,
            part_name_suffix_length=5,
            random_shuffle=random_shuffle,
            shuffle_after_epoch=shuffle_after_epoch,
            placement=placement,
            sbp=sbp,
        )

        self.label_decoder = flow.nn.OfrecordRawDecoder(
            "class/label", shape=tuple(), dtype=flow.int32
        )

        output_layout = "NHWC" if channel_last else "NCHW"
        color_space = "RGB"
        image_height = 224
        image_width = 224
        resize_shorter = 256
        rgb_mean = [123.68, 116.779, 103.939]
        rgb_std = [58.393, 57.12, 57.375]

        self.use_gpu_decode = use_gpu_decode
        if self.mode == "train":
            if self.use_gpu_decode:
                self.bytesdecoder_img = flow.nn.OFRecordBytesDecoder("encoded")
                self.image_decoder = flow.nn.OFRecordImageGpuDecoderRandomCropResize(
                    target_width=image_width, target_height=image_height, num_workers=3
                )
            else:
                self.image_decoder = flow.nn.OFRecordImageDecoderRandomCrop(
                    "encoded", color_space=color_space
                )
                self.resize = flow.nn.image.Resize(target_size=[image_width, image_height])
            self.flip = flow.nn.CoinFlip(
                batch_size=self.batch_size, placement=placement, sbp=sbp
            )
            self.crop_mirror_norm = flow.nn.CropMirrorNormalize(
                color_space=color_space,
                output_layout=output_layout,
                mean=rgb_mean,
                std=rgb_std,
                output_dtype=flow.float,
            )
        else:
            self.image_decoder = flow.nn.OFRecordImageDecoder(
                "encoded", color_space=color_space
            )
            self.resize = flow.nn.image.Resize(
                resize_side="shorter",
                keep_aspect_ratio=True,
                target_size=resize_shorter,
            )
            self.crop_mirror_norm = flow.nn.CropMirrorNormalize(
                color_space=color_space,
                output_layout=output_layout,
                crop_h=image_height,
                crop_w=image_width,
                crop_pos_y=0.5,
                crop_pos_x=0.5,
                mean=rgb_mean,
                std=rgb_std,
                output_dtype=flow.float,
            )

    def __len__(self):
        return self.dataset_size // self.total_batch_size

    def forward(self):
        image = flow.tensor(np.arange(self.batch_size * 3 * 224 * 224).reshape((self.batch_size, 3, 224, 224)).astype(np.float32))
        label = flow.tensor(np.arange(self.batch_size).astype(np.int32))
        return image, label
        if self.mode == "train":
            record = self.ofrecord_reader()
            if self.use_gpu_decode:
                encoded = self.bytesdecoder_img(record)
                image = self.image_decoder(encoded)
            else:
                image_raw_bytes = self.image_decoder(record)
                image = self.resize(image_raw_bytes)[0]

            label = self.label_decoder(record)
            flip_code = self.flip()
            if self.use_gpu_decode:
                flip_code = flip_code.to("cuda")
            image = self.crop_mirror_norm(image, flip_code)
        else:
            record = self.ofrecord_reader()
            image_raw_bytes = self.image_decoder(record)
            label = self.label_decoder(record)
            image = self.resize(image_raw_bytes)[0]
            image = self.crop_mirror_norm(image)

        return image, label
