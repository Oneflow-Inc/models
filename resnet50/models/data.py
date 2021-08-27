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

        if self.mode == "train":
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
        if self.mode == "train":
            record = self.ofrecord_reader()
            image_raw_bytes = self.image_decoder(record)
            label = self.label_decoder(record)
            image = self.resize(image_raw_bytes)[0]
            flip_code = self.flip()
            image = self.crop_mirror_norm(image, flip_code)
        else:
            record = self.ofrecord_reader()
            image_raw_bytes = self.image_decoder(record)
            label = self.label_decoder(record)
            image = self.resize(image_raw_bytes)[0]
            image = self.crop_mirror_norm(image)

        return image, label
