from image import *


class listDataset(object):
    def __init__(
        self,
        root,
        shape=None,
        shuffle=True,
        transform=None,
        train=False,
        seen=0,
        batch_size=1,
        num_workers=4,
    ):
        if train:
            root = root * 4
        random.shuffle(root)
        self.nSamples = len(root)
        self.lines = root
        self.transform = transform
        self.train = train
        self.shape = shape
        self.seen = seen
        self.batch_size = batch_size
        self.num_workers = num_workers

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), "index range error"
        img_path = self.lines[index]
        img, target = load_data(img_path, self.train)
        if self.transform is not None:
            self.transform.randomize_parameters()
            img = self.transform(img)
            img = np.asarray(img).astype(np.float32)
        return img, target
