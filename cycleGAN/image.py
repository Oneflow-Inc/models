import random
import oneflow.experimental as flow
import numpy as np
import cv2

class ImagePool():
    """This class implements an image buffer that stores previously generated images.

    This buffer enables us to update discriminators using a history of generated images
    rather than the ones produced by the latest generators.
    """

    def __init__(self, pool_size):
        """Initialize the ImagePool class

        Parameters:
            pool_size (int) -- the size of image buffer, if pool_size=0, no buffer will be created
        """
        self.pool_size = pool_size
        if self.pool_size > 0:  # create an empty pool
            self.num_imgs = 0
            self.images = []

    def query(self, image):
        """Return an image from the pool.

        Parameters:
            images: the latest generated images from the generator

        Returns images from the buffer.

        By 50/100, the buffer will return input images.
        By 50/100, the buffer will return images previously stored in the buffer,
        and insert the current images to the buffer.
        """
        if self.pool_size == 0:  # if the buffer size is 0, do nothing
            return image
        return_image = image
        if self.num_imgs < self.pool_size:   # if the buffer is not full; keep inserting current images to the buffer
            self.num_imgs = self.num_imgs + 1
            self.images.append(image)
        else:
            p = random.uniform(0, 1)
            if p > 0.5:  # by 50% chance, the buffer will return a previously stored image, and insert the current image into the buffer
                random_id = random.randint(0, self.pool_size - 1)  # randint is inclusive
                tmp = self.images[random_id]
                self.images[random_id] = image
                return_image = tmp
        return return_image.to("cuda")


def random_crop(image, crop_height, crop_width):

    max_x = image.shape[1] - crop_width
    max_y = image.shape[0] - crop_height

    x = np.random.randint(0, max_x)
    y = np.random.randint(0, max_y)

    crop = image[y: y + crop_height, x: x + crop_width]

    return crop

def load_image2ndarray(image_path, 
                       resize_and_crop = True,
                       load_size = 286,
                       crop_size = 256):
    im = cv2.imread(image_path)

    if resize_and_crop:
        im = cv2.resize(im, (load_size, load_size), interpolation = cv2.INTER_CUBIC)
        im = random_crop(im, crop_size, crop_size)
    else:
        im = cv2.resize(im, (crop_size, crop_size), interpolation = cv2.INTER_CUBIC)
    
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im = np.transpose(im, (2, 0, 1))
    im = ((im.astype(np.float32) / 255.0) - 0.5) / 0.5
    im = np.expand_dims(im, axis=0)
    return np.ascontiguousarray(im, 'float32')

def ndarray2image(im):
    im = np.squeeze(im)
    im = (np.transpose(im, (1, 2, 0)) + 1) / 2.0 * 255.0
    im = cv2.cvtColor(np.float32(im), cv2.COLOR_RGB2BGR)
    return im.astype(np.uint8)