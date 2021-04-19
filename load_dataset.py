import os
import numpy as np
from numpy import loadtxt
from PIL import Image
import torchvision.transforms as T
from torchvision import transforms
from torch.utils.data import Dataset

# Soan's code
class ImageDataset(Dataset):
    def __init__(self, txt_files, img_size=(320, 320),
                 transform=None, n_cutoff_imgs=None):
        """
        :param root_dir: root directory to the dataset folder, e.g ../02-Data/UOW-HSI/
        :param txt_files: text files contain filenames of image and its annotated image.
        :param img_size (tuple): H and W of the image to be resized to
        :param transform (torchvision's transform list): transformation for data
        :param n_cutoff_imgs: maximum number of used images in each text file
        """
        super(ImageDataset, self).__init__()
        self.txt_files = txt_files
        self.img_size = img_size
        self.transform = transform
        if transform is not None:
            assert any(isinstance(x, transforms.ToTensor)
                       for x in transform.transforms) == False, \
                "No need ToTensor operation because it is done implicitly already."
        self.to_tensor = T.ToTensor()

        if (not isinstance(n_cutoff_imgs, int)) or (n_cutoff_imgs <= 0):
            n_cutoff_imgs = None

        # Get filename of the training images stored in txt_files
        if isinstance(txt_files, str):
            txt_files = [txt_files]
        self.training_imgs = [item for file in txt_files
                              for item in list(loadtxt(file, dtype=np.str,
                                                       delimiter=', '))[:n_cutoff_imgs]]

    def __len__(self):
        """
        :return: the size of the dataset, i.e. the number of images
        """
        return len(self.training_imgs)

    def __getitem__(self, index):
        """
        Read the an image from the training images
        :param index: index of the image in the list of training images
        :return: image and segmentation ground-truth images
                + img: input hsi image of size (n_bands, H, W)
                + groundtruth: ground-truth segmentation image of size (H, W)                
        """
        # Set the ground truth and input files
        img_file = os.path.join(self.training_imgs[index][0])
        gt_file = os.path.join(self.training_imgs[index][1])

        # Read the images
        img = Image.open(img_file)
        groundtruth = Image.open(gt_file)

        if self.transform:
            img = self.transform(img)
            groundtruth = self.transform(groundtruth)

        # Convert the images into Pytorch tensors
        img = self.to_tensor(img)                # of size (in_channels, H, W)
        groundtruth = self.to_tensor(groundtruth) * 255  # as to_tensor() return tensor in the range [0.0, 1.0]
        groundtruth = groundtruth.squeeze().long()       # of size (H, W)

        return {'image' :img, 'label' : groundtruth}
