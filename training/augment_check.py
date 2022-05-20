import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image

from training.augment import AugmentPipe


def run_augment_check(image_in, image_out):
    aug = AugmentPipe(
        xflip=0,
        rotate90=0,
        xint=0,
        xint_max=0,
        scale=0,
        rotate=0,
        aniso=0,
        xfrac=0,
        brightness=0,
        contrast=0,
        lumaflip=0,
        hue=0,
        saturation=0,
    )

    image = Image.open(image_in)
    transform = transforms.ToTensor()
    img_tensor: torch.Tensor = transform(image)
    img_tensor_list: torch.Tensor = img_tensor[None, :, :, :]

    augment_images = []
    for i in range(16):
        augment_image = aug.forward(img_tensor_list)[0]
        augment_images.append(augment_image)

    torchvision.utils.save_image(augment_images, image_out, nrow=4)


if __name__ == "__main__":
    run_augment_check(image_in='../augment_check/lena.jpg', image_out='../augment_check/lena_result.jpg')
