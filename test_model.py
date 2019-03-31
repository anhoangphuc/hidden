from model.hidden import HiDDen
from options import TrainingOptions, HiDDenConfiguration
import hidden_utils

import torch
import torchvision
import torch.nn.functional as F
import torchvision.transforms.functional as TF

import argparse
import numpy as np
from PIL import Image


def crop_image(img, height, width):
    '''Random crop image to specific size, in order to feed network'''
    assert img.shape[0] >= height
    assert img.shape[1] >= width

    if img.shape[0] == height and img.shape[1] == width:
        return img

    x = np.random.randint(0, img.shape[1] - width)
    y = np.random.randint(0, img.shape[0] - height)
    img = img[y: y+height, x:x+width]
    return img

def show_image(image, name):
    image = (image + 1) / 2
    image.cpu()
    torchvision.utils.save_image(image, name, normalize=False)

def main():
    parser = argparse.ArgumentParser(description='Test trainied model')
    parser.add_argument('--source-image', '-s', required=True, type=str)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_options, hidden_config = hidden_utils.load_options('demo')
    checkpoint = torch.load('demo/cp299.pyt', map_location=device)

    hidden_model = HiDDen(hidden_config, device)
    hidden_utils.model_from_checkpoint(hidden_model, checkpoint)

    image_pil = Image.open(args.source_image)
    image = crop_image(np.array(image_pil), hidden_config.img_height, hidden_config.img_width)
    image_tensor = TF.to_tensor(image).to(device)
    image_tensor = image_tensor * 2 - 1
    image_tensor.unsqueeze_(0)
    message = torch.Tensor(
        np.random.choice([0, 1],
                         (image_tensor.shape[0], hidden_config.message_length))).to(device)

    ##################################################
    losses, (encoded_images, decoded_message) = hidden_model.validate_on_batch([image_tensor, message])
    print(message)
    decoded_rounded = decoded_message.detach().cpu().numpy().round().clip(0, 1)
    message_detachted = message.detach().cpu().numpy()

    print(message_detachted)
    print(decoded_rounded)
    print(np.sum(np.abs(decoded_rounded - message_detachted)))

    print('-' * 100)
    show_image(encoded_images, 'encode_image.jpg')
    show_image(image_tensor, 'original.jpg')

if __name__ == '__main__':
    main()
