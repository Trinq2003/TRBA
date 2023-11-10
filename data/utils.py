import torch
import cv2
import torchvision.transforms as transforms

def resize_image(image: torch.tensor, min_size=600, max_size=1024):
    # Rescaling Images
    C, H, W = image.shape
    min_size = min_size
    max_size = max_size
    scale1 = min_size / min(H, W)
    scale2 = max_size / max(H, W)
    scale = min(scale1, scale2)
    
    resize = transforms.Resize([int(H*scale), int(W*scale)], antialias=True)
    image = image / 255.
    image = resize(image)
    
    return image*255.

def normalize_tensor_image(image_tensor):
    norm = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    image = norm(image_tensor)
    return image

def padding(image:torch.tensor, min_size=600, max_size=1024):
    _, H, W = image.size()
    top = (min_size - H) // 2
    bottom = (min_size - H) - top
    left = (max_size - W) // 2
    right = (max_size - W) - left
    
    padder = transforms.Pad([left, top, right, bottom])
    padded_img = padder(image)
    return padded_img