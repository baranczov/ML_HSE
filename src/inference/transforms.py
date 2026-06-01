from torchvision import transforms


def build_val_transform(img_size: int = 224) -> transforms.Compose:
    """Val/inference preprocessing: matches training val pipeline."""
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std  = [0.229, 0.224, 0.225]
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std),
    ])
