from torchvision.transforms import transforms


def create_standard_image_transformer(input_size, phase='train', normalize_method='imagenet'):
    assert phase in ['train', 'valid', 'test'], "`phase` not found, only 'train', 'valid', 'test' supported!"
    normalize = {'imagenet': [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]],
                 '-1+1': [[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]]}
    assert normalize_method in normalize, "`normalize_method` not found, only 'imagenet', '-1+1' supported!"
    if phase == 'train':
        return transforms.Compose([
            transforms.Resize(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(*normalize[normalize_method])])
    else:
        return transforms.Compose([
            transforms.Resize(input_size),
            transforms.ToTensor(),
            transforms.Normalize(*normalize[normalize_method])])

