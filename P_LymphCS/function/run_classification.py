import argparse
import os
import logging
import torch
import torch.nn as nn
import torch.optim
import torch.utils as utils
import torch.utils.data

from P_LymphCS.core import create_optimizer, create_lr_scheduler, create_losses, create_model, create_standard_image_transformer
from P_LymphCS.custom import create_classification_dataset
from P_LymphCS import models


logger = logging.root
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("[%(asctime)s - %(filename)s:%(lineno)4s]\t%(levelname)s\t%(message)s",
                                       '%Y-%m-%d %H:%M:%S'))
logger.handlers = [handler]


def train_model(model, device, dataloaders, criterion, optimizer, lr_scheduler, num_epochs=25,
                iters_start=0, save_dir='./', save_per_epoch=False, **kwargs):
    iters_done = 0
    os.makedirs(os.path.join(save_dir, 'train'), exist_ok=True)
    train_dir = os.path.join(save_dir, 'train')

    for epoch in range(num_epochs):
        epoch_iters_done = 0
        # Each epoch has a training and validation phase
        epoch_loss = {'train': 0.0, 'valid': 0.0}
        epoch_samples = {'train': 0, 'valid': 0}
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()  # Set core to training mode
            else:
                model.eval()  # Set core to evaluate mode


            # Iterate over data.
            for inputs, labels, fnames in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                batch_size = inputs.size(0)
                epoch_samples[phase] += batch_size
                # zero the parameter gradients
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    epoch_loss[phase] += loss.item() * batch_size

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        iters_done += 1
                        epoch_iters_done += 1
                        loss.backward()
                        optimizer.step()
                        lr_scheduler.step()

                if phase == 'train':
                    if save_per_epoch:
                        torch.save({'global_step': iters_done + iters_start,
                                    'model_state_dict': model.state_dict(),
                                    'optimizer_state_dict': optimizer.state_dict(),
                                    'lr_scheduler_state_dict': lr_scheduler.state_dict()},
                                   os.path.join(train_dir, f'training-params-{iters_done + iters_start}.pth'))
        train_avg_loss = epoch_loss['train'] / epoch_samples['train']
        valid_avg_loss = epoch_loss['valid'] / epoch_samples['valid']
        logger.info(f"Epoch [{epoch+1}/{num_epochs}] | Train Loss: {train_avg_loss:.4f} | Valid Loss: {valid_avg_loss:.4f}")
                        
    torch.save({'global_step': iters_done + iters_start,
                                    'model_state_dict': model.state_dict(),
                                    'optimizer_state_dict': optimizer.state_dict(),
                                    'lr_scheduler_state_dict': lr_scheduler.state_dict()},
                                   os.path.join(train_dir, f'P_LymphCS.pth'))
        
    return model


def __check_dataset(records, data_pattern):
    if not isinstance(records, (list, tuple)):
        records = [records]
    if all(os.path.isfile(r) for r in records):
        return {'records': records, 'ori_img_root': data_pattern}
    else:
        return {'records': None, 'ori_img_root': data_pattern}


# def create_model(model_name, **kwargs):
#     supported_models = [k for k in models.__dict__
#                         if not k.startswith('_') and type(models.__dict__[k]).__name__ != 'module']
#     if model_name in supported_models:
#         return models.__dict__[model_name](**kwargs)
#     raise ValueError(f'{model_name} not supported!')


def main(args):
    assert args.gpus is None or args.batch_size % len(args.gpus) == 0, 'Batch size must exactly divide number of gpus'
    image_size = (512, 512)
    # Initialize data transformer for this run
    kwargs = {}
    data_transforms = {'train': create_standard_image_transformer(image_size, phase='train',
                                                                  normalize_method=args.normalize_method),
                       'valid': create_standard_image_transformer(image_size, phase='valid',
                                                                  normalize_method=args.normalize_method)}
    # Initialize datasets and dataloader for this run
    image_datasets = {'train': create_classification_dataset(
         recursive=True, transform=data_transforms['train'], labels_file=args.labels_file,
        batch_balance=args.batch_balance, **__check_dataset(args.train, args.data_pattern))}
    image_datasets['valid'] = create_classification_dataset(
         transform=data_transforms['valid'], classes=image_datasets['train'].classes,
        recursive=True, **__check_dataset(args.valid, args.data_pattern))
    assert image_datasets['train'].classes == image_datasets['valid'].classes
    # Initialize the core for this run
    logger.info(f'Creating model {args.model_name}...')

    # Creat Models
    model = create_model(args.model_name, num_classes=image_datasets['train'].num_classes, **kwargs)

    # Send the core to GPU
    if args.gpus and len(args.gpus) > 1:
        model_ft = nn.DataParallel(model, device_ids=args.gpus)
    elif len(args.gpus) == 1:
        device = torch.device(f"cuda:{args.gpus[0]}" if torch.cuda.is_available() else "cpu")
        model_ft = model.to(device)
    else:
        model_ft = model.to('cpu')
    dataloaders_dict = {x: utils.data.DataLoader(image_datasets[x],
                                                 batch_size=args.batch_size,
                                                 drop_last=False,
                                                 shuffle=True,
                                                 num_workers=args.j)
                        for x in ['train', 'valid']}
    # Initialize optimizer and learning rate for this run
    optimizer = create_optimizer(args.optimizer, parameters=model_ft.parameters(), lr=args.init_lr)
    lr_scheduler = create_lr_scheduler('cosine', optimizer,
                                       T_max=args.epochs * len(image_datasets['train']) // args.batch_size)
    # Setup the loss function
    criterion = create_losses('softmax_ce')

    if args.weights != "":
        assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
        weights_dict = torch.load(args.weights, map_location=device)

        model.load_state_dict(weights_dict['model_state_dict'], strict=False)


    # Train and evaluate
    result_save_root= args.model_root
    os.makedirs(result_save_root, exist_ok=True)
    model_dir = result_save_root
    logger.info('Start training...')
    train_model(model_ft, device, dataloaders_dict, criterion, optimizer,
                                 lr_scheduler, num_epochs=args.epochs,
                                 save_dir=os.path.join(model_dir, args.model_name),
                                 labels_file=args.labels_file,
                                 task_spec={'num_classes': image_datasets['train'].num_classes,
                                            'model_name': args.model_name,
                                            'transform': {'input_size': image_size,
                                                          'normalize_method': args.normalize_method},
                                            'device': str(device)},
                                 save_per_epoch=args.save_per_epoch)


DATA_ROOT = os.path.expanduser(r'./Label')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch Classification Training')

    parser.add_argument('--train', nargs='*', default=os.path.join(DATA_ROOT, 'train.txt'), help='Training dataset')
    parser.add_argument('--valid', nargs='*', default=os.path.join(DATA_ROOT, 'val.txt'), help='Validation dataset')
    parser.add_argument('--labels_file', default=os.path.join(DATA_ROOT, 'labels.txt'), help='Labels file')
    parser.add_argument('--data_pattern', default=os.path.join(DATA_ROOT, 'images'), nargs='*',
                        help='Where to save origin image data.')
    parser.add_argument('-j', '--worker', dest='j', default=4, type=int, help='Number of workers.(default=0)')
    parser.add_argument('--batch_balance', default=False, action='store_true', help='Batch balance samples for train.')
    parser.add_argument('--normalize_method', default='imagenet', choices=['-1+1', 'imagenet'],
                        help='Normalize method.')
    parser.add_argument('--model_name', default='resnet18', help='Model name')
    parser.add_argument('--gpus', type=int, nargs='*', default=[0], help='GPU index to be used!')
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--epochs', default=30, type=int, help='number of total epochs to run')
    parser.add_argument('--init_lr', default=0.01, type=float, help='initial learning rate')
    parser.add_argument('--optimizer', default='sgd', help='Optimizer')
    parser.add_argument('--iters_start', default=0, type=int, help='Iters start')
    parser.add_argument('--save_per_epoch', action='store_true', help='Whether save each Epoch，default False')
    parser.add_argument('--weights', type=str, default='./')

    main(parser.parse_args())
