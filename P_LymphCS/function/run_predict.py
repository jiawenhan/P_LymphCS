import argparse
import json
import os
import logging
import shutil
import time
from typing import Union, List
from torch import Tensor
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim
import torch.utils as utils
import torch.utils.data

from sklearn.utils import column_or_1d
from sklearn.metrics import roc_auc_score

from P_LymphCS.core import create_standard_image_transformer
from P_LymphCS.custom import create_classification_dataset
from P_LymphCS import models


logger = logging.root
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("[%(asctime)s - %(filename)s:%(lineno)4s]\t%(levelname)s\t%(message)s",
                                       '%Y-%m-%d %H:%M:%S'))
logger.handlers = [handler]


def calculate_sens_spec(y_true, y_score):
    thresholds = np.unique(y_score)
    thresholds = np.concatenate([[thresholds[0] - 1e-6], thresholds, [thresholds[-1] + 1e-6]])

    tpr_list = []
    tnr_list = []
    total_positives = np.sum(y_true == 1)
    total_negatives = np.sum(y_true == 0)

    for threshold in thresholds:
        y_pred = (y_score >= threshold).astype(int)

        tp = np.sum((y_true == 1) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))

        tpr = tp / total_positives if total_positives > 0 else 0.0
        tnr = tn / total_negatives if total_negatives > 0 else 0.0

        tpr_list.append(tpr)
        tnr_list.append(tnr)

    return np.array(tpr_list), np.array(tnr_list), thresholds


def calculate_auc_with_ci(y_true, y_score, alpha=0.05, n_bootstraps=1000):

    auc = roc_auc_score(y_true, y_score)

    n = len(y_true)
    bootstrapped_aucs = []

    for _ in range(n_bootstraps):
        indices = np.random.choice(n, n, replace=True)
        if len(np.unique(y_true[indices])) < 2:
            continue

        bootstrapped_aucs.append(roc_auc_score(y_true[indices], y_score[indices]))

    sorted_aucs = np.array(bootstrapped_aucs)
    sorted_aucs.sort()

    lower = sorted_aucs[int(alpha / 2 * len(sorted_aucs))]
    upper = sorted_aucs[int((1 - alpha / 2) * len(sorted_aucs))]

    return auc, (lower, upper)



def analysis_pred_binary(y_true: Union[List, np.ndarray, pd.DataFrame], y_score: Union[List, np.ndarray, pd.DataFrame],
                         y_pred: Union[List, np.ndarray, pd.DataFrame] = None, reverse: bool = False):

    if isinstance(y_score, (list, tuple)):
        y_score = np.array(y_score)
    y_true = column_or_1d(np.array(y_true))
    assert sorted(np.unique(y_true)) == [0, 1], f"The result must be a 2-category classification"
    assert len(y_true) == len(y_score), 'The number of samples must be equal'
    if len(y_score.shape) == 2 and y_score.shape[1] == 2:
        y_score = column_or_1d(y_score[:, 1])
    elif len(y_score.shape) > 2:
        raise ValueError(f"y_score should >2,now is {y_score.shape}")
    else:
        y_score = column_or_1d(y_score)
    if reverse:
        y_true = 1 - y_true
        y_score = 1 - y_score
    tpr, tnr, thres = calculate_sens_spec(y_true, y_score)

    acc = np.sum(y_true == y_pred) / len(y_true)
    tp = np.sum(y_true[y_true == 1] == y_pred[y_true == 1])
    tn = np.sum(y_true[y_true == 0] == y_pred[y_true == 0])
    fp = np.sum(y_pred[y_true == 0] == 1)
    fn = np.sum(y_pred[y_true == 1] == 0)
    # print(tp, tn, fp, fn)
    ppv = tp / (tp + fp + 1e-6)
    npv = tn / (tn + fn + 1e-6)
    auc, ci = calculate_auc_with_ci(y_true, y_score, alpha=0.05)
    tpr = tp / (tp + fn + 1e-6)
    tnr = tn / (fp + tn + 1e-6)
    f1 = 2 * tpr * ppv / (ppv + tpr)
    # print(tp, tn, fp, fn)
    return acc, auc, ci, tpr, tnr, ppv, npv, ppv, tpr, f1, thres



def metric_details(log_file, metric_spec, epoch, subset='',
                   metric_spec_agg='mean', metric_spec_ids=None):
    log = pd.read_csv(log_file, names=['fname', 'pred_score', 'pred_label', 'gt'], sep='\t')
    ul_labels = np.unique(log['pred_label'])
    metric_results = []
    if metric_spec_ids is None:
        if len(ul_labels) > 2:
            metric_spec_ids = list(ul_labels)
        else:
            metric_spec_ids = [1]
    elif not isinstance(metric_spec_ids, (list, tuple)):
        metric_spec_ids = [metric_spec_ids]
    for ul in metric_spec_ids:
        pred_score = list(map(lambda x: x[0] if x[1] == ul else 1 - x[0],
                              np.array(log[['pred_score', 'pred_label']])))
        gt = [1 if gt_ == ul else 0 for gt_ in np.array(log['gt'])]
        acc, auc, ci, tpr, tnr, ppv, npv, _, _, _, thres = analysis_pred_binary(gt, pred_score)
        ci = f"{ci[0]:.4f}-{ci[1]:.4f}"
        metric_results.append([epoch, ul, acc, auc, ci, tpr, tnr, ppv, npv, thres, subset])
    metric_results = pd.DataFrame(metric_results,
                                  columns=['Epoch', 'SpecID', 'Acc', 'AUC', '95% CI', 'Sensitivity', 'Specificity',
                                           'PPV', 'NPV', 'Threshold', 'Cohort'])
    bst_metric = metric_results.describe()[metric_spec][metric_spec_agg]
    return bst_metric, metric_results


def train_model(model, device, dataloaders, num_epochs=1, iters_start=0, save_dir='./', **kwargs):

    val_acc_history = []
    best_acc = 0.0
    best_epoch = None
    iters_done = 0
    os.makedirs(os.path.join(save_dir, 'test'))
    test_dir = os.path.join(save_dir, 'test')
    os.makedirs(os.path.join(save_dir, 'test_viz'))
    test_viz_dir = os.path.join(save_dir, 'test_viz')
    valid_log = [('Epoch', 'Iters', 'acc')]
    num_classes = kwargs['task_spec']['num_classes']
    metric_spec = 'acc'
    metric_spec_ids = None
    metric_spec_agg = 'mean'

    for epoch in range(num_epochs):
        epoch_iters_done = 0
        # Each epoch has a training and validation phase
        metric_results_ = []

        for phase in ['test']:
            model.eval()  # Set core to evaluate mode
            valid_time_since = time.time()
            test_file = open(os.path.join(test_dir, f'Epoch-{epoch}.txt'), 'w')
            test_file_spec = open(os.path.join(test_dir, f'Epoch-{epoch}_spec.csv'), 'w')
            print(f'fpath,{",".join([f"label-{iid}" for iid in range(num_classes)])}', file=test_file_spec)
            running_corrects = 0
            running_size = 0
            # Iterate over data.
            for inputs, labels, fnames in dataloaders[phase]:
                inputs = inputs.to(device)
                input_size = inputs.shape[0]
                labels = labels.to(device)
                # zero the parameter gradients
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    if isinstance(outputs, Tensor):
                        probabilities = nn.functional.softmax(outputs, dim=1)
                        probability, predictions = torch.max(probabilities, 1)
                    else:
                        probabilities = nn.functional.softmax(outputs.logits, dim=1)
                        probability, predictions = torch.max(probabilities, 1)
                        # backward + optimize only if in training phase
                    for fname, probs, prob, pred, label in zip(fnames, probabilities, probability, predictions,
                                                                   labels):
                        test_file.write('%s\t%.3f\t%d\t%d\n' % (fname, prob, pred, label))
                        test_file_spec.write('%s,%s\n' % (fname, ','.join(map(str, probs.detach().cpu().numpy()))))
                running_corrects += int(torch.sum(predictions == labels.data))
                running_size += input_size
            epoch_acc = running_corrects * 100 / len(dataloaders[phase].dataset)
            epoch_metric = epoch_acc
            test_file.close()
            test_file_spec.close()
            if metric_spec.lower() != 'acc':
                epoch_metric, metric_results = metric_details(os.path.join(test_dir, f'Epoch-{epoch}.txt'),
                                                                  metric_spec, epoch, 'Test',
                                                                  metric_spec_agg, metric_spec_ids)
                metric_results_.append(metric_results)
            val_acc_history.append(epoch_metric)
            info = 'Phase: {phase}\t{metric}: {acc}\tSpeed: {speed}img/s\tTime: {time}s'
            speed = len(dataloaders[phase].dataset) / (time.time() - valid_time_since)
            logger.info(info.format(
                phase='test',
                metric=metric_spec,
                acc="%.3f" % epoch_metric,
                speed="{:.4f}".format(speed),
                time='%.2f' % (time.time() - valid_time_since)
            ))
            valid_log.append((iters_done, epoch_metric))

    # Save training log to csv
    pd.DataFrame(valid_log).to_csv(os.path.join(test_viz_dir, 'validing_log.txt'),
                                   header=False, index=False, encoding='utf8')
    # Save labels file to viz directory
    shutil.copyfile(kwargs['labels_file'], os.path.join(test_viz_dir, 'labels.txt'))

    # Save task settings.
    with open(os.path.join(test_viz_dir, 'task.json'), 'w') as task_file:
        kwargs['task_spec'].update({"acc": best_acc, 'best_epoch': best_epoch})
        print(json.dumps(kwargs['task_spec'], indent=True, ensure_ascii=False), file=task_file)
    return model, val_acc_history


def __check_dataset(records, data_pattern):
    if not isinstance(records, (list, tuple)):
        records = [records]
    if all(os.path.isfile(r) for r in records):
        return {'records': records, 'ori_img_root': data_pattern}
    else:
        return {'records': None, 'ori_img_root': data_pattern}


def create_model(model_name, **kwargs):
    supported_models = [k for k in models.__dict__
                        if not k.startswith('_') and type(models.__dict__[k]).__name__ != 'module']
    if model_name in supported_models:
        return models.__dict__[model_name](**kwargs)
    raise ValueError(f'{model_name} not supported!')


def main(args):
    assert args.gpus is None or args.batch_size % len(args.gpus) == 0, 'Batch size must exactly divide number of gpus'
    image_size = (512, 512)
    # Initialize data transformer for this run
    kwargs = {}

    data_transforms = {'test': create_standard_image_transformer(image_size, phase='test',
                                                                  normalize_method=args.normalize_method)}
    # Initialize datasets and dataloader for this run
    image_datasets = {'test': create_classification_dataset(
         recursive=True, transform=data_transforms['test'], labels_file=args.labels_file,
        batch_balance=args.batch_balance, **__check_dataset(args.test, args.data_pattern))}
    # Initialize the core for this run
    logger.info(f'Creating model {args.model_name}...')

    model = create_model(args.model_name, num_classes=image_datasets['test'].num_classes, **kwargs)
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
                        for x in ['test']}

    # Pretrain Weights
    if args.weights != "":
        assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
        weights_dict = torch.load(args.weights, map_location=device)
        model.load_state_dict(weights_dict['model_state_dict'])

    # Train and evaluate
    result_save_root= args.model_root
    os.makedirs(result_save_root, exist_ok=True)
    model_dir = result_save_root
    logger.info('Start training...')
    model_ft = train_model(model_ft, device, dataloaders_dict, num_epochs=args.epochs,
                                 save_dir=os.path.join(model_dir, args.model_name),
                                 labels_file=args.labels_file,
                                 task_spec={'num_classes': image_datasets['test'].num_classes,
                                            'model_name': args.model_name,
                                            'transform': {'input_size': image_size,
                                                          'normalize_method': args.normalize_method},
                                            'device': str(device)}
                           )

DATA_ROOT = os.path.expanduser(r'./Label')
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch Classification Training')

    parser.add_argument('--test', nargs='*', default=os.path.join(DATA_ROOT, 'test.txt'), help='Test dataset')
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
    parser.add_argument('--epochs', default=3, type=int, help='number of total epochs to run')
    parser.add_argument('--init_lr', default=0.1, type=float, help='initial learning rate')
    parser.add_argument('--optimizer', default='sgd', help='Optimizer')
    parser.add_argument('--retrain', default=None, help='Retrain from path')
    parser.add_argument('--iters_verbose', default=1, type=int, help='print frequency')
    parser.add_argument('--iters_start', default=0, type=int, help='Iters start')
    parser.add_argument('--pretrained', action='store_true', help='Use pretrained core or not')
    parser.add_argument('--save_per_epoch', action='store_true', help='Whether save each Epoch，default False')
    parser.add_argument('--weights', type=str, default='./vit_base_patch32_224_in21k.pth', help='Weights of model')

    main(parser.parse_args())
