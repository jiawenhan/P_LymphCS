from __future__ import print_function

import argparse
import os

from utils.file_utils import save_pkl
from utils.core_utils import train
from dataset.dataset_generic import Generic_MIL_Dataset

import torch
import pandas as pd
import numpy as np


def main(args):
    os.makedirs(args.results_dir, exist_ok=True)

    if args.k_start == -1:
        start = 0
    else:
        start = args.k_start
    if args.k_end == -1:
        end = args.k
    else:
        end = args.k_end

    all_test_auc = []
    all_val_auc = []
    all_test_acc = []
    all_val_acc = []
    folds = np.arange(start, end)
    for i in folds:
        seed_torch(args.seed)
        train_dataset, val_dataset, test_dataset = dataset.return_splits(args.backbone, args.patch_size, from_id=False,
                csv_path='{}/splits_{}.csv'.format(args.split_dir, i))

        datasets = (train_dataset, val_dataset, test_dataset)
        if args.preloading == 'yes':
            for d in datasets:
                d.pre_loading()

        val_results, test_results, test_auc, val_auc, test_acc, val_acc = train(datasets, i, args)

        all_test_auc.append(test_auc)
        all_val_auc.append(val_auc)
        all_test_acc.append(test_acc)
        all_val_acc.append(val_acc)
        # Persist slide-level predictions for downstream analysis.
        filename = os.path.join(args.results_dir, 'split_{}_val_results.pkl'.format(i))
        save_pkl(filename, val_results)
        filename = os.path.join(args.results_dir, 'split_{}_test_results.pkl'.format(i))
        save_pkl(filename, test_results)

    final_df = pd.DataFrame({'folds': folds, 'test_auc': all_test_auc,
        'val_auc': all_val_auc, 'test_acc': all_test_acc, 'val_acc' : all_val_acc})

    if len(folds) != args.k:
        save_name = 'summary_partial_{}_{}.csv'.format(start, end)
    else:
        save_name = 'summary.csv'
    final_df.to_csv(os.path.join(args.results_dir, save_name))


parser = argparse.ArgumentParser(description='Configurations for WSI Training')
parser.add_argument('--data_root_dir', type=str, default=None,
                    help='root directory containing extracted WSI features; overrides the CSV dir column')
parser.add_argument('--max_epochs', type=int, default=50,
                    help='maximum number of epochs to train (default: 50)')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='learning rate (default: 0.0001)')
parser.add_argument('--label_frac', type=float, default=1.0,
                    help='fraction of training labels (default: 1.0)')
parser.add_argument('--reg', type=float, default=1e-5,
                    help='weight decay (default: 1e-5)')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed for reproducible experiment (default: 1)')
parser.add_argument('--k', type=int, default=1, help='number of folds (default: 1)')
parser.add_argument('--k_start', type=int, default=-1, help='first fold index; -1 enables automatic resume')
parser.add_argument('--k_end', type=int, default=-1, help='exclusive final fold index; -1 uses all folds')
parser.add_argument('--results_dir', default='./results', help='results directory (default: ./results)')
parser.add_argument('--split_dir', type=str, default=None,
                    help='manually specify the set of splits to use, '
                    +'instead of infering from the task and label_frac argument (default: None)')
parser.add_argument('--log_data', action='store_true', default=False, help='log data using tensorboard')
parser.add_argument('--testing', action='store_true', default=False, help='debugging tool')
parser.add_argument('--early_stopping', action='store_true', default=False, help='enable early stopping')
parser.add_argument('--opt', type=str, choices = ['adam', 'sgd'], default='sgd')
parser.add_argument('--drop_out', type=float, default=0., help='dropout probability (default: 0.0)')
parser.add_argument('--model_type', type=str, default='mamba_mil',
                    choices=['mean_mil', 'max_mil', 'att_mil', 'trans_mil', 's4model', 'mamba_mil', 'clam_mb'],
                    help='MIL architecture (default: mamba_mil)')
parser.add_argument('--exp_code', type=str, required=True, help='experiment name used for result files')
parser.add_argument('--weighted_sample', action='store_true', default=False, help='enable weighted sampling')
parser.add_argument('--task', type=str, required=True, choices=['LUAD_LUSC', 'BRACS', 'Lymph'])
parser.add_argument('--backbone', type=str, default='resnet50')
parser.add_argument('--patch_size', type=str, default='')
parser.add_argument('--preloading', type=str, choices=['yes', 'no'], default='no')
parser.add_argument('--in_dim', type=int, default=1024)

parser.add_argument('--mambamil_rate',type=int, default=10, help='mambamil_rate')
parser.add_argument('--mambamil_layer',type=int, default=2, help='mambamil_layer')
parser.add_argument('--mambamil_type',type=str, default='SRMamba', choices= ['Mamba', 'BiMamba', 'SRMamba'], help='mambamil_type')


args = parser.parse_args()
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Device is:', device)

def seed_torch(seed=7):
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

seed_torch(args.seed)

settings = {'num_splits': args.k,
            'k_start': args.k_start,
            'k_end': args.k_end,
            'task': args.task,
            'max_epochs': args.max_epochs,
            'results_dir': args.results_dir,
            'lr': args.lr,
            'experiment': args.exp_code,
            'reg': args.reg,
            'label_frac': args.label_frac,
            'seed': args.seed,
            'model_type': args.model_type,
            "use_drop_out": args.drop_out,
            'weighted_sample': args.weighted_sample,
            'opt': args.opt}


print('\nLoad Dataset')

if args.task == 'LUAD_LUSC':
    args.n_classes=2
    dataset = Generic_MIL_Dataset(csv_path = 'dataset_csv/LUAD_LUSC.csv',
                            data_dir=args.data_root_dir,
                            shuffle = False,
                            seed = args.seed,
                            print_info = True,
                            label_dict = {'LUAD':0, 'LUSC':1},
                            patient_strat=False,
                            ignore=[])

elif args.task == 'BRACS':
    args.n_classes=7
    dataset = Generic_MIL_Dataset(csv_path = 'dataset_csv/BRACS.csv',
                            data_dir=args.data_root_dir,
                            shuffle = False,
                            seed = args.seed,
                            print_info = True,
                            label_dict = {'PB':0, 'IC':1, 'DCIS':2, 'N':3, 'ADH': 4,
                                          'FEA':5, 'UDH': 6 },
                            patient_strat=False,
                            ignore=[])

elif args.task == 'Lymph':
    args.n_classes=5
    dataset = Generic_MIL_Dataset(csv_path = 'dataset_csv/Lymph.csv',
                            data_dir=args.data_root_dir,
                            shuffle = False,
                            seed = args.seed,
                            print_info = True,
                            label_dict = {'N':0, 'T':1, 'H':2, 'X':3, 'D': 4},
                            patient_strat=False,
                            ignore=[])

else:
    raise NotImplementedError

args.results_dir = os.path.join(args.results_dir, str(args.exp_code) + '_s{}'.format(args.seed))
if not os.path.isdir(args.results_dir):
    os.makedirs(args.results_dir)

if args.split_dir is None:
    args.split_dir = os.path.join('splits', args.task+'_{}'.format(int(args.label_frac*100)))

print('split_dir: ', args.split_dir)
assert os.path.isdir(args.split_dir)

settings.update({'split_dir': args.split_dir, 'results_dir': args.results_dir})

# Resume from the first fold without both saved prediction files.
if args.k_start == -1:
    folds = args.k if args.k_end == -1 else args.k_end
    args.k_start = folds
    for i in range(folds):
        val_file = os.path.join(args.results_dir, 'split_{}_val_results.pkl'.format(i))
        test_file = os.path.join(args.results_dir, 'split_{}_test_results.pkl'.format(i))
        if not (os.path.exists(val_file) and os.path.exists(test_file)):
            args.k_start = i
            break
    print('Training from fold: {}'.format(args.k_start))

settings['k_start'] = args.k_start
with open(os.path.join(args.results_dir, 'experiment.txt'), 'w') as f:
    print(settings, file=f)

print("################# Settings ###################")
for key, val in settings.items():
    print("{}:  {}".format(key, val))

if __name__ == "__main__":
    requested_end = args.k if args.k_end == -1 else args.k_end
    if args.k_start >= requested_end:
        print("All requested folds have already completed.")
    else:
        main(args)
        print("Training finished.")
