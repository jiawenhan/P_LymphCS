import math
import os
import random
from functools import partial
from typing import List
import logging
import torch.utils.data as data
from PIL import Image


logger = logging.root
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("[%(asctime)s - %(filename)s:%(lineno)4s]\t%(levelname)s\t%(message)s",
                                       '%Y-%m-%d %H:%M:%S'))
logger.handlers = [handler]


def annotation_parser(record, sep='\t'):
    file_name_ = []
    label_ = []
    bbox_ = []
    with open(record, encoding='utf8') as f:
        for l in f.readlines():
            items = l.strip().split(sep)
            if not (len(items) == 2 or len(items) == 6):
                raise ValueError("Annotation must be length of 2 for `file_name` and `label` or "
                                 "length of 6 for `file_name`, `label`, `top`, `left`, `height`, `width`.")
            file_name_.append(items[0])
            label_.append(items[1])
            bbox_.append([float(b) for b in items[2:]])
    return file_name_, label_, bbox_


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class ListDataset(data.Dataset):
    def __init__(self, records: List[str], labels_file: str = None,
                 ori_img_root: str = None, batch_balance: bool = False,
                 transform=None, target_transform=None,
                 retry: int = None, **kwargs):
        self.file_names = []
        self.labels = []
        self.boxes = []
        del kwargs
        record_parser = partial(annotation_parser, sep='\t')
        loader = pil_loader
        if isinstance(records, str):
            records = [records]
        assert all(r and os.path.exists(r) and os.path.isfile(r) for r in records), f"Not all records file exist! " \
                                                                                    f"Check {records}."

        # Parse records file.
        for record in records:
            logger.info(f'Parsing record file {record}')
            file_name_, label_, bbox_ = record_parser(record)
            for file_name, label, bbox in zip(file_name_, label_, bbox_):
                self.file_names.append(file_name)
                self.labels.append(label)
                self.boxes.append(bbox)

        # Cut samples and batch balance it if necessary.
        if batch_balance:
            logger.info(f'Batch Balance！')
            _labels_samples = {}
            _max_samples = 0
            for file_name, label, bbox in zip(self.file_names, self.labels, self.boxes):
                if label not in _labels_samples:
                    _labels_samples[label] = []
                _labels_samples[label].append((file_name, bbox))
            for l in _labels_samples:
                random.shuffle(_labels_samples[l])
                current_len = len(_labels_samples[l])
                if current_len > _max_samples:
                    _max_samples = current_len
            for l in _labels_samples:
                _enlarge_ratio = math.ceil(_max_samples / len(_labels_samples[l]))
                _labels_samples[l] = _labels_samples[l] * _enlarge_ratio
                _labels_samples[l] = _labels_samples[l][:_max_samples]
                random.shuffle(_labels_samples[l])
            # unzip data and reinitialise
            self.file_names = []
            self.labels = []
            self.boxes = []
            for l in _labels_samples:
                for file_name, bbox in _labels_samples[l]:
                    self.file_names.append(file_name)
                    self.labels.append(l)
                    self.boxes.append(bbox)

        if labels_file and os.path.exists(labels_file) and os.path.isfile(labels_file):
            with open(labels_file, encoding='utf8') as f:
                self.classes = [l.strip() for l in f.readlines()]
                if len(self.labels) == 0:
                    raise ValueError('No Label Data Find！')
                if not sorted(self.classes) == sorted(list(set(self.labels))):
                    if set(self.classes) - set(self.labels):
                        raise ValueError(f"Labels's set must equal to {labels_file}")
        else:
            self.classes = sorted(list(set(self.labels)))
            if labels_file:
                os.makedirs(os.path.dirname(labels_file), exist_ok=True)
                with open(labels_file, 'w', encoding='utf8') as f:
                    f.write('\n'.join(self.classes))

        self.classes_to_idx = dict([(c, i) for i, c in enumerate(self.classes)])
        self.labels = [self.classes_to_idx[l] for l in self.labels]

        self.labels_file = labels_file
        self.root = ori_img_root
        self.loader = loader
        self.transform = transform
        self.target_transform = target_transform
        self.retry = retry
        if not len(self.labels):
            logger.warning('0 sample in this dataset!')

    def __get_item(self, index):
        path_ = self.file_names[index]
        label_ = self.labels[index]

        sample_ = self.loader(path_)
        if self.transform is not None:
            sample_ = self.transform(sample_)
        if self.target_transform is not None:
            label_ = self.target_transform(label_)
        return sample_, label_, path_

    @property
    def num_classes(self):
        return len(self.classes)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index: int):
        """
        Get an example.

        :param index: Index
        :return: (sample, bbox, label) where label is class_index of the target class.
        """
        try:
            return self.__get_item(index)
        except Exception as e:
            attempt = 0
            times = 'infinite' if self.retry is None else self.retry
            logger.warning(f'{self.file_names[index]} is dropped because of {e}')
            logger.info(f'We now attempt {times} times to get datasets sample randomly!')
            while self.retry is None or attempt < self.retry:
                _rand_idx = random.randint(0, len(self.labels) - 1)
                logger.info(f'Attempting at {attempt + 1} using index {_rand_idx}!')
                try:
                    return self.__get_item(_rand_idx)
                except:
                    pass
                attempt += 1
            raise e


def create_classification_dataset(**kwargs):
    dataset = ListDataset(**kwargs)
    return dataset


def save_classification_dataset_labels(dataset, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        f.write('\n'.join(dataset.classes))


if __name__ == '__main__':
    pass
