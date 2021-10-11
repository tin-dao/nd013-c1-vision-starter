import argparse
import glob
import os
import random

import numpy as np

from utils import get_module_logger
from sklearn.model_selection import train_test_split


def split(data_dir):
    """
    Create three splits from the processed records. The files should be moved to new folders in the 
    same directory. This folder should be named train, val and test.

    args:
        - data_dir [str]: data directory, /mnt/data
    """
    # Get list of tfrecord file
    file_names = [os.path.basename(x) for x in glob.glob(os.path.join(data_dir, '*.tfrecord'))]
    
    # Create folders for training, validation and test sets
    train_path = os.path.join(data_dir, 'train')
    val_path = os.path.join(data_dir, 'val')
    test_path = os.path.join(data_dir, 'test')
    print('Creating training, validation and test folders...')
    os.mkdir(train_path)
    os.mkdir(val_path)
    os.mkdir(test_path)

    # We have small data set so we will split 60% train, 20 % validation and 20% test.
    train_set, val_and_test_set = train_test_split(file_names, test_size=0.4, random_state=99)
    val_set, test_set = train_test_split(val_and_test_set, test_size=0.5, random_state=99)

    # Move files to new folders
    print('Moving files in training set...')
    for file_name in train_set:
        os.rename(os.path.join(data_dir, file_name), os.path.join(train_path, file_name))
    print('Moving files in validation set...')
    for file_name in val_set:
        os.rename(os.path.join(data_dir, file_name), os.path.join(val_path, file_name))
    print('Moving files in test set...')
    for file_name in test_set:
        os.rename(os.path.join(data_dir, file_name), os.path.join(test_path, file_name))


if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description='Split data into training / validation / testing')
    parser.add_argument('--data_dir', required=True,
                        help='data directory')
    args = parser.parse_args()

    logger = get_module_logger(__name__)
    logger.info('Creating splits...')
    split(args.data_dir)