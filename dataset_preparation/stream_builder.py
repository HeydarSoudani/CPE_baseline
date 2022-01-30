import pandas as pd 
import numpy as np
import argparse
import random
import gzip
import time
import os

## == Params ===========================
parser = argparse.ArgumentParser()
parser.add_argument('--seen_class_num', type=int, default=5, help='')
parser.add_argument('--spc', type=int, default=1200, help='samples per class for initial dataset')
parser.add_argument('--dataset', type=str, default='mnist', help='') #[mnist, fmnist, cifar10]
parser.add_argument('--saved', type=str, default='./data/', help='')
parser.add_argument('--seed', type=int, default=1, help='')  # seed=1 for regular novel class selection
args = parser.parse_args()

# = Add some variables to args =========
args.data_path = 'data/{}'.format(args.dataset)
args.train_file = '{}_train.csv'.format(args.dataset)
args.stream_file = '{}_stream.csv'.format(args.dataset)

## == Apply seed =======================
np.random.seed(args.seed)

## == Set class number =================
if args.dataset in ['mnist', 'fmnist', 'cifar10']:
  args.n_classes = 10
elif args.dataset in ['cifar100']:
  args.n_classes = 100

## == Add novel points params ==========
start_point = 3
if args.dataset in ['mnist', 'fmnist']:
  last_point = 35
elif args.dataset in ['cifar10', 'cifar100']:
  last_point = 25


if __name__ == '__main__':
  ## ========================================
  # == Get MNIST dataset ====================
  if args.dataset == 'mnist':
    train_data = pd.read_csv(os.path.join(args.data_path, "mnist_train.csv"), sep=',').values
    test_data = pd.read_csv(os.path.join(args.data_path, "mnist_test.csv"), sep=',').values
    X_train, y_train = train_data[:, 1:], train_data[:, 0]
    X_test, y_test = test_data[:, 1:], test_data[:, 0]
  ## ========================================
  ## ========================================

  ## ========================================
  # == Get FashionMNIST dataset =============
  if args.dataset == 'fmnist':
    train_data = pd.read_csv(os.path.join(args.data_path, "fmnist_train.csv"), sep=',').values
    test_data = pd.read_csv(os.path.join(args.data_path, "fmnist_test.csv"), sep=',').values
    X_train, y_train = train_data[:, 1:], train_data[:, 0]
    X_test, y_test = test_data[:, 1:], test_data[:, 0]
  ## ========================================
  ## ========================================

  ## ========================================
  # == Get CIFAR10 dataset ==================
  if args.dataset == 'cifar10':
    train_data = pd.read_csv(os.path.join(args.data_path, 'cifar10_train.csv'), sep=',', header=None).values
    test_data = pd.read_csv(os.path.join(args.data_path, 'cifar10_test.csv'), sep=',', header=None).values
    X_train, y_train = train_data[:, :-1], train_data[:, -1]
    X_test, y_test = test_data[:, :-1], test_data[:, -1]
  ## ========================================
  ## ========================================

  data = np.concatenate((X_train, X_test), axis=0)  #(70000, 784)
  labels = np.concatenate((y_train, y_test), axis=0)#(70000,)
  n_data = data.shape[0]

  # == Select seen & unseen classes ==========
  seen_class = np.random.choice(args.n_classes, args.seen_class_num, replace=False)
  unseen_class = [x for x in list(set(labels)) if x not in seen_class]
  # seen_class = np.array([0, 1, 2, 3, 4]) 
  # unseen_class = [5, 6, 7, 8, 9]
  print('seen: {}'.format(seen_class))
  print('unseen: {}'.format(unseen_class))

  # == split data by classes =================
  class_data = {}
  for class_label in set(labels):
    class_data[class_label] = []  
  for idx, sample in enumerate(data):
    class_data[labels[idx]].append(sample)
  
  for label in class_data.keys():
    class_data[label] = np.array(class_data[label])

  for label, data in class_data.items():
    print('Label: {} -> {}'.format(label, data.shape))  


  # == Preparing train dataset and test seen data ===
  train_data = []
  test_data_seen = []
  for seen_class_item in seen_class:
    train_idx = np.random.choice(class_data[seen_class_item].shape[0], args.spc, replace=False)
    seen_data = class_data[seen_class_item][train_idx]
    class_data[seen_class_item] = np.delete(class_data[seen_class_item], train_idx, axis=0)

    train_data_class = np.concatenate((seen_data, np.full((args.spc , 1), seen_class_item)), axis=1)
    train_data.extend(train_data_class)

  train_data = np.array(train_data) #(6000, 785)
  
  np.random.shuffle(train_data)
  print('train data: {}'.format(train_data.shape))
  pd.DataFrame(train_data).to_csv(os.path.join(args.saved, args.train_file),
    header=None,
    index=None
  )
  print('train data saved in {}'.format(os.path.join(args.saved, args.train_file)))

 
  all_class_to_select = seen_class.tolist()
  chunk_size = 1000
  n_chunk = int(n_data / chunk_size) 
  n_chunk_stream = n_chunk - 6
  chunks = []
  add_new_class_points = np.random.choice(np.arange(start_point, n_chunk_stream-last_point), len(unseen_class), replace=False)
  print('Novel class points: {}'.format(add_new_class_points))
  for i_chunk in range(n_chunk_stream):
    chunk_data = []
    
    # add novel class to test data pool
    if i_chunk in add_new_class_points:  
      rnd_uns_class = unseen_class[0]
      unseen_class.remove(rnd_uns_class)
      all_class_to_select.append(rnd_uns_class)
    
    # Select data from every known class
    if len(all_class_to_select) > 5:
      select_class_idx = np.random.choice(len(all_class_to_select), len(all_class_to_select)-1, replace=False)
      class_to_select = [all_class_to_select[i] for i in select_class_idx]
    else:
      class_to_select = all_class_to_select

    items_per_class = int(chunk_size / len(class_to_select))
    removed_class = []

    for known_class in class_to_select:
      n = class_data[known_class].shape[0]
      if n > items_per_class:
        idxs = np.random.choice(range(n), size=items_per_class, replace=False)  
        selected_data_class = np.concatenate((class_data[known_class][idxs], np.full((items_per_class , 1), known_class)), axis=1)
        chunk_data.extend(selected_data_class)  
        class_data[known_class] = np.delete(class_data[known_class], idxs, axis=0)
      
      else:
        selected_data_class = np.concatenate((class_data[known_class], np.full((class_data[known_class].shape[0] , 1), known_class)), axis=1)
        chunk_data.extend(selected_data_class)
        removed_class.append(known_class)
        del class_data[known_class]

    if len(removed_class) > 0:
      all_class_to_select = [e for e in all_class_to_select if e not in removed_class]

    chunk_data = np.array(chunk_data)

    # check if chunk_data < chunk_size
    if chunk_data.shape[0] < chunk_size:
      needed_data = chunk_size - chunk_data.shape[0]
      helper_class = all_class_to_select[-1]

      n = class_data[helper_class].shape[0]
      idxs = np.random.choice(range(n), size=needed_data, replace=False)
      selected_data_class = np.concatenate((class_data[helper_class][idxs], np.full((needed_data , 1), helper_class)), axis=1)
      chunk_data = np.concatenate((chunk_data, selected_data_class), axis=0)

    np.random.shuffle(chunk_data)
    chunks.append(chunk_data)
    
  stream_data = np.concatenate(chunks, axis=0)
  print('stream_data size: {}'.format(stream_data.shape))
  pd.DataFrame(stream_data).to_csv(os.path.join(args.saved, args.stream_file),
    header=None,
    index=None
  )
  print('stream data saved in {}'.format(os.path.join(args.saved, args.stream_file)))
  