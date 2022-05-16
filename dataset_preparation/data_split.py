import torch
import torchvision.transforms as transforms
import pandas as pd 
import numpy as np
import argparse
import pickle
import time
import os

def unpickle(file):
  with open(file, 'rb') as fo:
    dict = pickle.load(fo, encoding='bytes')
  return dict

## == Params ==========================
parser = argparse.ArgumentParser()
parser.add_argument('--n_tasks', type=int, default=5, help='')
parser.add_argument(
  '--dataset',
  type=str,
  choices=[
    'mnist',
    'pmnist',
    'rmnist',
    'fmnist',
    'pfmnist',
    'rfmnist',
    'cifar10',
    'cifar100',
  ],
  default='cifar10',
  help='')
parser.add_argument('--seed', type=int, default=1, help='')
args = parser.parse_args()

# = Add some variables to args ========
if args.dataset in ['mnist', 'pmnist', 'rmnist']:
  data_folder = 'mnist'
elif args.dataset in ['fmnist', 'pfmnist', 'rfmnist']:
  data_folder = 'fmnist'
else:
  data_folder = args.dataset

args.data_path = 'data/{}'.format(data_folder)
args.saved = './data/split_{}'.format(args.dataset)
args.train_path = 'train'
args.test_path = 'test'

## == Apply seed ======================
torch.manual_seed(args.seed)
np.random.seed(args.seed)

## == Save dir ========================
if not os.path.exists(os.path.join(args.saved, args.train_path)):
  os.makedirs(os.path.join(args.saved, args.train_path))
if not os.path.exists(os.path.join(args.saved, args.test_path)):
  os.makedirs(os.path.join(args.saved, args.test_path))

if __name__ == '__main__':
  ## ========================================
  # == Get MNIST dataset ====================
  if args.dataset in ['mnist', 'pmnist', 'rmnist']:
    train_data = pd.read_csv(os.path.join(args.data_path, "mnist_train.csv"), sep=',').values
    test_data = pd.read_csv(os.path.join(args.data_path, "mnist_test.csv"), sep=',').values
    X_train, y_train = train_data[:, 1:], train_data[:, 0]
    X_test, y_test = test_data[:, 1:], test_data[:, 0]
  ## ========================================
  ## ========================================

  ## ========================================
  # == Get Fashion-MNIST dataset ============
  if args.dataset in ['fmnist', 'pfmnist', 'rfmnist']:
    train_data = pd.read_csv(os.path.join(args.data_path, "fmnist_train.csv"), sep=',').values
    test_data = pd.read_csv(os.path.join(args.data_path, "fmnist_test.csv"), sep=',').values
    X_train, y_train = train_data[:, 1:], train_data[:, 0]
    X_test, y_test = test_data[:, 1:], test_data[:, 0]
  ## ========================================
  ## ========================================

  ## ========================================
  # == Get Cifar10 dataset ==================
  if args.dataset == 'cifar10':
    train_data = pd.read_csv(os.path.join(args.data_path, 'cifar10_train.csv'), sep=',', header=None).values
    test_data = pd.read_csv(os.path.join(args.data_path, 'cifar10_test.csv'), sep=',', header=None).values
    X_train, y_train = train_data[:, :-1], train_data[:, -1]
    X_test, y_test = test_data[:, :-1], test_data[:, -1]
  ## ========================================
  ## ========================================
  
  ## ========================================
  # == Get Cifar100 dataset =================
  if args.dataset == 'cifar100':
    # train_data = pd.read_csv(os.path.join(args.data_path, 'cifar100_train.csv'), sep=',', header=None).values
    # test_data = pd.read_csv(os.path.join(args.data_path, 'cifar100_test.csv'), sep=',', header=None).values
    cifar100_train = unpickle(os.path.join(args.data_path, 'cifar100_train'))
    cifar100_test = unpickle(os.path.join(args.data_path, 'cifar100_test'))
 
    X_train = np.array(cifar100_train[b'data'])
    y_train = np.array(cifar100_train[b'fine_labels'])
    X_test = np.array(cifar100_test[b'data'])
    y_test = np.array(cifar100_test[b'fine_labels'])
    
  ## ========================================
  ## ========================================

  ### === Permuted dataset (Vector) ===============
  # if args.dataset in ['pmnist', 'pfmnist']:
  #   for t in range(args.n_tasks):
  #     perm = torch.arange(X_train.shape[-1]) if t == 0 else torch.randperm(X_train.shape[-1])
  #     # inv_perm = torch.zeros_like(perm)
  #     # for i in range(perm.size(0)):
  #     #   inv_perm[perm[i]] = i
  #     train_data = np.concatenate((X_train[:, perm], y_train.reshape(-1, 1)), axis=1)
  #     test_data = np.concatenate((X_test[:, perm], y_test.reshape(-1, 1)), axis=1)
  #     pd.DataFrame(train_data).to_csv(os.path.join(args.saved, args.train_path, 'task_{}.csv'.format(t)),
  #       header=None,
  #       index=None)
  #     pd.DataFrame(test_data).to_csv(os.path.join(args.saved, args.test_path, 'task_{}.csv'.format(t)),
  #       header=None,
  #       index=None)

  ### === Permuted dataset (Image) ===============
  if args.dataset in ['pmnist', 'pfmnist']:
    for t in range(args.n_tasks):
      tensor_view = (1, 28, 28)
      xtrain_tensor = torch.tensor(X_train, dtype=torch.float).view((X_train.shape[0], *tensor_view))
      xtest_tensor = torch.tensor(X_test, dtype=torch.float).view((X_test.shape[0], *tensor_view))
      
      ## col or row permutetion for each task
      if t % 2 == 0: # for even task -> col permuted
        perm = torch.arange(xtrain_tensor.shape[3]) if t == 0 else torch.randperm(xtrain_tensor.shape[3])
        perm_xtrain = xtrain_tensor[:, :, :, perm].clone().detach().numpy()
        perm_xtest = xtest_tensor[:, :, :, perm].clone().detach().numpy()
      else: # for odd task -> row permuted
        perm = torch.randperm(xtrain_tensor.shape[2])
        perm_xtrain = xtrain_tensor[:, :, perm, :].clone().detach().numpy()
        perm_xtest = xtest_tensor[:, :, perm, :].clone().detach().numpy()

      ## both permutetions 
      # first_perm = torch.arange(xtrain_tensor.shape[3]) if t == 0 else torch.randperm(xtrain_tensor.shape[3])
      # perm_xtrain = xtrain_tensor[:, :, :, first_perm]
      # perm_xtest = xtest_tensor[:, :, :, first_perm]
      # second_perm = torch.arange(xtrain_tensor.shape[2]) if t == 0 else torch.randperm(xtrain_tensor.shape[2])
      # perm_xtrain = perm_xtrain[:, :, second_perm, :].clone().detach().numpy()
      # perm_xtest = perm_xtest[:, :, second_perm, :].clone().detach().numpy()

      # save dataset
      perm_xtrain = perm_xtrain.reshape(perm_xtrain.shape[0], -1)
      train_data = np.concatenate((perm_xtrain, y_train.reshape(-1, 1)), axis=1)
      perm_xtest = perm_xtest.reshape(perm_xtest.shape[0], -1)
      test_data = np.concatenate((perm_xtest, y_test.reshape(-1, 1)), axis=1)
      pd.DataFrame(train_data).to_csv(os.path.join(args.saved, args.train_path, 'task_{}.csv'.format(t)),
        header=None,
        index=None)
      pd.DataFrame(test_data).to_csv(os.path.join(args.saved, args.test_path, 'task_{}.csv'.format(t)),
        header=None,
        index=None)
      
      print('task {} dataset done!'.format(t))

  ### === Rotated dataset ========================
  elif args.dataset in ['rmnist', 'rfmnist']:
    
    angles = [0, 10, 20, 30, 40]
    for t in range(args.n_tasks):
      
      if t == 0: 
        train_data = np.concatenate((X_train, y_train.reshape(-1, 1)), axis=1)
        test_data = np.concatenate((X_test, y_test.reshape(-1, 1)), axis=1)
        pd.DataFrame(train_data).to_csv(os.path.join(args.saved, args.train_path, 'task_{}.csv'.format(t)),
          header=None,
          index=None)
        pd.DataFrame(test_data).to_csv(os.path.join(args.saved, args.test_path, 'task_{}.csv'.format(t)),
          header=None,
          index=None)
      
      else:
        tensor_view = (1, 28, 28)
        rotated_xtrain_list = []
        rotated_xtest_list = []
        
        for img in X_train:
          x_tensor = (torch.tensor(img, dtype=torch.float) / 255).view(tensor_view)
          pil_img = transforms.ToPILImage()(x_tensor)
          rotated_pil_img = transforms.functional.rotate(pil_img, angles[t])
          rotated_img = transforms.ToTensor()(rotated_pil_img)
          rotated_img = rotated_img*255.0

          rotated_xtrain_list.append(rotated_img)
        rotated_xtrain = torch.stack(rotated_xtrain_list)
        rotated_xtrain = rotated_xtrain.clone().detach().numpy()
        rotated_xtrain = rotated_xtrain.reshape(rotated_xtrain.shape[0], -1)
        train_data = np.concatenate((rotated_xtrain, y_train.reshape(-1, 1)), axis=1)
        pd.DataFrame(train_data).to_csv(os.path.join(args.saved, args.train_path, 'task_{}.csv'.format(t)),
          header=None,
          index=None)
        
        for img in X_test:
          x_tensor = (torch.tensor(img, dtype=torch.float) / 255).view(tensor_view)
          pil_img = transforms.ToPILImage()(x_tensor)
          rotated_pil_img = transforms.functional.rotate(pil_img, angles[t])
          rotated_img = transforms.ToTensor()(rotated_pil_img)
          rotated_img = rotated_img*255.0

          rotated_xtest_list.append(rotated_img)
        rotated_xtest = torch.stack(rotated_xtest_list)
        rotated_xtest = rotated_xtest.clone().detach().numpy()
        rotated_xtest = rotated_xtest.reshape(rotated_xtest.shape[0], -1)
        test_data = np.concatenate((rotated_xtest, y_test.reshape(-1, 1)), axis=1)
        pd.DataFrame(test_data).to_csv(os.path.join(args.saved, args.test_path, 'task_{}.csv'.format(t)),
          header=None,
          index=None)

      print('task {} dataset done!'.format(t))

  ### === Split dataset ==========================
  else:
    train_data = np.concatenate((X_train, y_train.reshape(-1, 1)), axis=1)
    test_data = np.concatenate((X_test, y_test.reshape(-1, 1)), axis=1)

    if args.dataset == 'cifar100': cpt = int(100 / args.n_tasks)
    else: cpt = int(10 / args.n_tasks)
    
    for t in range(args.n_tasks):
      c1 = t * cpt
      c2 = (t + 1) * cpt
      i_tr = np.where((y_train >= c1) & (y_train < c2))[0]
      i_te = np.where((y_test >= c1) & (y_test < c2))[0]
      
      pd.DataFrame(train_data[i_tr]).to_csv(os.path.join(args.saved, args.train_path, 'task_{}.csv'.format(t)),
        header=None,
        index=None
      )
      pd.DataFrame(test_data[i_te]).to_csv(os.path.join(args.saved, args.test_path, 'task_{}.csv'.format(t)),
        header=None,
        index=None
      )
      print('task {} dataset done!'.format(t))
    