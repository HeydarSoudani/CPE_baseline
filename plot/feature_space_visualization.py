import torch
from torch.utils.data import DataLoader
import os
import numpy as np
import seaborn as sns
from pandas import read_csv
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from hausdorff import hausdorff_distance

import dataset
from samplers.pt_sampler import PtSampler


def set_novel_label(known_labels, args, data=[]):
  print(data.shape)
  
  if data == []:
    data = read_csv(
      os.path.join(args.data_path, args.stream_file),
      sep=',', header=None).values

  for idx, item in enumerate(data):
    label = item[-1]
    # print(known_labels)
    # print(label)
    if label not in known_labels:
      data[idx, -1] = 100

  return data


def tsne_plot(features, labels, file_name='tsne', n_color=6):
  # tsne = TSNE()
  # X_embedded = tsne.fit_transform(features)

  # sns.set(rc={'figure.figsize':(11.7,8.27)})
  # palette = sns.color_palette("bright", n_color)
  # sns.scatterplot(
  #   x=X_embedded[:,0],
  #   y=X_embedded[:,1],
  #   hue=labels,
  #   legend='full',
  #   palette=palette
  # )

  # plt.savefig('{}.png'.format(file_name))
  # # plt.show()
  # plt.clf()


  colors = [
    'royalblue', 'forestgreen',
    'darkorchid', 'brown',
    'gold', 'red',
    'darkcyan', 'greenyellow',
    'peru',
    'hotpink'
  ]
  
  tsne = TSNE()
  X_embedded = tsne.fit_transform(features)
  
  # fig, ax = plt.subplots()
  plt_colors = np.array([colors[9] if i==100 else colors[i] for i in labels])
  # plt_labels = np.array(['Novel' if i==100 else i for i in labels])
  plt_labels = np.unique(np.array(labels))
  plt.tick_params(
    left=False,
    labelleft=False,
    bottom = False,
    labelbottom=False
  )

  for idx, label in enumerate(plt_labels):  
    plt.scatter(
      X_embedded[np.where(labels==label)[0], 0],
      X_embedded[np.where(labels==label)[0], 1],
      marker='o',
      c= colors[9] if label==100 else colors[idx],
      label= 'Novel' if label==100 else str(idx),
    )

  plt.legend(
    loc="upper right",
    title="Classes",
    fontsize=9
  )

  plt.savefig('{}.png'.format(file_name))
  plt.clf()


def pca_plot(features, labels, file_name='pca'):
  pca = PCA(n_components=2)
  X_embedded = pca.fit_transform(features)

  sns.set(rc={'figure.figsize':(11.7,8.27)})
  palette = sns.color_palette("bright", 6)
  sns.scatterplot(
    x=X_embedded[:,0],
    y=X_embedded[:,1],
    hue=labels,
    legend='full',
    palette=palette
  )

  plt.savefig('{}.png'.format(file_name))
  plt.show()


def hausdorff_calculate(features, labels):
  features_novel = features[np.where(labels == 100)[0]]
  features_known = features[np.where(labels != 100)[0]]
  
  dist = hausdorff_distance(features_novel, features_known, distance="cosine")
  print('Hausdorff distance is {}'.format(dist))


def visualization(model, data, config, filename, n_label=6):  
  
  ### === Create dataset ========================
  if config.dataset  == 'mnist':
    data_set = dataset.Mnist(dataset=data)
  elif config.dataset == 'fmnist':
    data_set = dataset.FashionMnist(dataset=data)
  elif config.dataset == 'cifar10':
    data_set = dataset.Cifar10(dataset=data)
  
  print(n_label)
  print(data_set.label_set)
  print(len(data_set))
  sampler = PtSampler(
    data_set,
    n_way=n_label,
    n_shot=500,
    n_query=0,
    n_tasks=1
  )
  dataloader = DataLoader(
    data_set,
    batch_sampler=sampler,
    num_workers=1,
    pin_memory=True,
    collate_fn=sampler.episodic_collate_fn,
  )
  
  ### == Plot ============================
  with torch.no_grad():
    batch = next(iter(dataloader))
    support_images, support_labels, _, _ = batch
    support_images = support_images.reshape(-1, *support_images.shape[2:])
    support_labels = support_labels.flatten()
    support_images = support_images.to(config.device)
    support_labels = support_labels.to(config.device)

    outputs, features = model.forward(support_images)
    features = features.cpu().detach().numpy()
    support_labels = support_labels.cpu().detach().numpy()

    # for feature in features:
    #   print(feature)
    # print(support_labels)
    # print(features.shape)
    # print(support_labels.shape)
  # features += 1e-12

  tsne_plot(features, support_labels, file_name=filename, n_color=n_label)
  # pca_plot(features, support_labels, file_name='pca_last')
  hausdorff_calculate(features, support_labels)

