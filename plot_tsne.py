
import torch
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader

import dataset
import models

def run_plot():
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  number_layers = 6
  growth_rate = 12
  drop_rate = 0.0
  model_path = './running/fm/model.pkl'
  pts_path = './running/fm/prototypes.pkl'

  trainset = dataset.FashionMnist(for_plot=True)
  train_loader = DataLoader(dataset=trainset, batch_size=3000)
 
  # model = models.DenseNet(device=torch.device(device),
  #                         tensor_view=trainset.tensor_view,
  #                         number_layers=number_layers,
  #                         growth_rate=growth_rate,
  #                         drop_rate=drop_rate)
  model = models.CNNEncoder(device=torch.device(device))
  model.load(model_path)

  ### ======================================
  ### == Feature space visualization =======
  ### ======================================
  print('=== Feature-Space visualization (t-SNE) ===')
  sns.set(rc={'figure.figsize':(11.7,8.27)})
  palette = sns.color_palette("bright", 10)

  with torch.no_grad():
    samples, labels = next(iter(train_loader))
    samples, labels = samples.to(device), labels.to(device)
    feature, out = model(samples)
    feature = feature.cpu().detach().numpy()
    # batches = batches.view(batches.size(0), -1)
    # batches = batches.cpu().detach().numpy()
    labels = labels.cpu().detach().numpy()

  tsne = TSNE()
  X_embedded = tsne.fit_transform(feature)
  sns.scatterplot(x=X_embedded[:,0], y=X_embedded[:,1], hue=labels, legend='full', palette=palette)
  plt.show()
  ### ======================================

if __name__ == '__main__':
  run_plot()