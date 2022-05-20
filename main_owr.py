import torch
from torch import optim
from torch.utils.data import DataLoader
import argparse
import logging
import time
import numpy as np
import os
import pandas as pd

import tool
import models
import dataset
from config import Config
from plot_tsne import run_plot
from evaluation import in_stream_evaluation
from added_components.memory_selector import IncrementalMemory
from plot.feature_space_visualization import set_novel_label, visualization


def owr(memory, config):
  logger = logging.getLogger(__name__)
  
  if config.dataset == 'mnist':
    n_inputs, n_feature, n_outputs = 784, 100, 10
    # net = models.MLP(n_inputs, n_feature, n_outputs, config)
    net = models.Conv_4(config)
  else:
    net = models.Conv_4(config)
  net.to(config.device)

  criterion = models.CPELoss(gamma=config.gamma, tao=config.tao, b=config.b, beta=config.beta, lambda_=config.lambda_)
  optimizer = optim.SGD(net.parameters(), lr=config.learning_rate, momentum=0.9)

  #todo
  soft = True
  use_log = True
  logger.info("Soft: {}, Use_log: {}".format(soft, use_log))

  prototypes = models.Prototypes(threshold=config.threshold)
  # load saved prototypes
  try:
      prototypes.load(config.prototypes_path)
  except FileNotFoundError:
      pass
  else:
      logger.info("load prototypes from file '%s'.", config.prototypes_path)
  logger.info("original prototype count: %d", len(prototypes))

  detector = None

  def train(train_dataset, plot=False):
    logger.info('---------------- train ----------------')
    
    # == Train with pt loss =================
    dataloader = DataLoader(dataset=train_dataset, batch_size=1, shuffle=True)
    for epoch in range(config.epoch_number):
        logger.info('---------------- epoch: %d ----------------', epoch + 1)
        logger.info("threshold: %.4f, gamma: %.4f, tao: %.4f, b: %.4f", config.threshold, config.gamma, config.tao, config.b)
        logger.info("prototypes count before training: %d", len(prototypes))

        net.train()
        for i, (feature, label) in enumerate(dataloader):
            feature, label = feature.to(net.device), label.to(net.device)
            optimizer.zero_grad()
            out, feature = net(feature)
            loss, distance = criterion(feature, out, label, prototypes)

            loss.backward()
            optimizer.step()
            if i == 0 or (i+1) % 500 == 0:
                logger.debug("[train %d, %d] %7.4f %7.4f", epoch + 1, i + 1, loss.item(), distance)

        logger.info("prototypes count after training: %d", len(prototypes))
        prototypes.update()
        logger.info("prototypes count after update: %d", len(prototypes))

    net.save(config.net_path)
    logger.info("net has been saved.")
    prototypes.save(config.prototypes_path)
    logger.info("prototypes has been saved.")

    if plot:
        run_plot()

    intra_distances = []
    with torch.no_grad():
        net.eval()
        for i, (feature, label) in enumerate(dataloader):
            feature, label = feature.to(net.device), label.item()
            out, feature = net(feature)
            closest_prototype, distance = prototypes.closest(feature, label)
            intra_distances.append((label, distance))

    novelty_detector = models.Detector(intra_distances, train_dataset.label_set, config.std_coefficient)
    logger.info("distance average: %s", novelty_detector.average_distances)
    logger.info("distance std: %s", novelty_detector.std_distances)
    logger.info("detector threshold: %s", novelty_detector.thresholds)
    novelty_detector.save(config.detector_path)
    logger.info("detector has been saved.")
    # print(train_dataset.label_set)
    # time.sleep(5)

    return novelty_detector

  def test(test_dataset, novelty_detector):
      logger.info('---------------- test ----------------')
      dataloader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

      logger.info("known labels: %s", novelty_detector.known_labels)
      logger.info("distance average: %s", novelty_detector.average_distances)
      logger.info("distance std: %s", novelty_detector.std_distances)
      logger.info("detector threshold: %s", novelty_detector.thresholds)

      detection_results = []

      with torch.no_grad():
        net.eval()
        for i, (feature, label) in enumerate(dataloader):
          feature, label = feature.to(net.device), label.item()
          out, feature = net(feature)
          predicted_label, distance = models.predict(feature, prototypes)
          prob = models.probability(feature, predicted_label, prototypes, gamma=config.gamma)
          detected_novelty = novelty_detector(predicted_label, distance)
          real_novelty = label not in novelty_detector.known_labels

          detection_results.append((label, predicted_label, real_novelty, detected_novelty))
          
          if (i+1) % 1000 == 0:
            logger.debug("[test %5d]: %d, %d, %7.4f, %7.4f, %5s, %5s",
                        i + 1, label, predicted_label, prob, distance, real_novelty, detected_novelty)

      tp, fp, fn, tn, cm, acc, acc_all, NCA = novelty_detector.evaluate(detection_results)
      precision = tp / (tp + fp + 1)
      recall = tp / (tp + fn + 1)

      M_new = fn / (tp + fn + 1e-8)
      F_new = fp / (fp + tn + 1e-8)

      logger.info("true positive: %d", tp)
      logger.info("false positive: %d", fp)
      logger.info("false negative: %d", fn)
      logger.info("true negative: %d", tn)
      logger.info("precision: %7.4f", precision)
      logger.info("recall: %7.4f", recall)
      # print("M_new: %7.4f"% M_new)
      # print("F_new: %7.4f"% F_new)
      # print("Accuracy: %7.4f"% acc)
      # print("Accuracy All: %7.4f"% acc_all)
      # print("New Class Accuracy: %7.4f"% NCA)
      print("Evaluation: %7.2f, %7.2f, %7.2f"%(acc*100, M_new*100, F_new*100))
      logger.info("confusion matrix: \n%s", cm)

  for current_task in range(config.n_tasks):  
    ### === Task data loading =====================
    task_data = pd.read_csv(
                  os.path.join(config.split_train_path, "task_{}.csv".format(current_task)),
                  sep=',', header=None).values 
    print('task_data: {}'.format(task_data.shape))

    ### === train data with memory ================
    if current_task != 0:
      replay_mem = memory()
      train_data = np.concatenate((task_data, replay_mem))
      print('replay_mem: {}'.format(replay_mem.shape))
      print('train_data(new): {}'.format(train_data.shape))
    else:
      train_data = task_data
      print('train_data: {}'.format(train_data.shape))
    
    ### === test data =============================
    all_data = []
    task_range = config.n_tasks if current_task == config.n_tasks-1 else current_task+2
    for prev_task in range(task_range):
      task_test_data = pd.read_csv(
        os.path.join(config.split_test_path, "task_{}.csv".format(prev_task)),
        sep=',', header=None).values
      all_data.append(task_test_data)

    test_data = np.concatenate(all_data)

    ### === Create dataset ========================
    if config.dataset  == 'mnist':
      trainset = dataset.Mnist(dataset=train_data)
      testset = dataset.Mnist(dataset=test_data)
    elif config.dataset == 'fmnist':
      trainset = dataset.FashionMnist(dataset=train_data)
      testset = dataset.FashionMnist(dataset=test_data)
    elif config.dataset == 'cifar10':
      trainset = dataset.Cifar10(dataset=train_data)
      testset = dataset.Cifar10(dataset=test_data)
    
    logger.info("trainset size: %d", len(trainset))
    logger.info("testset size: %d", len(testset))

    
    novelty_detector = train(trainset, plot=False)
    test(testset, novelty_detector)

    ## == Plot ==================
    print('-- Ploting ... ----')
    known_labels = set(np.arange((current_task+1)*2))
    n_label = len(known_labels) if current_task == config.n_tasks-1 else len(known_labels)+1

    new_test_data = set_novel_label(known_labels, config, data=test_data)
    visualization(
      net,
      new_test_data,
      config,
      'tsne_taks_{}'.format(current_task),
      n_label=n_label
    )
    print('--- Plot done! ---')

    ### === Update memoty ===================
    memory.update(task_data)

def main(args):
  config = Config(args)
  logger = logging.getLogger(__name__)

  def setup_logger(level=logging.DEBUG, filename=None):
    logger.setLevel(level)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[%(asctime)s] %(message)s')
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if filename is not None:
      file_handler = logging.FileHandler(filename=filename, mode='a')
      file_handler.setLevel(logging.INFO)
      file_handler.setFormatter(formatter)
      logger.addHandler(file_handler)

    logger.debug("logger '%s' has been setup.", __name__)

  setup_logger(level=logging.DEBUG, filename=config.log_path)

  logger.info("****************************************************************")
  logger.info("%s", config)

  ## == Add memory ===============
  memory = IncrementalMemory(
    selection_type=config.mem_sel_type, 
    total_size=config.mem_total_size,
    per_class=config.mem_per_class,
    selection_method=config.mem_sel_method)

  start_time = time.time()

  if config.type == 'owr':
    owr(memory, config=config)

  logger.info("-------------------------------- %.3fs --------------------------------", time.time() - start_time)

  ### plot 


if __name__ == '__main__':
  arg_parser = argparse.ArgumentParser(prog="CPE")

  argument_group = arg_parser.add_argument_group(title='arguments')
  argument_group.add_argument('-t', '--type', type=str, help="Running type.", choices=['ce', 'cpe', 'stream', 'owr'], required=True)
  argument_group.add_argument('-d', '--dir', type=str, help="Running directory path.", required=True)
  argument_group.add_argument('--dataset', type=str, help="Dataset.", choices=dataset.DATASETS, required=True)
  argument_group.add_argument('--device', type=str, help="Torch device.", default='cpu', choices=['cpu', 'cuda:0', 'cuda:1', 'cuda:2', 'cuda:3'])
  argument_group.add_argument('-c', '--clear', help="Clear running path.", action="store_true")

  training_group = arg_parser.add_argument_group(title='training arguments')
  training_group.add_argument('--train', help="Whether do training process.", action="store_true")
  training_group.add_argument('-p', '--period', type=int, help="Run the whole process for how many times.", default=1)
  training_group.add_argument('-e', '--epoch', type=int, help="Epoch Number.", default=1)
  
  ## === we added =============
  training_group.add_argument('--n_tasks', type=int, default=5, help='')
  training_group.add_argument('--dropout', type=float, default=0.2, help='')
  training_group.add_argument('--hidden_dims', type=int, default=128, help='') #768
  training_group.add_argument('--seed', type=int, default=2, help='')

  # memory
  training_group.add_argument('--mem_sel_type', type=str, default='fixed_mem', choices=['fixed_mem', 'pre_class'], help='')
  training_group.add_argument('--mem_total_size', type=int, default=2000, help='')
  training_group.add_argument('--mem_per_class', type=int, default=100, help='')
  training_group.add_argument('--mem_sel_method', type=str, default='rand', choices=['rand', 'soft_rand'], help='')
  
  stream_group = arg_parser.add_argument_group(title='stream arguments')
  stream_group.add_argument('-r', '--rate', type=float, help='Novelty buffer sample rate.', default=0.3)

  parsed_args = arg_parser.parse_args()

  ## == params ===========================
  parsed_args.split_train_path = 'data/split_{}/train'.format(parsed_args.dataset)
  parsed_args.split_test_path = 'data/split_{}/test'.format(parsed_args.dataset)

  ## == Apply seed =======================
  np.random.seed(parsed_args.seed)
  torch.manual_seed(parsed_args.seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False
  torch.cuda.manual_seed_all(parsed_args.seed)

  main(parsed_args)
