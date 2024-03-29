import argparse
import logging
import time
import torch
import models
import dataset
import tool
from config import Config
from torch import optim
from torch.utils.data import DataLoader
from plot_tsne import run_plot
import numpy as np

from evaluation import in_stream_evaluation

# from plot.feature_space_visualization import set_novel_label, visualization

def stream(config, trainset, streamset):
    logger = logging.getLogger(__name__)

    if config.dataset == 'mnist':
      n_inputs, n_feature, n_outputs = 784, 100, 10
      net = models.MLP(n_inputs, n_feature, n_outputs, config)
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
                if i == 0 or (i+1) % 100 == 0:
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

        M_new = fn / (tp + fn + 1)
        F_new = fp / (fp + tn + 1)

        logger.info("true positive: %d", tp)
        logger.info("false positive: %d", fp)
        logger.info("false negative: %d", fn)
        logger.info("true negative: %d", tn)
        logger.info("precision: %7.4f", precision)
        logger.info("recall: %7.4f", recall)
        print("M_new: %7.4f"% M_new)
        print("F_new: %7.4f"% F_new)
        print("Accuracy: %7.4f"% acc)
        print("Accuracy All: %7.4f"% acc_all)
        print("New Class Accuracy: %7.4f"% NCA)
        logger.info("confusion matrix: \n%s", cm)

    def stream_train(train_dataset, stream_dataset):
        logger.info('---------------- stream train ----------------')

        logger.info('---------------- initial train ---------------')
        novelty_detector = train(trainset, plot=False)
        logger.info('---------------- initial test ----------------')
        test(stream_dataset, novelty_detector)

        ### Plot t-SNE
        # base_labels = train_dataset.label_set
        # stream_data_novel = set_novel_label(base_labels, args)
        # visualization(net, stream_data_novel, args, device, filename=args.vis_filename)

        f = open('output.txt', 'w')
        labels = stream_dataset.labels
        label_set = stream_dataset.label_set
        for label in label_set:
          start_point = np.where(labels == label)[0][0]
          print('Class {} starts at {}'.format(label, start_point))
          f.write("[Class %5d], Start point: %5d \n" % (label, start_point))

        novelty_dataset = dataset.NoveltyDataset(train_dataset)
        iter_streamloader = enumerate(DataLoader(dataset=stream_dataset, batch_size=1, shuffle=True))
        buffer = []
        detection_results = []
        last_idx = 0

        for i, (image, label) in iter_streamloader:
          sample = (image.squeeze(dim=0), label.squeeze(dim=0))
          with torch.no_grad():
            net.eval()
            image, label = image.to(net.device), label.item()
            out, feature = net(image)
            predicted_label, distance = models.predict(feature, prototypes)
            prob = models.probability(feature, predicted_label, prototypes, gamma=config.gamma)
            detected_novelty = novelty_detector(predicted_label, distance)
            real_novelty = label not in novelty_detector.known_labels

            # append to 
            detection_results.append((label, predicted_label, real_novelty, detected_novelty))

          if detected_novelty:
            # sample = (image.squeeze(dim=0), label.squeeze(dim=0), feature)
            buffer.append(sample)

          if (i+1) % 500 == 0:
            logger.debug("[stream %5d]: %d, %d, %7.4f, %7.4f, %5s, %5s, %4d",
                          i + 1, label, predicted_label, prob, distance, real_novelty, detected_novelty, len(buffer))

          if len(buffer) == 1000:
            ## === in stream evaluation ========== 
            sample_num = i-last_idx

            CwCA, M_new, F_new, cm, acc_per_class = in_stream_evaluation(
              detection_results, novelty_detector.known_labels)
            print("[On %5d samples]: %7.4f, %7.4f, %7.4f" %
                  (sample_num, CwCA, M_new, F_new))
            print("confusion matrix: \n%s\n" % cm)
            print("acc per class: %s\n" % acc_per_class)
            f.write("[In sample %2d], [On %5d samples]: %7.4f, %7.4f, %7.4f \n" %
                    (i, sample_num, CwCA, M_new, F_new))
            f.write("acc per class: %s\n" % acc_per_class)
            # =
            detection_results.clear()
            last_idx = i
            

            logger.info("novelty dataset size before extending: %d", len(novelty_dataset))
            # todo try different sample methods by YW.
            # novelty_dataset.extend(buffer, config.novelty_buffer_sample_rate)
            novelty_dataset.extend_by_select(buffer, config.novelty_buffer_sample_rate, prototypes, net, soft, use_log)

            logger.info("novelty dataset size after extending: %d", len(novelty_dataset))
            logger.info('---------------- incremental train ----------------')
            novelty_detector = train(novelty_dataset)

            buffer.clear()
        
        # == Last evaluation ========================
        sample_num = i-last_idx
        CwCA, M_new, F_new, cm, acc_per_class = in_stream_evaluation(
          detection_results, novelty_detector.known_labels)
        print("[On %5d samples]: %7.4f, %7.4f, %7.4f" %
              (sample_num, CwCA, M_new, F_new))
        print("confusion matrix: \n%s\n" % cm)
        print("acc per class: %s\n" % acc_per_class)
        f.write("[In sample %2d], [On %5d samples]: %7.4f, %7.4f, %7.4f \n" %
                (i, sample_num, CwCA, M_new, F_new))
        f.write("acc per class: %s\n" % acc_per_class)
        
        # =
        detection_results.clear()
        last_idx = i

        return novelty_detector

    if config.train:
        for period in range(config.period):
            logger.info('---------------- period: %d ----------------', period + 1)
            detector = stream_train(trainset, streamset)
            test(streamset, detector)
    else:
        test(streamset, detector)

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

  if config.dataset  == 'mnist':
    trainset = dataset.Mnist(train=True)
    testset = dataset.Mnist(train=False)
  elif config.dataset == 'fmnist':
    trainset = dataset.FashionMnist(train=True)
    testset = dataset.FashionMnist(train=False)
  elif config.dataset == 'cifar10':
    trainset = dataset.Cifar10(train=True)
    testset = dataset.Cifar10(train=False)
  elif config.dataset == 'svhn':
    trainset = dataset.SVHN(train=True)
    testset = dataset.SVHN(train=False)
  elif config.dataset == 'cinic':
    trainset = dataset.Cinic(train=True)
    testset = dataset.Cinic(train=False)
  else:
    raise ValueError("Dataset '{}' not found.".format(config.dataset))

  logger.info("****************************************************************")
  logger.info("%s", config)
  logger.info("trainset size: %d", len(trainset))
  logger.info("testset size: %d", len(testset))

  start_time = time.time()

  if config.type == 'stream':
    stream(config=config, trainset=trainset, streamset=testset)

  logger.info("-------------------------------- %.3fs --------------------------------", time.time() - start_time)

  ### plot 


if __name__ == '__main__':
  arg_parser = argparse.ArgumentParser(prog="CPE")

  argument_group = arg_parser.add_argument_group(title='arguments')
  argument_group.add_argument('-t', '--type', type=str, help="Running type.", choices=['ce', 'cpe', 'stream'], required=True)
  argument_group.add_argument('-d', '--dir', type=str, help="Running directory path.", required=True)
  argument_group.add_argument('--dataset', type=str, help="Dataset.", choices=dataset.DATASETS, required=True)
  argument_group.add_argument('--device', type=str, help="Torch device.", default='cpu', choices=['cpu', 'cuda:0', 'cuda:1', 'cuda:2', 'cuda:3'])
  argument_group.add_argument('-c', '--clear', help="Clear running path.", action="store_true")

  training_group = arg_parser.add_argument_group(title='training arguments')
  training_group.add_argument('--train', help="Whether do training process.", action="store_true")
  training_group.add_argument('-p', '--period', type=int, help="Run the whole process for how many times.", default=1)
  training_group.add_argument('-e', '--epoch', type=int, help="Epoch Number.", default=1)
  
  # we added
  training_group.add_argument('--dropout', type=float, default=0.2, help='')
  training_group.add_argument('--hidden_dims', type=int, default=128, help='') #768
  training_group.add_argument('--seed', type=int, default=2, help='')

  stream_group = arg_parser.add_argument_group(title='stream arguments')
  stream_group.add_argument('-r', '--rate', type=float, help='Novelty buffer sample rate.', default=0.3)

  parsed_args = arg_parser.parse_args()

  ## == Apply seed =======================
  np.random.seed(parsed_args.seed)
  torch.manual_seed(parsed_args.seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False
  torch.cuda.manual_seed_all(parsed_args.seed)

  main(parsed_args)
