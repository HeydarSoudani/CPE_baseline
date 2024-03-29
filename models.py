import torch
import torch.nn as nn
import numpy as np
from math import ceil
import math
from sklearn.metrics import accuracy_score, confusion_matrix

compute_distance = nn.PairwiseDistance(p=2, eps=1e-6)
compute_multi_distance = nn.PairwiseDistance(p=2, eps=1e-6, keepdim=True)


def probability(feature, label, prototypes, gamma=0.1):
    distances = compute_multi_distance(feature, prototypes.cat(label))
    prob = (-gamma * distances.pow(2)).exp().sum()
    # prob = (-self.gamma * distances).exp().sum()

    distances = compute_multi_distance(feature, prototypes.cat())
    one = (-gamma * distances.pow(2)).exp().sum()
    # one = (-self.gamma * distances).exp().sum()

    prob = prob / one if one > 0 else prob + 1e-6

    return prob


def predict(feature, prototypes):
    prototype, distance = prototypes.closest(feature)
    predicted_label = prototype.label

    return predicted_label, distance


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, drop_rate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True)
        self.dropout = nn.Dropout(p=drop_rate, inplace=False)

    def forward(self, x):
        out = self.dropout(self.conv1(self.relu(self.bn1(x))))
        return torch.cat([x, out], dim=1)


class BottleneckBlock(nn.Module):
    def __init__(self, in_channels, out_channels, drop_rate=0.0):
        super(BottleneckBlock, self).__init__()

        inter_channels = out_channels * 4

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, inter_channels, kernel_size=1, stride=1, padding=0, bias=True)

        self.bn2 = nn.BatchNorm2d(inter_channels)
        self.conv2 = nn.Conv2d(inter_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True)

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=drop_rate, inplace=False)

    def forward(self, x):
        out = self.dropout(self.conv1(self.relu(self.bn1(x))))
        out = self.dropout(self.conv2(self.relu(self.bn2(out))))
        return torch.cat([x, out], dim=1)


class TransitionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, drop_rate=0.0):
        super(TransitionBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.dropout = nn.Dropout(p=drop_rate, inplace=False)
        self.pooling = nn.AvgPool2d(kernel_size=2, ceil_mode=True)

    def forward(self, x):
        return self.pooling(self.dropout(self.conv1(self.relu(self.bn1(x)))))


class DenseBlock(nn.Module):
    def __init__(self, number_layers, in_channels, block, growth_rate, drop_rate):
        super(DenseBlock, self).__init__()

        layers = []

        for i in range(number_layers):
            layers.append(block(in_channels=in_channels + i * growth_rate, out_channels=growth_rate, drop_rate=drop_rate))

        self.layer = nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)

class DenseNet(nn.Module):
    def __init__(self, device, tensor_view, number_layers=6, growth_rate=12, reduction=2, bottleneck=True, drop_rate=0.0):
        super(DenseNet, self).__init__()

        assert len(tensor_view) == 3

        channels = 2 * growth_rate

        if bottleneck:
            block = BottleneckBlock
        else:
            block = BasicBlock

        # 1st conv before any dense block
        self.conv1 = nn.Conv2d(in_channels=tensor_view[0], out_channels=channels, kernel_size=3, stride=1, padding=1, bias=True)

        # 1st block
        self.block1 = DenseBlock(number_layers, channels, block, growth_rate, drop_rate)
        channels = channels + number_layers * growth_rate
        self.trans1 = TransitionBlock(channels, channels // reduction, drop_rate)
        channels = channels // reduction

        # 2nd block
        self.block2 = DenseBlock(number_layers, channels, block, growth_rate, drop_rate)
        channels = channels + number_layers * growth_rate
        self.trans2 = TransitionBlock(channels, channels // reduction, drop_rate)
        channels = channels // reduction

        # 3rd block
        self.block3 = DenseBlock(number_layers, channels, block, growth_rate, drop_rate)
        channels = channels + number_layers * growth_rate

        # global average pooling and classifier
        self.bn = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.pooling = nn.MaxPool2d(kernel_size=2, ceil_mode=True)

        self.fc1 = nn.Linear(channels * ceil(tensor_view[1] / 8) * ceil(tensor_view[2] / 8), 10000)
        self.fc2 = nn.Linear(10000, 1000)
        self.fc3 = nn.Linear(1000, 10)

        self.channels = channels
        self.tensor_view = tensor_view

        self.device = device
        self.to(device)

    @property
    def shape(self):
        return torch.Size((self.channels, ceil(self.tensor_view[1] / 8), ceil(self.tensor_view[2] / 8)))

    def forward(self, x):
        out = self.conv1(x)
        out = self.trans1(self.block1(out))
        out = self.trans2(self.block2(out))
        out = self.block3(out)
        out = self.relu(self.bn(out))
        out = self.pooling(out).view(1, -1)
        out = self.relu(self.fc1(out))
        feature = self.fc2(out)
        out = self.fc3(self.relu(feature))
        return feature, out

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        state_dict = torch.load(path)
        self.load_state_dict(state_dict)

class Conv_4(nn.Module):
    def __init__(self, args):
        super(Conv_4, self).__init__()

        self.device = args.device
        # self.to(self.device)

        if args.dataset in ['mnist', 'fmnist']:
            img_channels = 1	  	# 1
            self.last_layer = 1 	# 3 for 3-layers - 1 for 4-layers
        elif args.dataset in ['cifar10', 'cifar100']:
            img_channels = 3	  	# 3 
            self.last_layer = 2 	# 4 for 3-layers - 2 for 4-layers

        self.filters_length = 256    # 128 for 3-layers - 256 for 4-layers

        self.layer1 = nn.Sequential(
            nn.Conv2d(img_channels, 32, kernel_size=5, padding=2), #input: 28 * 28 * 3, output: 28 * 28 * 32
            # nn.ReLU(),
            nn.PReLU(),
            nn.Conv2d(32, 32, kernel_size=5, padding=2), #input: 28 * 28 * 3, output: 28 * 28 * 32
            nn.BatchNorm2d(32),
            nn.PReLU(),
            # nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),   #input: 28 * 28 * 32, output: 14 * 14 * 32
            nn.Dropout(args.dropout)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, padding=2), #input: 14 * 14 * 32, output: 14 * 14 * 64
            nn.PReLU(),
            # nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=5, padding=2), #input: 14 * 14 * 64, output: 14 * 14 * 64
            nn.BatchNorm2d(64),
            nn.PReLU(),
            # nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),   #input: 14 * 14 * 64, output: 7* 7 * 64
            nn.Dropout(args.dropout)
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1), #input: 7 * 7 * 64, output: 7 * 7 * 128
            nn.PReLU(),
            # nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), #input: 7 * 7 * 128, output: 7 * 7 * 128
            nn.BatchNorm2d(128),
            nn.PReLU(),
            # nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),   #input: 7 * 7 * 128, output: 3* 3 * 128
            nn.Dropout(args.dropout)
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1), #input: 3 * 3 * 128, output: 3 * 3 * 256
            nn.PReLU(),
            # nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), #input: 3*3*256, output: 3*3*256
            nn.BatchNorm2d(256),
            nn.PReLU(),
            # nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),   #input: 3*3*256, output: 1*1*256
            nn.Dropout(args.dropout)
        )

        self.ip1 = nn.Linear(self.filters_length*self.last_layer*self.last_layer, args.hidden_dims)
        self.preluip1 = nn.PReLU()
        self.dropoutip1 = nn.Dropout(args.dropout)
        self.classifier = nn.Linear(args.hidden_dims, 10)
        

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.view(-1, self.filters_length*self.last_layer*self.last_layer)

        features = self.preluip1(self.ip1(x))
        x = self.dropoutip1(features)
        logits = self.classifier(x)
        
        return logits, features

    def to(self, *args, **kwargs):
        self = super().to(*args, **kwargs)
        self.device = args[0] # store device
        
        self.layer1 = self.layer1.to(*args, **kwargs)
        self.layer2 = self.layer2.to(*args, **kwargs)
        self.layer3 = self.layer3.to(*args, **kwargs)
        self.layer4 = self.layer4.to(*args, **kwargs)

        self.ip1 = self.ip1.to(*args, **kwargs)
        self.preluip1 = self.preluip1.to(*args, **kwargs)
        self.dropoutip1 = self.dropoutip1.to(*args, **kwargs)
        self.classifier = self.classifier.to(*args, **kwargs)
        return self

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        state_dict = torch.load(path)
        self.load_state_dict(state_dict)


def Xavier(m):
  if m.__class__.__name__ == 'Linear':
    fan_in, fan_out = m.weight.data.size(1), m.weight.data.size(0)
    std = 1.0 * math.sqrt(2.0 / (fan_in + fan_out))
    a = math.sqrt(3.0) * std
    m.weight.data.uniform_(-a, a)
    if m.bias is not None:
      m.bias.data.fill_(0.0)

class MLP(nn.Module):
  def __init__(self, n_input, n_feature, n_output, args, bias=True):
    super(MLP, self).__init__()
    self.device = None

    self.hidden = nn.Sequential(nn.Linear(n_input, 100),
                                nn.ReLU(True),
                                nn.Dropout(args.dropout),
                                nn.Linear(100, n_feature),
                                nn.ReLU(True),
                                nn.Dropout(args.dropout))
    self.linear = nn.Linear(n_feature, n_output, bias=bias)
    self.hidden.apply(Xavier)
  
  def forward(self, samples):
    x = samples.view(samples.size(0), -1)
    features = self.hidden(x)
    outputs = self.linear(features)
    return outputs, features

  def to(self, *args, **kwargs):
    self = super().to(*args, **kwargs)
    self.device = args[0] # store device
    return self

  def save(self, path):
    torch.save(self.state_dict(), path)

  def load(self, path):
    state_dict = torch.load(path)
    self.load_state_dict(state_dict)

class Prototypes(object):
    class Prototype:
        def __init__(self, feature, label, weight=1):
            self.feature = feature
            self.label = label
            self.weight = weight

        @property
        def id(self):
            return id(self)

        def update(self, feature):
            weight = self.weight
            self.feature = (self.feature * weight + feature) / (weight + 1)
            self.weight = weight + 1

    def __init__(self, threshold):
        super().__init__()

        self._list = []
        self._dict = {}
        self.threshold = threshold

    @property
    def label_set(self):
        return set(self._dict)

    def assign(self, feature, label):
        """
        Assign the sample to a prototype.
        :param feature: Feature of the sample, the feature must be tensor detached from computation graph (.clone().detach()).
        :param label: Label of the sample
        :return: prototype, distance
        """
        if label not in self._dict:
            prototype = Prototypes.Prototype(feature, label)
            distance = 0.0
            self._append(prototype)
        else:
            closest_prototype, distance = self.closest(feature, label)

            if distance < 2 * self.threshold:
                prototype = closest_prototype
                prototype.update(feature)
            else:
                prototype = Prototypes.Prototype(feature, label)
                self._append(prototype)

        return prototype, distance

    def closest(self, feature, label=None):
        """
        find closest prototype from all prototypes or prototypes with same label
        :param feature:
        :param label:
        :return: closest prototype and distance
        """
        distances = compute_multi_distance(feature, self.cat(label))
        distance, index = distances.min(dim=0)
        distance = distance.item()
        closest_prototype = self[index] if label is None else self[label, index]

        return closest_prototype, distance

    def _append(self, prototype):
        
        # Orginal
        self._list.append(prototype)
        if prototype.label not in self._dict:
            self._dict[prototype.label] = []
        self._dict[prototype.label].append(prototype)
        
        # For one prototype
        # self._dict[prototype.label] = [prototype]
        # self._list = [item[0] for item in list(self._dict.values())]

    def cat(self, label=None):
        collection = self._list if label is None else self._dict[label]
        return torch.cat(list(map(lambda p: p.feature, collection)))

    def clear(self):
        self._list.clear()
        self._dict.clear()

    def update(self):
        # Orginal
        temp_list = self._list
        self._list = list()
        self._dict = dict()
        for p in temp_list:
            if p.weight > 1:
                p.weight = ceil(p.weight / 2)
                self._append(p)
        
        # For one prototype
        # temp_dict = self._dict
        # self._list = list()
        # self._dict = dict()
        # for label, pt_list in temp_dict.items():
        #     new_weight = 0
        #     all_features = []
        #     for pt in pt_list:
        #         new_weight += pt.weight
        #         all_features.append(pt.feature)
             
        #     new_feature = torch.mean(torch.cat(all_features), dim=0).reshape(1, -1)
        #     new_prototype = Prototypes.Prototype(new_feature, label, new_weight)
        #     self._dict[label] = [new_prototype]
        # self._list = [item[0] for item in list(self._dict.values())]



    def load(self, pkl_path):
        self.__dict__.update(torch.load(pkl_path))

    def save(self, pkl_path):
        torch.save(self.__dict__, pkl_path)

    def __getitem__(self, key):
        if isinstance(key, tuple):
            value = self._dict[key[0]][key[1]]
        else:
            value = self._list[key]

        return value

    def __setitem__(self, key, value):
        if isinstance(key, tuple):
            self._dict[key[0]][key[1]] = value
        elif isinstance(key, int):
            self._list[key] = value

    def __iter__(self):
        return iter(self._list)

    def __next__(self):
        return next(self._list)

    def __len__(self):
        return len(self._list)


class DCELoss(nn.Module):
    def __init__(self, gamma=0.1):
        super().__init__()

        self.gamma = gamma

    def forward(self, feature, prototype, prototypes):
        # distances = compute_multi_distance(feature, prototypes.cat(label))
        distance = compute_distance(feature, prototype.feature)
        prob = (-self.gamma * distance.pow(2)).exp().sum()
        # prob = (-self.gamma * distances).exp().sum()

        distances = compute_multi_distance(feature, prototypes.cat())
        one = (-self.gamma * distances.pow(2)).exp().sum()
        # one = (-self.gamma * distances).exp().sum()

        dce_loss = -(prob / one).log()

        # prob = probability(feature, label, prototypes, gamma=self.gamma)
        # dce_loss = -prob.log()

        return dce_loss


class PairwiseLoss(nn.Module):
    def __init__(self, tao=1.0, b=1.0, beta=0.1):
        super().__init__()

        self.b = b
        self.tao = tao
        self.beta = beta

    def forward(self, feature, label, prototype):
        distance = compute_distance(feature, prototype.feature)
        like = 1 if prototype.label == label else -1
        pw_loss = self._g(self.b - like * (self.tao - distance.pow(2)))

        return pw_loss

    def _g(self, z):
        return (1 + (self.beta * z).exp()).log() / self.beta if z < 10.0 else z


class CPELoss(nn.Module):
    def __init__(self, gamma=0.1, tao=10.0, b=1.0, beta=1.0, lambda_=0.1):
        super().__init__()

        self.lambda_ = lambda_
        # self.lambda_ = 0.0

        self.dce = DCELoss(gamma=gamma)
        self.pairwise = PairwiseLoss(tao=tao, b=b, beta=beta)
        self.ce = torch.nn.CrossEntropyLoss()

    def forward(self, feature, out, label, prototypes):
        prototype, distance = prototypes.assign(feature.clone().detach(), label.item())
        closest_prototype, min_distance = prototypes.closest(feature.clone().detach())

        dce_loss = self.dce(feature, prototype, prototypes)
        pairwise_loss = self.pairwise(feature, label.item(), prototype)
        pairwise_loss += self.pairwise(feature, closest_prototype.label, closest_prototype)
        ce_loss = self.ce(out, label)

        # if closest_prototype is not None:
        #     pl_loss = compute_distance(feature, closest_prototype.feature).pow(2)

        return dce_loss + ce_loss + self.lambda_ * pairwise_loss, distance

    def update(self, gamma=0.1, tao=10.0, b=1.0, beta=1.0):
        self.dce.gamma = gamma
        self.pairwise.tao = tao
        self.pairwise.b = b
        self.pairwise.beta = beta


class Detector(object):
    def __init__(self, distances: list, known_labels: set, std_coefficient=1.0):
        self.distances = np.array(distances, dtype=[('label', np.int32), ('distance', np.float32)])
        self._known_labels = set(known_labels)
        self.std_coefficient = std_coefficient

        self.average_distances = {l: np.average(self.distances[self.distances['label'] == l]['distance'])
                                  for l in self._known_labels}
        self.std_distances = {l: self.distances[self.distances['label'] == l]['distance'].std()
                              for l in self._known_labels}
        self.thresholds = {l: self.average_distances[l] + (self.std_coefficient * self.std_distances[l])
                           for l in self._known_labels}
        self.results = None

    def __call__(self, predicted_label, distance):
        novelty = False

        if predicted_label not in self._known_labels or distance > self.thresholds.get(predicted_label, 0.0):
            novelty = True

        return novelty

    @property
    def known_labels(self):
        return self._known_labels

    @known_labels.setter
    def known_labels(self, label_set):
        self._known_labels = set(label_set)

        self.average_distances = {l: np.average(self.distances[self.distances['label'] == l]['distance'])
                                  for l in self._known_labels}
        self.std_distances = {l: self.distances[self.distances['label'] == l]['distance'].std()
                              for l in self._known_labels}
        self.thresholds = {l: self.average_distances[l] + (self.std_coefficient * self.std_distances[l])
                           for l in self._known_labels}

    def evaluate(self, results):
        self.results = np.array(results, dtype=[
            ('true_label', np.int32),
            ('predicted_label', np.int32),
            # ('probability', np.float32),
            # ('distance', np.float32),
            ('real_novelty', np.bool),
            ('detected_novelty', np.bool)
        ])

        real_novelties = self.results[self.results['real_novelty']]
        detected_novelties = self.results[self.results['detected_novelty']]
        detected_real_novelties = self.results[self.results['detected_novelty'] & self.results['real_novelty']]

        true_positive = len(detected_real_novelties)
        false_positive = len(detected_novelties) - len(detected_real_novelties)
        false_negative = len(real_novelties) - len(detected_real_novelties)
        true_negative = len(self.results) - true_positive - false_positive - false_negative

        cm = confusion_matrix(
            self.results['true_label'],
            self.results['predicted_label'],
            labels=sorted(list(np.unique(self.results['true_label'])))
        )
        results = self.results[np.isin(self.results['true_label'], list(self._known_labels))]
        acc = accuracy_score(results['true_label'], results['predicted_label'])
        acc_all = accuracy_score(self.results['true_label'], self.results['predicted_label'])

        
        ## === Accuracy on new classes =====
        # unknown_results = results[np.isin(self.results['true_label'], list(self._known_labels), invert=True)]
        # NCA = accuracy_score(
        #     unknown_results['true_label'],
        #     unknown_results['predicted_label']
        # )
        NCA = 0.0

        return true_positive, false_positive, false_negative, true_negative, cm, acc, acc_all, NCA

    def load(self, pkl_path):
        self.__dict__.update(torch.load(pkl_path))

    def save(self, pkl_path):
        torch.save(self.__dict__, pkl_path)
