import models
import math
import numpy as np
import torch
from sklearn.mixture import GaussianMixture

def pseudo_labeler(data, n_component=2, ratio=1.0):
	"""
	Compute ...
	Args:
		data: list of (sample, label, feature)
	Returns:
		---
	""" 
	p_data = []
	samples = torch.stack([item[0] for item in data])
	labels = torch.tensor([item[1] for item in data])
	features = torch.squeeze(torch.stack([item[2] for item in data]))

	gmm = GaussianMixture(n_components=n_component, random_state=0)
	gmm.fit(features.detach().cpu().numpy())
	gmm_predict = gmm.predict(features.detach().cpu().numpy())

	for i in range(n_component):
		component_idx = np.where(gmm_predict == i)[0].astype(int)
		n = component_idx.shape[0]
		component_idx_ratio = \
			component_idx[np.random.choice(
				range(n), size=int(n*ratio), replace=False
			)]

		plabel = torch.argmax(torch.bincount(labels[component_idx_ratio])).item()
		component_samples = samples[component_idx]
		component_features = features[component_idx]
		p_data.extend([
			(component_samples[i], plabel, component_features[i])
			for i in range(n)
		])

	return p_data



# todo new sample method by YW
class Sampler(object):
    def __init__(self, data: list, num: int, prototypes: models.Prototypes, net: models.DenseNet, is_novel: bool, soft: bool, use_log: bool):
        self.prototypes = prototypes
        self.data = data
        self.num = num
        self.net = net
        self.selected = []
        self.is_novel = is_novel
        self.soft = soft
        self.use_log = use_log

    def return_data(self):
        return self.selected

    def sampling(self):
        self.selected = []
        scores = []
        metric = self.noval_metric if self.is_novel else self.original_metric

        for i, d in enumerate(self.data):
            scores.append(metric(d))

        for i in range(self.num):
            m = max(scores)
            j = scores.index(m)
            self.selected.append(self.data[j])
            scores.pop(j)
            self.data.pop(j)

    def original_metric(self, d):
        feature, label = d
        feature, label = feature.to(self.net.device).unsqueeze(0), label.item()
        _, feature = self.net(feature)
        prototype, distance = self.prototypes.closest(feature, label)
        distance = max(0.001, distance)
        score = prototype.weight / distance
        if self.use_log:
            score = max(score * 1000, 1.0001)
            score = math.log2(score)
        return score

    def noval_metric(self, d):
        feature, _ = d
        feature = feature.to(self.net.device).unsqueeze(0)
        _, feature = self.net(feature)
        prototype, score = self.prototypes.closest(feature)
        if self.use_log:
            score = max(score * 1000, 1.0001)
            score = math.log2(score)
        return score

    def soft_sampling(self):
        self.selected = []
        scores = []
        metric = self.noval_metric if self.is_novel else self.original_metric

        for i, d in enumerate(self.data):
            scores.append(metric(d))

        temp_list = range(len(self.data))
        temp_list = np.array(temp_list)
        scores = np.array(scores)
        scores = scores / scores.sum()
        temp_list = np.random.choice(temp_list, size=self.num, p=scores, replace=False)

        for x in temp_list:
            self.selected.append(self.data[x])

        return
