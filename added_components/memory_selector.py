import numpy as np



class IncrementalMemory():
  def __init__(self,
              selection_type='fixed_mem',   # ['fixed_mem', 'pre_class']
              total_size=1000,
              per_class=100,
              selection_method='rand'):
    
    self.selection_type = selection_type
    self.total_size = total_size
    self.per_class = per_class
    self.selection_method = selection_method

    self.class_data = {}
  
  def __call__(self):
    # for label, features in self.class_data.items():
    #   print('{} -> {}'.format(label, features.shape))
    return np.concatenate(list(self.class_data.values()), axis=0)

  def update(self, data):
    
    new_samples = np.array(data)
    labels = np.array(data[:, -1]).flatten()
    unique_labels = list(np.unique(labels))
    # print('unique_labels: {}'.format(unique_labels))

    new_class_data = {
      l: new_samples[np.where(labels == l)[0]]
      for l in unique_labels
    }

    if self.selection_type == 'fixed_mem':

      if not self.class_data:
        class_size = int(self.total_size / len(unique_labels))
      else:
        known_labels = list(self.class_data.keys())
        all_labels = unique_labels + known_labels
        class_size = int(self.total_size / len(all_labels))
        
        for label, samples in self.class_data.items():
          n = samples.shape[0]
          if n > class_size:
            idxs = np.random.choice(range(n), size=class_size, replace=False)
            self.class_data[label] = samples[idxs]
          else:
            self.class_data[label] = samples

      for label, samples in new_class_data.items():
        n = samples.shape[0]
        if n > class_size:
          idxs = np.random.choice(range(n), size=class_size, replace=False)
          self.class_data[label] = samples[idxs]
        else:
          self.class_data[label] = samples
    
    elif self.selection_type == 'pre_class':
      for label in unique_labels:
        n = new_class_data[label].shape[0]
        idxs = np.random.choice(range(n), size=self.per_class, replace=False)
        self.class_data[label] = new_samples[idxs]

