# Path
import os
root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
fig_dir = os.path.join(root, 'figs')

import pandas as pd
import matplotlib.pyplot as plt
import torch as pt
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

def get_data(file_path) -> pd.DataFrame:
  data = pd.read_csv(file_path)
  return data

def min_max_normalize(data: pd.DataFrame) -> pd.DataFrame:
  data_filtered = data.iloc[:, -8:]
  mins = data_filtered.min()
  maxs = data_filtered.max()
  norm_data_filtered = (data_filtered - mins) / (maxs - mins)
  norm_data_filtered = 2 * norm_data_filtered - 1  # Scale to [-1, 1]
  norm_data = data.copy()
  norm_data.iloc[:, -8:] = norm_data_filtered
  return norm_data, mins, maxs

def plot_heatmap(data, norm_data, force: int = 15, sensor_id: int = 1):
  # Prepare Data
  data_filtered = data[data['F'] == force][['X', 'Y', f'Sensor R{sensor_id}']].reset_index(drop=True)
  heatmap_data = data_filtered.pivot(index='Y', columns='X', values=f'Sensor R{sensor_id}') # Matrix for heatmap
  norm_data_filtered = norm_data[norm_data['F'] == force][['X', 'Y', f'Sensor R{sensor_id}']].reset_index(drop=True)
  norm_heatmap_data = norm_data_filtered.pivot(index='Y', columns='X', values=f'Sensor R{sensor_id}')

  # Plot Heatmap for Original Data
  fig = plt.figure(figsize=(14, 8))
  ax1 = fig.add_subplot(1, 2, 1)
  im = ax1.imshow(heatmap_data, origin='lower', cmap='viridis', extent=[data_filtered['X'].min(), data_filtered['X'].max(), data_filtered['Y'].min(), data_filtered['Y'].max()])
  plt.colorbar(im, ax=ax1, label=f'Sensor R{sensor_id}')
  ax1.set_title(f'Heatmap for Sensor R{sensor_id} at Force {force}N')
  ax1.set_xlabel('X')
  ax1.set_ylabel('Y')

  # Plot Heatmap for Normalized Data
  ax2 = fig.add_subplot(1, 2, 2)
  im2 = ax2.imshow(norm_heatmap_data, origin='lower', cmap='viridis', extent=[norm_data_filtered['X'].min(), norm_data_filtered['X'].max(), norm_data_filtered['Y'].min(), norm_data_filtered['Y'].max()])
  plt.colorbar(im2, ax=ax2, label=f'Sensor R{sensor_id}')
  ax2.set_title(f'Normalized Heatmap for Sensor R{sensor_id} at Force {force}N')
  ax2.set_xlabel('X')
  ax2.set_ylabel('Y')

  plt.show()

def prepare_data(data: pd.DataFrame, train_size=0.7, val_size=0.15, test_size=0.15, targets='xyf', random_seed=42) -> pt.Tensor:
  # Inputs
  X = data[[f"Sensor R{i}" for i in range(1, 9)]].values # Shape (num_samples, 8)
  # Labels
  if targets == 'xy':
    y = data[['X', 'Y']].values # Shape (num_samples, 2)
  elif targets == 'f':
    y = data[['F']].values # Shape (num_samples, 1)
  else:  # 'xyf'
    y = data[['X', 'Y', 'F']].values # Shape (num_samples, 3)

  # Split Data
  X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=test_size, random_state=random_seed, shuffle=True) # split off test set
  relative_val_size = val_size / (train_size + val_size) # adjust validation size
  X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=relative_val_size, random_state=random_seed, shuffle=True) # split train and validation set
  return X_train, X_val, X_test, y_train, y_val, y_test

def create_dataloaders(X_train, X_val, X_test, y_train, y_val, y_test, batch_size=32):

  # Convert to Pytorch Tensors
  X_train_tensor = pt.tensor(X_train, dtype=pt.float32)
  y_train_tensor = pt.tensor(y_train, dtype=pt.float32)
  X_val_tensor = pt.tensor(X_val, dtype=pt.float32)
  y_val_tensor = pt.tensor(y_val, dtype=pt.float32)
  X_test_tensor = pt.tensor(X_test, dtype=pt.float32)
  y_test_tensor = pt.tensor(y_test, dtype=pt.float32)

  # Create Datasets
  train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
  val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
  test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

  # Create DataLoaders
  train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
  val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
  test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

  return train_loader, val_loader, test_loader

def set_seed(seed=42):
    import torch, random, numpy as np, os
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)