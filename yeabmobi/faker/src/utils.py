## Necessary Packages
import numpy as np
import tensorflow as tf


def min_max_scaler(data):
  """Min Max normalizer.

  Args:
    - data: original data

  Returns:
    - norm_data: normalized data
  """
  numerator = data - np.min(data, 0)
  denominator = np.max(data, 0) - np.min(data, 0)
  norm_data = numerator / (denominator + 1e-7)
  return norm_data, np.max(data, 0), np.min(data, 0)

def pad_seq(data, seq_len):
  data_result = []
  for d in data:
    data_result.append(np.pad(d, ((0, seq_len - len(d)), (0, 0)), 'constant', constant_values=(0)))
  return data_result

def train_test_divide (data_x, data_x_hat, data_t, data_t_hat, train_rate = 0.8):
  """Divide train and test data for both original and synthetic data.

  Args:
    - data_x: original data
    - data_x_hat: generated data
    - data_t: original time
    - data_t_hat: generated time
    - train_rate: ratio of training data from the original data
  """
  # Divide train/test index (original data)
  no = len(data_x)
  idx = np.random.permutation(no)
  train_idx = idx[:int(no*train_rate)]
  test_idx = idx[int(no*train_rate):]

  train_x = [data_x[i] for i in train_idx]
  test_x = [data_x[i] for i in test_idx]
  train_t = [data_t[i] for i in train_idx]
  test_t = [data_t[i] for i in test_idx]

  # Divide train/test index (synthetic data)
  no = len(data_x_hat)
  idx = np.random.permutation(no)
  train_idx = idx[:int(no*train_rate)]
  test_idx = idx[int(no*train_rate):]

  train_x_hat = [data_x_hat[i] for i in train_idx]
  test_x_hat = [data_x_hat[i] for i in test_idx]
  train_t_hat = [data_t_hat[i] for i in train_idx]
  test_t_hat = [data_t_hat[i] for i in test_idx]

  return train_x, train_x_hat, test_x, test_x_hat, train_t, train_t_hat, test_t, test_t_hat


def extract_time (data):
  """Returns Maximum sequence length and each sequence length.

  Args:
    - data: original data

  Returns:
    - time: extracted time information
    - max_seq_len: maximum sequence length
  """
  time = list()
  max_seq_len = 0
  for i in range(len(data)):
    max_seq_len = max(max_seq_len, len(data[i][:,0]))
    time.append(len(data[i][:,0]))

  return time, max_seq_len


def rnn_cell(module, hidden_dim):
  """Basic RNN Cell.

  Args:
    - module: gru, lstm, or lstmLN

  Returns:
    - rnn_cell: RNN Cell
  """
  assert module in ['gru','lstm','lstmLN']

  # GRU
  if (module == 'gru'):
    rnn_cell = tf.nn.rnn_cell.GRUCell(num_units=hidden_dim, activation=tf.nn.tanh)
  # LSTM
  elif (module == 'lstm'):
    rnn_cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_dim, activation=tf.nn.tanh)
  # LSTM Layer Normalization
  elif (module == 'lstmLN'):
    rnn_cell = tf.contrib.rnn.LayerNormBasicLSTMCell(num_units=hidden_dim, activation=tf.nn.tanh)
  return rnn_cell


def random_generator(batch_size, z_dim, T_mb, max_seq_len):
  """Random vector generation.

  Args:
    - batch_size: size of the random vector
    - z_dim: dimension of random vector
    - T_mb: time information for the random vector
    - max_seq_len: maximum sequence length

  Returns:
    - Z_mb: generated random vector
  """
  Z_mb = list()
  for i in range(batch_size):
    temp = np.zeros([max_seq_len, z_dim])
    temp_Z = np.random.uniform(0., 1, [T_mb[i], z_dim])
    temp[:T_mb[i], :] = temp_Z

    pad_Z = np.pad(temp_Z, ((0, max_seq_len - len(temp_Z)), (0, 0)), 'constant', constant_values=(0))
    Z_mb.append(pad_Z)
    del temp
  return Z_mb


def batch_generator(data, time, batch_size):
  """Mini-batch generator.

  Args:
    - data: time-series data
    - time: time information
    - batch_size: the number of samples in each batch

  Returns:
    - X_mb: time-series data in each batch
    - T_mb: time information in each batch
  """
  no = len(data)
  idx = np.random.permutation(no)
  train_idx = idx[:batch_size]

  X_mb = list(data[i] for i in train_idx)
  T_mb = list(time[i] for i in train_idx)
  del idx
  return X_mb, T_mb
