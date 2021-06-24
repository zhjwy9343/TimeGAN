import tensorflow as tf
import numpy as np
import pandas as pd
import os

np.set_printoptions(linewidth=np.inf, precision=2)

def rnn_cell(module, hidden_dim):
  assert module in ['gru', 'lstm', 'lstmLN']

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


def read_train_data(train_file, max_seq_len):
  train_df = pd.read_csv(train_file, usecols=[0, 1, 2, 3, 4, 5, 6]).to_numpy()

  ori_data = []
  ori_time = []
  temp_ori = []

  id_col = train_df[:, :1]
  last_id = id_col[0][0]

  value_col = train_df[:, 1:]
  value_col, max_val, min_val = min_max_scaler(value_col)

  temp_ori.append(value_col[0])
  for i in range(1, len(id_col)):
    if id_col[i][0] != last_id:
      ori_time.append(min(len(temp_ori), max_seq_len))
      pad_len = max_seq_len - min(len(temp_ori), max_seq_len)
      ori_data.append(np.pad(temp_ori, ((0, pad_len), (0, 0)), 'constant', constant_values=(0)))
      temp_ori = []
    if (len(temp_ori)<max_seq_len):
      temp_ori.append(value_col[i])
    last_id = id_col[i]

  pad_len = max_seq_len - min(len(temp_ori), max_seq_len)
  ori_data.append(np.pad(temp_ori, ((0, pad_len), (0, 0)), 'constant', constant_values=(0)))
  ori_time.append(len(temp_ori))

  return ori_data, ori_time, max_val, min_val

def min_max_scaler(data):
  numerator = data - np.min(data, 0)
  denominator = np.max(data, 0) - np.min(data, 0)
  scaled_data = numerator / (denominator + 1e-7)
  return scaled_data, np.max(data, 0), np.min(data, 0)

def random_generator(batch_size, z_dim, T_mb, max_seq_len):
  Z_mb = list()
  for i in range(batch_size):
    temp = np.zeros([max_seq_len, z_dim])
    temp_Z = np.random.uniform(0., 1, [T_mb[i], z_dim])
    temp[:T_mb[i], :] = temp_Z

    pad_Z = np.pad(temp_Z, ((0, max_seq_len - len(temp_Z)), (0, 0)), 'constant', constant_values=(0))
    Z_mb.append(pad_Z)
  return Z_mb

def calc_metric(bhv_data_list):

  bhv_num_time_dict = {}
  time_segment_cnt_dict = {}
  for bhv_data in bhv_data_list:
    bhv_num = len(bhv_data)
    sess_time = 0
    for bhv in bhv_data:
      [wait_time, x_beg, y_beg, x_end, y_end, is_click] = bhv
      sess_time = sess_time + wait_time

    time_segment = int(sess_time / 1000 / 60)
    if time_segment in time_segment_cnt_dict.keys():
      time_segment_cnt_dict[time_segment] = time_segment_cnt_dict[time_segment] + 1
    else:
      time_segment_cnt_dict[time_segment] = 1

    if bhv_num in bhv_num_time_dict.keys():
      bhv_num_time_dict[bhv_num].append(sess_time)
    else:
      bhv_num_time_dict[bhv_num] = [sess_time]

  for key in sorted(bhv_num_time_dict):
    value = bhv_num_time_dict.get(key)
    print("bhv_num: {}, cnt: {}, time_mean: {}, time_std: {}".format(key, len(value), np.mean(value), np.std(value, ddof=1)))

  time_segment_cnt_dict = sorted(time_segment_cnt_dict.items(), key=lambda x: x[0])
  print(time_segment_cnt_dict)


def predict(train_file, ckpts_dir, max_seq_len, z_dim, dim, module, hidden_dim, num_layers):
  tf.reset_default_graph()
  def embedder(X, T):
    with tf.variable_scope("embedder", reuse=tf.AUTO_REUSE):
      e_cell = tf.nn.rnn_cell.MultiRNNCell([rnn_cell(module, hidden_dim) for _ in range(num_layers)])
      e_outputs, e_last_states = tf.nn.dynamic_rnn(e_cell, X, dtype=tf.float32, sequence_length=T)
      H = tf.contrib.layers.fully_connected(e_outputs, hidden_dim, activation_fn=tf.nn.sigmoid)
    return H

  def recovery(H, T):
    with tf.variable_scope("recovery", reuse=tf.AUTO_REUSE):
      r_cell = tf.nn.rnn_cell.MultiRNNCell([rnn_cell(module, hidden_dim) for _ in range(num_layers)])
      r_outputs, r_last_states = tf.nn.dynamic_rnn(r_cell, H, dtype=tf.float32, sequence_length=T)
      X_tilde = tf.contrib.layers.fully_connected(r_outputs, dim, activation_fn=tf.nn.sigmoid)
    return X_tilde

  def generator(Z, T):
    with tf.variable_scope("generator", reuse=tf.AUTO_REUSE):
      e_cell = tf.nn.rnn_cell.MultiRNNCell([rnn_cell(module, hidden_dim) for _ in range(num_layers)])
      e_outputs, e_last_states = tf.nn.dynamic_rnn(e_cell, Z, dtype=tf.float32, sequence_length=T)
      E = tf.contrib.layers.fully_connected(e_outputs, hidden_dim, activation_fn=tf.nn.sigmoid)
    return E

  def supervisor(H, T):
    with tf.variable_scope("supervisor", reuse=tf.AUTO_REUSE):
      e_cell = tf.nn.rnn_cell.MultiRNNCell([rnn_cell(module, hidden_dim) for _ in range(num_layers - 1)])
      e_outputs, e_last_states = tf.nn.dynamic_rnn(e_cell, H, dtype=tf.float32, sequence_length=T)
      S = tf.contrib.layers.fully_connected(e_outputs, hidden_dim, activation_fn=tf.nn.sigmoid)
    return S

  ori_data, ori_time, max_val, min_val = read_train_data(train_file, max_seq_len)
  Z_mb = random_generator(len(ori_data), z_dim, ori_time, max_seq_len)

  # Input place holders
  X = tf.placeholder(tf.float32, [None, max_seq_len, dim], name="myinput_x")
  Z = tf.placeholder(tf.float32, [None, max_seq_len, z_dim], name="myinput_z")
  T = tf.placeholder(tf.int32, [None], name="myinput_t")

  # Embedder
  H = embedder(X, T)

  # Generator
  E_hat = generator(Z, T)
  H_hat = supervisor(E_hat, T)

  # Synthetic data
  X_hat = recovery(H_hat, T)

  ori_data = np.asarray(ori_data)
  print("ori_data.shape:", np.asarray(ori_data).shape)
  print("ori_time.shape:", np.asarray(ori_time).shape)
  print("Z_mb.shape:", np.asarray(Z_mb).shape)
  with tf.Session() as sess:
    model_file = tf.train.latest_checkpoint(ckpts_dir)
    saver = tf.train.Saver()
    saver.restore(sess, model_file)
    gen_data = sess.run(X_hat, feed_dict={X: ori_data, Z: Z_mb, T: ori_time})

    resize_ori_data = list()
    resize_gen_data = list()
    for i in range(len(ori_time)):
      ori_data_i = ori_data[i, :ori_time[i], :]
      rescale_ori_data_i = ori_data_i * (max_val - min_val) + min_val
      resize_ori_data_i = rescale_ori_data_i*np.array([1, 720, 1280, 720, 1280, 1])
      resize_ori_data.append(resize_ori_data_i)

      gen_data_i = gen_data[i, :ori_time[i], :]
      rescale_gen_data_i = gen_data_i * (max_val - min_val) + min_val
      resize_gen_data_i = rescale_gen_data_i * np.array([1, 720, 1280, 720, 1280, 1])
      resize_gen_data.append(resize_gen_data_i)
      if i< 100:
        print("ori_data ", i , ":")
        print(resize_ori_data_i)
        print("gen_data ", i, ":")
        print(resize_gen_data_i)

  return resize_ori_data, resize_gen_data



if __name__=="__main__":
  # 超参
  train_file = '/Users/zhanghao/dianyi/faker/data/input/data/train/train.txt'
  ckpts_dir = '/Users/zhanghao/dianyi/faker/data/checkpoints'
  output_dir = '/Users/zhanghao/dianyi/faker/data/output'
  max_seq_len = 64
  z_dim = dim = 6
  module = "lstm"
  hidden_dim = 32
  num_layers = 3

  # Generate
  ori_data, gen_data = predict(train_file, ckpts_dir, max_seq_len, z_dim,
                               dim, module, hidden_dim, num_layers)

  for i in range(len(gen_data)):
    for j in range(len(gen_data[i])):
      if gen_data[i][j][5] > 0.5:
        gen_data[i][j][5] = 1
      else:
        gen_data[i][j][5] = 0

  # Save
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)
  with open(os.path.join(output_dir, 'ori_data_1280_720.txt'), 'w') as outfile:
    outfile.write('# Array shape: {0}\n'.format(np.asarray(ori_data).shape))
    for data_slice in ori_data:
      outfile.write('# New slice, bhv_num:{}\n'.format(len(data_slice)))
      np.savetxt(outfile, data_slice, fmt='%-7.2f')

  with open(os.path.join(output_dir, 'gen_data_1280_720.txt'), 'w') as outfile:
    outfile.write('# Array shape: {0}\n'.format(np.asarray(gen_data).shape))
    for data_slice in gen_data:
      outfile.write('# New slice, bhv_num:{}\n'.format(len(data_slice)))
      np.savetxt(outfile, data_slice, fmt='%-7.2f')

  # Evaluate
  print("\ncal ori data metric\n")
  calc_metric(ori_data)
  print("\ncal gen data metric\n")
  calc_metric(gen_data)
