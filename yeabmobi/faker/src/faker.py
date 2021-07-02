## Necessary packages
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import argparse
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from timegan import timegan

from metrics import discriminative_score_metrics, predictive_score_metrics, visualization
from utils import pad_seq, min_max_scaler

import tensorflow as tf
# number_CPU = int(os.environ.get('SM_NUM_CPUS'))
# tf.config.threading.set_intra_op_parallelism_threads(number_CPU)
# tf.config.threading.set_inter_op_parallelism_threads(number_CPU)
#
# os.environ["KMP_AFFINITY"] = "verbose,disabled"
# # os.environ["KMP_AFFINITY"]= "granularity=fine,verbose,compact,1,0"
# # os.environ["KMP_AFFINITY"]= "granularity=fine,verbose,scatter,1,0"
# os.environ['OMP_NUM_THREADS'] = str(number_CPU)
# os.environ['KMP_SETTINGS'] = '1'


def read_bhv_data(data_dir, data_name, seq_len):
  import pandas as pd
  file_path = os.path.join(data_dir, data_name)
  train_df = pd.read_csv(file_path, usecols=[0, 1, 2, 3, 4, 5, 6]).to_numpy()

  id_col = train_df[:, :1]
  value_col = train_df[:, 1:]

  norm_value_col, max_val, min_val = min_max_scaler(value_col)

  ori_data = []
  norm_data = []
  ori_time = []
  temp_ori = []
  temp_norm = []
  last_id = id_col[0][0]
  temp_ori.append(value_col[0])
  temp_norm.append(norm_value_col[0])
  for i in range(1, len(id_col)):
      if id_col[i][0] != last_id:
          ori_time.append(min(len(temp_ori), seq_len))
          ori_data.append(temp_ori)
          pad_data = np.pad(temp_norm, ((0, seq_len - min(len(temp_norm), seq_len)), (0, 0)), 'constant', constant_values=(0))
          norm_data.append(pad_data)
          temp_ori = []
          temp_norm = []
      if (len(temp_ori)<seq_len):
        temp_ori.append(value_col[i])
        temp_norm.append(norm_value_col[i])
      last_id = id_col[i]
  ori_time.append(len(temp_ori))
  ori_data.append(temp_ori)
  pad_data = np.pad(temp_norm, ((0, seq_len - min(len(temp_norm), seq_len)), (0, 0)), 'constant', constant_values=(0))
  norm_data.append(pad_data)

  return ori_data, norm_data, ori_time, max_val, min_val


def main (args):
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(args.gpu)

    # data_dir = args.work_dir
    seq_data, norm_data, seq_time, max_val, min_val = read_bhv_data(args.data_path, args.seq_len)
    print(np.asarray(norm_data).shape)
    print(args.data_path + ' dataset is ready.')

    ## Synthetic data generation by TimeGAN
    # Set newtork parameters
    parameters = dict()
    parameters['work_dir'] = args.work_dir + str(np.random.randint(0, 10000))
    parameters['module'] = args.module
    parameters['hidden_dim'] = args.hidden_dim
    parameters['num_layer'] = args.num_layer
    parameters['iterations'] = args.iteration
    parameters['batch_size'] = args.batch_size
    parameters['learning_rate'] = args.learning_rate

    generated_data = timegan(norm_data, seq_time, max_val, min_val, args.seq_len, parameters)
    print('Finish Synthetic Data Generation')

    ## Performance metrics
    # Output initialization
    metric_results = dict()
    seq_data = pad_seq(seq_data, args.seq_len)
    generated_data = pad_seq(generated_data, args.seq_len)

    # 1. Discriminative Score
    discriminative_score = list()
    for _ in range(args.metric_iteration):
        temp_disc = discriminative_score_metrics(seq_data, generated_data)
        discriminative_score.append(temp_disc)

    metric_results['discriminative'] = np.mean(discriminative_score)

    # 2. Predictive score
    predictive_score = list()
    for tt in range(args.metric_iteration):
        temp_pred = predictive_score_metrics(seq_data, generated_data)
        predictive_score.append(temp_pred)

    metric_results['predictive'] = np.mean(predictive_score)

    # 3. Visualization (PCA and tSNE)
    # visualization(seq_data, generated_data, 'pca')
    # visualization(seq_data, generated_data, 'tsne')

    ## Print discriminative and predictive scores
    print(metric_results)

    return seq_data, generated_data, metric_results


if __name__ == '__main__':
    # Inputs for the main function
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument(
        '--data_path',
        default='train.txt',
        type=str)
    parser.add_argument(
        '--seq_len',
        help='sequence length',
        default=64,
        type=int)
    parser.add_argument(
        '--module',
        choices=['gru', 'lstm', 'lstmLN'],
        default='gru',
        type=str)
    parser.add_argument(
        '--hidden_dim',
        help='hidden state dimensions (should be optimized)',
        default=24,
        type=int)
    parser.add_argument(
        '--num_layer',
        help='number of layers (should be optimized)',
        default=3,
        type=int)
    parser.add_argument(
        '--iteration',
        help='Training iterations (should be optimized)',
        default=10,
        type=int)
    parser.add_argument(
        '--batch_size',
        help='the number of samples in mini-batch (should be optimized)',
        default=128,
        type=int)
    parser.add_argument(
        '--learning_rate',
        help='learning rate',
        default=0.001,
        type=float)
    parser.add_argument(
        '--metric_iteration',
        help='iterations of the metric computation',
        default=3,
        type=int)
    parser.add_argument(
        '--work_dir',
        type=str,
        default='../')
    parser.add_argument(
        '--model_dir',
        type=str,
        default='./data')

    args = parser.parse_args()
    print(args)

    # Calls main function
    ori_data, gen_data, metrics = main(args)

    output_dir = os.path.join(args.work_dir, 'output/data')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(os.path.join(output_dir, 'ori_data.txt'), 'w') as outfile:
        outfile.write('# Array shape: {0}\n'.format(np.asarray(ori_data).shape))
        for data_slice in ori_data:
            np.savetxt(outfile, data_slice, fmt='%-7.2f')
            outfile.write('# New slice\n')

    with open(os.path.join(output_dir, 'generated_data.txt'), 'w') as outfile:
        outfile.write('# Array shape: {0}\n'.format(np.asarray(gen_data).shape))
        for data_slice in gen_data:
            np.savetxt(outfile, data_slice, fmt='%-7.2f')
            outfile.write('# New slice\n')

    with open(os.path.join(output_dir, 'ori_data_1280_720.txt'), 'w') as outfile:
        outfile.write('# Array shape: {0}\n'.format(np.asarray(ori_data).shape))
        for data_slice in ori_data:
            data_resize = data_slice * np.array([1, 720, 1280, 720, 1280, 1])
            np.savetxt(outfile, data_resize, fmt='%-7.2f')
            outfile.write('# New slice\n')

    with open(os.path.join(output_dir, 'gen_data_1280_720.txt'), 'w') as outfile:
        outfile.write('# Array shape: {0}\n'.format(np.asarray(gen_data).shape))
        for data_slice in gen_data:
            data_resize = data_slice * np.array([1, 720, 1280, 720, 1280, 1])
            np.savetxt(outfile, data_resize, fmt='%-7.2f')
            outfile.write('# New slice\n')