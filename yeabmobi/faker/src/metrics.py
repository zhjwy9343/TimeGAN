from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import accuracy_score
from utils import train_test_divide, extract_time, batch_generator


def visualization(ori_data, generated_data, analysis):
  """Using PCA or tSNE for generated and original data visualization.

  Args:
    - ori_data: original data
    - generated_data: generated synthetic data
    - analysis: tsne or pca
  """
  # Analysis sample size (for faster computation)
  anal_sample_no = min([1000, len(ori_data)])
  idx = np.random.permutation(len(ori_data))[:anal_sample_no]

  # Data preprocessing
  ori_data = np.asarray(ori_data)
  generated_data = np.asarray(generated_data)

  ori_data = ori_data[idx]
  generated_data = generated_data[idx]

  no, seq_len, dim = ori_data.shape

  for i in range(anal_sample_no):
    if (i == 0):
      prep_data = np.reshape(np.mean(ori_data[0, :, :], 1), [1, seq_len])
      prep_data_hat = np.reshape(np.mean(generated_data[0, :, :], 1), [1, seq_len])
    else:
      prep_data = np.concatenate((prep_data,
                                  np.reshape(np.mean(ori_data[i, :, :], 1), [1, seq_len])))
      prep_data_hat = np.concatenate((prep_data_hat,
                                      np.reshape(np.mean(generated_data[i, :, :], 1), [1, seq_len])))

  # Visualization parameter
  colors = ["red" for i in range(anal_sample_no)] + ["blue" for i in range(anal_sample_no)]

  if analysis == 'pca':
    # PCA Analysis
    pca = PCA(n_components=2)
    pca.fit(prep_data)
    pca_results = pca.transform(prep_data)
    pca_hat_results = pca.transform(prep_data_hat)

    # Plotting
    f, ax = plt.subplots(1)
    plt.scatter(pca_results[:, 0], pca_results[:, 1],
                c=colors[:anal_sample_no], alpha=0.2, label="Original")
    plt.scatter(pca_hat_results[:, 0], pca_hat_results[:, 1],
                c=colors[anal_sample_no:], alpha=0.2, label="Synthetic")

    ax.legend()
    plt.title('PCA plot')
    plt.xlabel('x-pca')
    plt.ylabel('y_pca')
    plt.show()

  elif analysis == 'tsne':

    # Do t-SNE Analysis together
    prep_data_final = np.concatenate((prep_data, prep_data_hat), axis=0)

    # TSNE anlaysis
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(prep_data_final)

    # Plotting
    f, ax = plt.subplots(1)

    plt.scatter(tsne_results[:anal_sample_no, 0], tsne_results[:anal_sample_no, 1],
                c=colors[:anal_sample_no], alpha=0.2, label="Original")
    plt.scatter(tsne_results[anal_sample_no:, 0], tsne_results[anal_sample_no:, 1],
                c=colors[anal_sample_no:], alpha=0.2, label="Synthetic")

    ax.legend()

    plt.title('t-SNE plot')
    plt.xlabel('x-tsne')
    plt.ylabel('y_tsne')
    plt.show()


def predictive_score_metrics(ori_data, generated_data):
    """Report the performance of Post-hoc RNN one-step ahead prediction.

    Args:
      - ori_data: original data
      - generated_data: generated synthetic data

    Returns:
      - predictive_score: MAE of the predictions on the original data
    """
    # Initialization on the Graph
    tf.reset_default_graph()

    # Basic Parameters
    no = len(ori_data)
    dim = 6

    # Set maximum sequence length and each sequence length
    ori_time, ori_max_seq_len = extract_time(ori_data)
    generated_time, generated_max_seq_len = extract_time(generated_data)
    max_seq_len = max([ori_max_seq_len, generated_max_seq_len])

    ## Builde a post-hoc RNN predictive network
    # Network parameters
    hidden_dim = int(dim / 2)
    iterations = 5000
    batch_size = 128

    # Input place holders
    X = tf.placeholder(tf.float32, [None, max_seq_len - 1, dim - 1], name="myinput_x")
    T = tf.placeholder(tf.int32, [None], name="myinput_t")
    Y = tf.placeholder(tf.float32, [None, max_seq_len - 1, 1], name="myinput_y")

    # Predictor function
    def predictor(x, t):
        """Simple predictor function.

        Args:
          - x: time-series data
          - t: time information

        Returns:
          - y_hat: prediction
          - p_vars: predictor variables
        """
        with tf.variable_scope("predictor", reuse=tf.AUTO_REUSE) as vs:
            p_cell = tf.nn.rnn_cell.GRUCell(num_units=hidden_dim, activation=tf.nn.tanh, name='p_cell')
            p_outputs, p_last_states = tf.nn.dynamic_rnn(p_cell, x, dtype=tf.float32, sequence_length=t)
            y_hat_logit = tf.contrib.layers.fully_connected(p_outputs, 1, activation_fn=None)
            y_hat = tf.nn.sigmoid(y_hat_logit)
            p_vars = [v for v in tf.all_variables() if v.name.startswith(vs.name)]

        return y_hat, p_vars

    y_pred, p_vars = predictor(X, T)
    # Loss for the predictor
    p_loss = tf.losses.absolute_difference(Y, y_pred)
    # optimizer
    p_solver = tf.train.AdamOptimizer().minimize(p_loss, var_list=p_vars)

    ## Training
    # Session start
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # Training using Synthetic dataset
    for itt in range(iterations):
        # Set mini-batch
        idx = np.random.permutation(len(generated_data))
        train_idx = idx[:batch_size]

        X_mb = list(generated_data[i][:-1, :(dim - 1)] for i in train_idx)
        T_mb = list(generated_time[i] - 1 for i in train_idx)
        Y_mb = list(
            np.reshape(generated_data[i][1:, (dim - 1)], [len(generated_data[i][1:, (dim - 1)]), 1]) for i in train_idx)

        # Train predictor
        _, step_p_loss = sess.run([p_solver, p_loss], feed_dict={X: X_mb, T: T_mb, Y: Y_mb})

        ## Test the trained model on the original data
    idx = np.random.permutation(len(ori_data))
    train_idx = idx[:no]

    X_mb = list(ori_data[i][:-1, :(dim - 1)] for i in train_idx)
    T_mb = list(ori_time[i] - 1 for i in train_idx)
    Y_mb = list(np.reshape(ori_data[i][1:, (dim - 1)], [len(ori_data[i][1:, (dim - 1)]), 1]) for i in train_idx)

    # Prediction
    pred_Y_curr = sess.run(y_pred, feed_dict={X: X_mb, T: T_mb})

    # Compute the performance in terms of MAE
    MAE_temp = 0
    for i in range(no):
        MAE_temp = MAE_temp + mean_absolute_error(Y_mb[i], pred_Y_curr[i, :, :])

    predictive_score = MAE_temp / no

    return predictive_score


def discriminative_score_metrics(ori_data, generated_data):
  """Use post-hoc RNN to classify original data and synthetic data

  Args:
    - ori_data: original data
    - generated_data: generated synthetic data

  Returns:
    - discriminative_score: np.abs(classification accuracy - 0.5)
  """
  # Initialization on the Graph
  tf.reset_default_graph()

  # Basic Parameters
  no = len(ori_data)
  dim = 6

  # Set maximum sequence length and each sequence length
  ori_time, ori_max_seq_len = extract_time(ori_data)
  generated_time, generated_max_seq_len = extract_time(generated_data)

  max_seq_len = max([ori_max_seq_len, generated_max_seq_len])

  ## Build a post-hoc RNN discriminator network
  # Network parameters
  hidden_dim = int(dim / 2)
  iterations = 2000
  batch_size = 128

  # Input place holders
  # Feature
  X = tf.placeholder(tf.float32, [None, max_seq_len, dim], name="myinput_x")
  X_hat = tf.placeholder(tf.float32, [None, max_seq_len, dim], name="myinput_x_hat")

  T = tf.placeholder(tf.int32, [None], name="myinput_t")
  T_hat = tf.placeholder(tf.int32, [None], name="myinput_t_hat")

  # discriminator function
  def discriminator(x, t):
    """Simple discriminator function.

    Args:
      - x: time-series data
      - t: time information

    Returns:
      - y_hat_logit: logits of the discriminator output
      - y_hat: discriminator output
      - d_vars: discriminator variables
    """
    with tf.variable_scope("discriminator", reuse=tf.AUTO_REUSE) as vs:
      d_cell = tf.nn.rnn_cell.GRUCell(num_units=hidden_dim, activation=tf.nn.tanh, name='d_cell')
      d_outputs, d_last_states = tf.nn.dynamic_rnn(d_cell, x, dtype=tf.float32, sequence_length=t)
      y_hat_logit = tf.contrib.layers.fully_connected(d_last_states, 1, activation_fn=None)
      y_hat = tf.nn.sigmoid(y_hat_logit)
      d_vars = [v for v in tf.all_variables() if v.name.startswith(vs.name)]

    return y_hat_logit, y_hat, d_vars

  y_logit_real, y_pred_real, d_vars = discriminator(X, T)
  y_logit_fake, y_pred_fake, _ = discriminator(X_hat, T_hat)

  # Loss for the discriminator
  d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y_logit_real,
                                                                       labels=tf.ones_like(y_logit_real)))
  d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y_logit_fake,
                                                                       labels=tf.zeros_like(y_logit_fake)))
  d_loss = d_loss_real + d_loss_fake

  # optimizer
  d_solver = tf.train.AdamOptimizer().minimize(d_loss, var_list=d_vars)

  ## Train the discriminator
  # Start session and initialize
  sess = tf.Session()
  sess.run(tf.global_variables_initializer())

  # Train/test division for both original and generated data
  train_x, train_x_hat, test_x, test_x_hat, train_t, train_t_hat, test_t, test_t_hat = \
    train_test_divide(ori_data, generated_data, ori_time, generated_time)

  # Training step
  for itt in range(iterations):
    # Batch setting
    X_mb, T_mb = batch_generator(train_x, train_t, batch_size)
    X_hat_mb, T_hat_mb = batch_generator(train_x_hat, train_t_hat, batch_size)

    # Train discriminator
    _, step_d_loss = sess.run([d_solver, d_loss],
                              feed_dict={X: X_mb, T: T_mb, X_hat: X_hat_mb, T_hat: T_hat_mb})

    ## Test the performance on the testing set
  y_pred_real_curr, y_pred_fake_curr = sess.run([y_pred_real, y_pred_fake],
                                                feed_dict={X: test_x, T: test_t, X_hat: test_x_hat, T_hat: test_t_hat})

  y_pred_final = np.squeeze(np.concatenate((y_pred_real_curr, y_pred_fake_curr), axis=0))
  y_label_final = np.concatenate((np.ones([len(y_pred_real_curr), ]), np.zeros([len(y_pred_fake_curr), ])), axis=0)

  # Compute the accuracy
  acc = accuracy_score(y_label_final, (y_pred_final > 0.5))
  discriminative_score = np.abs(0.5 - acc)

  return discriminative_score