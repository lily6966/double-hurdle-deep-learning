import tensorflow as tf
import numpy as np
from tqdm import tqdm
from utlis import Dataset, THRESHOLDS  
from model import VAE, compute_loss  
from evals import compute_best_metrics, compute_metrics  
import json
import os

device = "/GPU:0" if tf.config.list_physical_devices('GPU') else "/CPU:0"
class Object(object):
    pass


if os.path.getsize('./TRAINED/args.json') == 0:
    print("The file is empty.")
else:
    with open('./TRAINED/args.json', 'r') as f:
        args_dict = json.load(f)

# Convert back to Namespace if needed (for argparse users)
from types import SimpleNamespace
args = SimpleNamespace(**args_dict)

##  load dataset  #####
feat_data = np.load(args.feat_data_dir)
feat_data = (feat_data - np.mean(feat_data, axis=0))/(np.std(feat_data, axis=0)+1e-8)
count_label_data = np.load(args.count_label_dir)
binary_label_data = np.load(args.binary_label_dir)

nonzero_idx = np.load(args.nonzero_idx_dir)
train_idx = np.load(args.train_idx_dir)
val_idx = np.load(args.val_idx_dir)
test_idx = np.load(args.test_idx_dir)

##  load dataset  #####
feat_data = np.load(args.feat_data_dir)
feat_data = (feat_data - np.mean(feat_data, axis=0))/(np.std(feat_data, axis=0)+1e-8)
count_label_data = np.load(args.count_label_dir)
binary_label_data = np.load(args.binary_label_dir)


# If using a specific device, you can use tf.device:
with tf.device(device):  
    feat_data = tf.convert_to_tensor(feat_data.astype(float), dtype=tf.float32)
    count_label_data = tf.convert_to_tensor(count_label_data.astype(float), dtype=tf.float32)
    binary_label_data = tf.convert_to_tensor(binary_label_data.astype(float), dtype=tf.float32)

##  prepare datasets and data loader ##
# Create TensorFlow datasets
testing_set = Dataset(test_idx, feat_data, count_label_data, binary_label_data)

# Define the batch size
batch_size = args.batch_size  # Same as in PyTorch

# Convert PyTorch dataset to TensorFlow dataset

# Batch and shuffle (if needed)
test_dataset = testing_set.batch(args.batch_size).shuffle(buffer_size=10000)  # Set shuffle=False for testing


# Apply batching and shuffling (if needed)
test_loader = test_dataset.prefetch(tf.data.experimental.AUTOTUNE)

## build models ##
vae = VAE(args=args, training=False)  # Initialize model
vae.build(input_shape=(None, args.input_dim))  # Build model with correct shape
vae.load_weights("./TRAINED/model_classifier.weights.h5")  # Load weights

# Poisson Negative Log Likelihood Loss
criterion = tf.keras.losses.Poisson(reduction=tf.keras.losses.Reduction.NONE)
# Alternatively, use L1 Loss (MAE)
# criterion = tf.keras.losses.MeanAbsoluteError()

# Define optimizer
# Adam optimizer in TensorFlow
optimizer = tf.keras.optimizers.Adam(
    learning_rate=args.lr, beta_1=0.9, beta_2=0.98, epsilon=1e-6
)
one_epoch_iters = np.ceil(len(train_idx) / args.batch_size)
lr_step_size = int(one_epoch_iters * args.max_epoch / args.lr_decay_times)  # compute the number of iterations in each epoch

# Learning rate scheduler
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=args.lr,
    decay_steps=lr_step_size,
    decay_rate=args.lr_decay,
    staircase=True
)

# Update optimizer to use scheduled learning rate
optimizer = tf.keras.optimizers.Adam(
    learning_rate=lr_schedule, beta_1=0.9, beta_2=0.98, epsilon=1e-6
)


# LOOPs ###
smooth_total_loss = 0.0
smooth_nll_loss = 0.0
smooth_nll_loss_x = 0.0
smooth_kl_loss = 0.0
smooth_cpc_loss = 0.0
step_count = 0

prediction_y = []
prediction_x = []
count_label_gt = []
binary_label_gt = []

# Validation phase
for data in tqdm(test_loader, mininterval=0.5, desc=f'(TESTING)', position=0, leave=True, ascii=True):
    input_feat = data[0]
    input_label_count = data[1]
    input_label_binary = data[2]

    # Convert to TensorFlow tensors
    input_feat = tf.convert_to_tensor(input_feat, dtype=tf.float32)
    input_label_count = tf.convert_to_tensor(input_label_count, dtype=tf.float32)
    input_label_binary = tf.convert_to_tensor(input_label_binary, dtype=tf.float32)

    # forward pass
    output = vae(input_feat, input_label_count, input_label_binary, training=False)
    total_loss, nll_loss, nll_loss_x, kl_loss, cpc_loss, pred_e, pred_x = compute_loss(
        input_label_binary, input_label_count, output, criterion, mode='classification'
    )

    # Update counters
    step_count += 1
    smooth_total_loss += total_loss.numpy()
    smooth_nll_loss += nll_loss.numpy()
    smooth_nll_loss_x += nll_loss_x.numpy()
    smooth_kl_loss += kl_loss.numpy()
    smooth_cpc_loss += cpc_loss.numpy()

    # Store predictions
    prediction_y.append(pred_e.numpy())
    prediction_x.append(pred_x.numpy())
    count_label_gt.append(input_label_count.numpy())
    binary_label_gt.append(input_label_binary.numpy())
   

# Compute average losses
smooth_total_loss /= step_count
smooth_nll_loss /= step_count
smooth_nll_loss_x /= step_count
smooth_kl_loss /= step_count
smooth_cpc_loss /= step_count

# Concatenate arrays
prediction_y = np.concatenate(prediction_y, axis=0)
prediction_x = np.concatenate(prediction_x, axis=0)
count_label_gt = np.concatenate(count_label_gt, axis=0)
binary_label_gt = np.concatenate(binary_label_gt, axis=0)



train_metric_x = compute_metrics(prediction_x, binary_label_gt, 0.5)
acc_x, ha_x, ebf1_x, maf1_x, mif1_x = train_metric_x['ACC'], train_metric_x['HA'], train_metric_x['ebF1'], \
                                        train_metric_x['maF1'], train_metric_x['miF1']

print("\n********** VALIDATION STATISTIC ***********")
print("total_loss =%.6f\t nll_loss =%.6f\t nll_loss_x =%.6f\t kl_loss =%.6f\t cpc_loss=%.6f" %
      (smooth_total_loss,smooth_nll_loss,smooth_nll_loss_x,smooth_kl_loss,smooth_cpc_loss))
print("acc_x=%.6f\t ha_x=%.6f\t ebf1_x=%.6f\t maf1_x=%.6f\t mif1_x=%.6f" % (acc_x, ha_x, ebf1_x, maf1_x, mif1_x))
#print("acc_y=%.6f\t ha_y=%.6f\t ebf1_y=%.6f\t maf1_y=%.6f\t mif1_y=%.6f" % (acc_y, ha_y, ebf1_y, maf1_y, mif1_y))
print("\n*****************************************")














