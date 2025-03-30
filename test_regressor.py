import tensorflow as tf
import numpy as np
import os
from tqdm import tqdm
import random
from tqdm import tqdm
from utlis import Dataset, THRESHOLDS
from model import VAE, compute_loss
from evals import compute_best_metrics, compute_metrics
import json

test_wo_gt = True
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

##  define parameters ####
args.input_dim = 47
args.label_dim = 829
args.max_epoch = 100
args.batch_size = 128
args.feat_data_dir = './DATA/CostaRica_Data_DHN/habitat_feat.npy'
args.count_label_dir = './DATA/CostaRica_Data_DHN/count_label.npy'
args.binary_label_dir = './DATA/CostaRica_Data_DHN/binary_label.npy'
args.nonzero_idx_dir = './DATA/CostaRica_Data_DHN/non_zero_idx.npy'
args.train_idx_dir = './DATA/CostaRica_Data_DHN/train_idx.npy'
args.val_idx_dir = './DATA/CostaRica_Data_DHN/val_idx.npy'
args.test_idx_dir = './DATA/CostaRica_Data_DHN/test_idx.npy'
args.test_wo_gt = './DATA/CostaRica_Data_DHN/feat_wo_label.npy'

args.latent_dim = 64
args.emb_size = 2048
args.scale_coeff = 1
args.reg = "gmvae"
args.test_sample = False
args.keep_prob = 0.5
args.lr_decay = 0.5
args.lr = 2e-4
args.lr_decay_times = 4
args.mode = 'regression'

##  load dataset  #####
feat_data = np.load(args.feat_data_dir)
feat_data = (feat_data - np.mean(feat_data, axis=0))/(np.std(feat_data, axis=0)+1e-8)
count_label_data = np.load(args.count_label_dir)
binary_label_data = np.load(args.binary_label_dir)
feat_wo_gt_data = np.load(args.test_wo_gt)
feat_wo_gt_data = (feat_wo_gt_data- np.mean(feat_wo_gt_data, axis=0)) / (np.std(feat_wo_gt_data, axis=0)+1e-8)
N_feat_wo_gt = feat_wo_gt_data.shape[0]

nonzero_idx = np.load(args.nonzero_idx_dir)
train_idx = np.load(args.train_idx_dir)
val_idx = np.load(args.val_idx_dir)
test_idx = np.load(args.test_idx_dir)


# If using a specific device, you can use tf.device:
with tf.device(device):  
    feat_data = tf.convert_to_tensor(feat_data.astype(float), dtype=tf.float32)
    feat_wo_gt_data = tf.convert_to_tensor(feat_wo_gt_data.astype(float), dtype=tf.float32)
    count_label_data = tf.convert_to_tensor(count_label_data.astype(float), dtype=tf.float32)
    binary_label_data = tf.convert_to_tensor(binary_label_data.astype(float), dtype=tf.float32)




##  prepare datasets and data loader ##

##  prepare datasets and data loader ##
# Create TensorFlow datasets
testing_set = Dataset(test_idx, feat_data, count_label_data, binary_label_data)
test_wo_gt_set = Dataset(list(range(N_feat_wo_gt)), feat_wo_gt_data, count_label_data, binary_label_data) # count_label_data and binary_label_data are not useful here

# Define the batch size
batch_size = args.batch_size  # Same as in PyTorch

# Batch and shuffle (if needed)
test_dataset = testing_set.batch(args.batch_size).shuffle(buffer_size=10000)  # Set shuffle=False for testing


# Apply batching and shuffling (if needed)
test_loader = test_dataset.prefetch(tf.data.experimental.AUTOTUNE)


# Apply batching and shuffling (if needed)
test_wo_gt_loader = test_wo_gt_set.batch(batch_size).shuffle(buffer_size=10000).prefetch(tf.data.experimental.AUTOTUNE)



## build models ##
regressor = VAE(args=args, training=True)  # Initialize model
regressor.build(input_shape=(None, args.input_dim))  # Build model with correct shape
regressor.load_weights("./TRAINED/model_regressor.weights.h5")  # Load weights

classifier = VAE(args=args, training=True)  # Initialize model
classifier.build(input_shape=(None, args.input_dim))  # Build model with correct shape
classifier.load_weights("./TRAINED/model_classifier.weights.h5")  # Load weights


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
prediction_b = []

## testing phase

if test_wo_gt:
    loader = test_wo_gt_loader
else:
    loader = test_loader

# Validation phase
for data in tqdm(loader, mininterval=0.5, desc=f'(TESTING)', position=0, leave=True, ascii=True):
    input_feat = data[0]
    input_label_count = data[1]
    input_label_binary = data[2]

    # Convert to TensorFlow tensors
    input_feat = tf.convert_to_tensor(input_feat, dtype=tf.float32)
    input_label_count = tf.convert_to_tensor(input_label_count, dtype=tf.float32)
    input_label_binary = tf.convert_to_tensor(input_label_binary, dtype=tf.float32)

    # Classifier forward pass
    output = classifier(input_feat, input_label_count, input_label_binary, training=False)
    pred_x_class_b = tf.nn.sigmoid(output['feat_out'])

    # Regressor forward pass
    output = regressor(input_feat, input_label_count, input_label_binary, training=False)
    total_loss, nll_loss, nll_loss_x, kl_loss, cpc_loss, pred_e, pred_x = compute_loss(
        input_label_binary, input_label_count, output, criterion, mode='regression', pred_binary=pred_x_class_b
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
    prediction_b.append(pred_x_class_b.numpy())

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
prediction_b = np.concatenate(prediction_b, axis=0)

if test_wo_gt: # saving prediction results for feature wo ground-truth
    # for layer in regressor.layers:
    #     weights = layer.get_weights()
    #     if weights:
    #         print(f"Layer: {layer.name}, Weights Shape: {[w.shape for w in weights]}")

    emb2 = np.transpose(classifier.get_layer('sequential_3').get_weights()[0])    # [0] gives you the weight matrix
    emb = np.transpose(regressor.get_layer('sequential').get_weights()[0])  # Similarly for regressor

    prediction_x = np.array(prediction_x)  # Convert list to NumPy array
    prediction_x = (prediction_x + 0.5).astype(int)  # Apply transformation
    # Ensure the RESULTS directory exists
    os.makedirs("./RESULTS", exist_ok=True)


    np.save('./RESULTS/costaRica_srd_2019_species_count_prediction.npy', prediction_x)
    np.save('./RESULTS/ccostaRica_srd_2019_species_present_probability.npy', prediction_b)
    np.save('./RESULTS/cspecies_embedding_from_species_count_pred_model.npy', emb)
    np.save('./RESULTS/cspecies_embedding_from_species_present_pred_model.npy', emb2)

    if prediction_x.size == 0:
        print("Error: prediction_x is empty!")
    else:
        print(np.max(prediction_b), np.min(prediction_x))

    
    print(np.max(prediction_x), np.min(prediction_x))
    print(np.max(prediction_b), np.min(prediction_b))

    nonzero_ratio = np.count_nonzero(prediction_x) / (prediction_x.shape[0]*prediction_x.shape[1])

    print(prediction_x.shape[0]*prediction_x.shape[1])
    print(nonzero_ratio)
    print(np.isnan(emb).any())
    print(np.isnan(emb2).any())
    print(np.count_nonzero(np.sum(emb,axis=1)))
    print(np.count_nonzero(np.sum(emb2,axis=1)))

else:
    prediction_b[prediction_b<0.01] = 0
    prediction_b[prediction_b>=0.01] = 1

    rescaled = (prediction_x*prediction_b+0.5).astype(int)
    ae = np.abs(rescaled-count_label_gt)
    mae_x = np.mean(ae)
    mae_x_positive_part = np.sum(ae*binary_label_gt)/np.sum(binary_label_gt)
    max_x_zero_part = np.sum(ae*(1-binary_label_gt))/np.sum(1-binary_label_gt)

    print("\n********** TESTING STATISTIC ***********")
    print("total_loss =%.6f\t nll_loss =%.6f\t nll_loss_x =%.6f\t kl_loss =%.6f\t cpc_loss=%.6f" %
          (smooth_total_loss,smooth_nll_loss,smooth_nll_loss_x,smooth_kl_loss,smooth_cpc_loss))
    print("mae_x=%.6f\t mae_x_positive_part=%.6f\t mae_x_zero_part=%.6f\t" % (mae_x, mae_x_positive_part, max_x_zero_part))
    print("\n*****************************************")











