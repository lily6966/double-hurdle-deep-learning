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

device = tf.device("cuda:0" if tf.cuda.is_available() else "cpu")

class Object(object):
    pass

with open('./TRAINED/args.json', 'r') as f:
    args = json.load(f)

# Load the model using tf.keras.models.load_model
loaded_model = tf.keras.models.load_model('./TRAINED/model_classifier')


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


nonzero_idx = np.load(args.nonzero_idx_dir)
train_idx = np.load(args.train_idx_dir)
val_idx = np.load(args.val_idx_dir)
test_idx = np.load(args.test_idx_dir)

# If using a specific device, you can use tf.device:
with tf.device("cuda:0" if tf.cuda.is_available() else "cpu"):  
    feat_data = tf.convert_to_tensor(feat_data.astype(float), dtype=tf.float32)
    count_label_data = tf.convert_to_tensor(count_label_data.astype(float), dtype=tf.float32)
    binary_label_data = tf.convert_to_tensor(binary_label_data.astype(float), dtype=tf.float32)

##  prepare datasets and data loader ##
# Create TensorFlow datasets
train_dataset = tf.data.Dataset.from_tensor_slices((feat_data[train_idx], count_label_data[train_idx], binary_label_data[train_idx]))
val_dataset = tf.data.Dataset.from_tensor_slices((feat_data[val_idx], count_label_data[val_idx], binary_label_data[val_idx]))

# Batch the datasets
train_dataset = train_dataset.batch(128).shuffle(1000)
val_dataset = val_dataset.batch(128)


## build models ##
regressor = VAE(args).to(device)
classifier = VAE(args).to(device)

# Load weights from the saved model
checkpoint_path = "./TRAINED/model_classifier"
regressor.load_weights(checkpoint_path)
classifier.load_weights(checkpoint_path)

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
import tensorflow as tf
import numpy as np
from tqdm import tqdm

# Reset metrics
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

best_value = 1e+10
checkpoint_path = './TRAINED/'

# Training Loop
for epoch in range(args.max_epoch):
    print(f"\n(EPOCH:{epoch} TRAINING)")

    # Training stage
    for data in tqdm(train_loader, mininterval=0.5, desc=f'Epoch {epoch} Training', position=0, leave=True, ascii=True):
        input_feat = data[0]
        input_label_count = data[1]
        input_label_binary = data[2]

        # Convert to TensorFlow tensors
        input_feat = tf.convert_to_tensor(input_feat, dtype=tf.float32)
        input_label_count = tf.convert_to_tensor(input_label_count, dtype=tf.float32)
        input_label_binary = tf.convert_to_tensor(input_label_binary, dtype=tf.float32)

        # Classifier forward pass (no gradients needed)
        output = classifier(input_feat, training=False)
        pred_x_class_b = tf.nn.sigmoid(output['feat_out'])

        # Regressor forward pass with gradients
        with tf.GradientTape() as tape:
            output = regressor(input_feat, training=True)
            total_loss, nll_loss, nll_loss_x, kl_loss, cpc_loss, pred_e, pred_x = compute_loss(
                input_label_binary, input_label_count, output, criterion, mode='regression', pred_binary=pred_x_class_b
            )

        # Compute and apply gradients
        gradients = tape.gradient(total_loss, regressor.trainable_variables)
        optimizer.apply_gradients(zip(gradients, regressor.trainable_variables))

        # Update learning rate
        lr_schedule(step_count)

        # Track step count
        step_count += 1

        # Store predictions
        prediction_b.append(pred_x_class_b.numpy())

        smooth_total_loss += total_loss.numpy()
        smooth_nll_loss += nll_loss.numpy()
        smooth_nll_loss_x += nll_loss_x.numpy()
        smooth_kl_loss += kl_loss.numpy()
        smooth_cpc_loss += cpc_loss.numpy()

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

    # Concatenate results
    prediction_y = np.concatenate(prediction_y, axis=0)
    prediction_x = np.concatenate(prediction_x, axis=0)
    count_label_gt = np.concatenate(count_label_gt, axis=0)
    binary_label_gt = np.concatenate(binary_label_gt, axis=0)

    prediction_b = np.concatenate(prediction_b, axis=0)
    prediction_b[prediction_b < 0.01] = 0
    prediction_b[prediction_b >= 0.01] = 1

    rescaled = (prediction_x * prediction_b + 0.5).astype(int)
    ae = np.abs(rescaled - count_label_gt)
    mae_x = np.mean(ae)
    mae_x_positive_part = np.sum(ae * binary_label_gt) / np.sum(binary_label_gt)
    max_x_zero_part = np.sum(ae * (1 - binary_label_gt)) / np.sum(1 - binary_label_gt)
    
    print("\n********** TRAINING STATISTIC ***********")
    print("total_loss =%.6f\t nll_loss =%.6f\t nll_loss_x =%.6f\t kl_loss =%.6f\t cpc_loss=%.6f" %
          (smooth_total_loss,smooth_nll_loss,smooth_nll_loss_x,smooth_kl_loss,smooth_cpc_loss))
    print("mae_x=%.6f\t mae_x_positive_part=%.6f\t mae_x_zero_part=%.6f\t" % (mae_x, mae_x_positive_part, max_x_zero_part))
    print("\n*****************************************")

        # Reset metrics
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

    # Validation phase
    for data in tqdm(val_loader, mininterval=0.5, desc=f'(EPOCH:{epoch} VALIDATING)', position=0, leave=True, ascii=True):
        input_feat = data[0]
        input_label_count = data[1]
        input_label_binary = data[2]

        # Convert to TensorFlow tensors
        input_feat = tf.convert_to_tensor(input_feat, dtype=tf.float32)
        input_label_count = tf.convert_to_tensor(input_label_count, dtype=tf.float32)
        input_label_binary = tf.convert_to_tensor(input_label_binary, dtype=tf.float32)

        # Classifier forward pass
        output = classifier(input_feat, training=False)
        pred_x_class_b = tf.nn.sigmoid(output['feat_out'])

        # Regressor forward pass
        output = regressor(input_feat, training=False)
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

    # Apply thresholding
    prediction_b[prediction_b < 0.01] = 0
    prediction_b[prediction_b >= 0.01] = 1

    # Rescale and compute metrics
    rescaled = (prediction_x * prediction_b + 0.5).astype(int)
    ae = np.abs(rescaled - count_label_gt)
    mae_x = np.mean(ae)
    mae_x_positive_part = np.sum(ae * binary_label_gt) / np.sum(binary_label_gt)
    max_x_zero_part = np.sum(ae * (1 - binary_label_gt)) / np.sum(1 - binary_label_gt)

        # Save best model
    if best_value > mae_x_positive_part:
        best_value = mae_x_positive_part
        print("\n********** SAVING MODEL ***********")

        # Save model as a directory (TensorFlow-style)
        save_path = checkpoint_path + 'model_regressor'
        regressor.save(save_path)
        print("A new model has been saved to " + checkpoint_path + 'model_classifier')

        # Save args (the model parameters) to a separate file (e.g., as a JSON file)
        with open(checkpoint_path + 'args.json', 'w') as f:
            json.dump(args, f, indent=4)
        print("\n*****************************************")

    # Reset metrics for next epoch
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










