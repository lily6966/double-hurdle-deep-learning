import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from utlis import Dataset, sample_generator
from model import MODEL
from WGAN import Generator
import random

device = tf.device("cuda:0" if tf.cuda.is_available() else "cpu")
class Object(object):
    pass

# Load checkpoint
vae = tf.keras.models.load_model('./TRAINED/model_regressor')
args = np.load('./TRAINED/args.npy', allow_pickle=True).item()


# Load dataset
feat_data = np.load(args['feat_data_dir'])
label_data = np.load(args['label_data_dir'])

feat_data = tf.convert_to_tensor(feat_data, dtype=tf.float32)
label_data = tf.convert_to_tensor(label_data, dtype=tf.float32)

label_mean = tf.reduce_mean(label_data, axis=0)
label_std = tf.math.reduce_std(label_data, axis=0)



##  prepare train, test, val indceis, datasets, data loader ##
random_num = 456
random.seed(random_num)
idx_list = list(range(feat_data.shape[0]))

train_idx, test_val = train_test_split(idx_list, train_size=0.8, random_state=random_num) # list
test_idx, val_idx = train_test_split(test_val, test_size=0.5, random_state=random_num) # list

train_idx = np.load('./data/train_idx.npy')
test_idx = np.load('./data/test_idx.npy')
val_idx = np.load('./data/val_idx.npy')


# Create TensorFlow Dataset
test_feat_data = tf.gather(feat_data, test_idx)
test_label_data = tf.gather(label_data, test_idx)

testing_dataset = tf.data.Dataset.from_tensor_slices((test_feat_data, test_label_data))
# Define data loader (batching and shuffling)
testing_loader = testing_dataset.batch(args['batch_size']).shuffle(False)  # shuffle=False as in PyTorch


## build models ##
vae = MODEL(args).to(device)
vae.build((None, args.feat_dim))  # Build the model (e.g., define input shapes)
vae.load_weights(checkpoint_path + 'model_regressor')  # Load the pre-trained weights

# Load Generator weights
G = Generator(label_dim=args.add_feat_dim, latent_dim=50, feature_dim=args.feat_dim)
G.load_weights('generator_MP2020.pt')  # Load the pre-trained generator model weights
G.eval()

# Loss function (L1 loss is equivalent to Mean Absolute Error)
L1_loss = tf.keras.losses.MeanAbsoluteError()

# Initialize loss accumulators
smooth_kl_loss = 0.0
smooth_loss = 0.0
smooth_l1_x = 0.0
smooth_l1_y = 0.0
inter_loss_weight = 0.2

step_count = 0

prediction_y = []
prediction_x = []
label_s = []

best_valid_loss = 1e+10
checkpoint_path = './TRAINED/'

# Validation phase
for data in tqdm(testing_loader, mininterval=0.5, desc=f'(TESTING)', position=0, leave=True, ascii=True):
    input_feat, input_label = data  # TensorFlow dataset returns (features, labels)

    input_label_std = (input_label - label_mean) / label_std
    add_feat = sample_generator(G, tf.shape(input_feat)[0], input_feat)

    # Forward pass
    logit_x, intermediate_logits_x, logit_y, intermediate_logits_y, mu_x, mu_y, r_sqrt_sigma = vae([input_feat, input_label_std, add_feat])

    # Compute L1 losses
    loss_x = L1_loss(input_label_std, logit_x)
    loss_y = L1_loss(input_label_std, logit_y)

    smooth_l1_x += loss_x.numpy()
    smooth_l1_y += loss_y.numpy()

    # Compute KL divergence
    sigma = tf.matmul(r_sqrt_sigma, tf.transpose(r_sqrt_sigma))  # Equivalent of torch.mm
    sigma = tf.expand_dims(sigma, axis=0)
    sigma = tf.repeat(sigma, tf.shape(input_feat)[0], axis=0)

    m_x = tfp.distributions.MultivariateNormalFullCovariance(loc=mu_x, covariance_matrix=sigma)
    m_y = tfp.distributions.MultivariateNormalFullCovariance(loc=mu_y, covariance_matrix=sigma)
    
    kl_loss = tf.reduce_mean(tfp.distributions.kl_divergence(m_y, m_x))
    smooth_kl_loss += kl_loss.numpy()

    # Intermediate losses
    if intermediate_logits_x is not None:
        for logit_tmp in intermediate_logits_x:
            loss_tmp = L1_loss(input_label_std, logit_tmp)
            loss_x += inter_loss_weight * loss_tmp

    if intermediate_logits_y is not None:
        for logit_tmp in intermediate_logits_y:
            loss_tmp = L1_loss(input_label_std, logit_tmp)
            loss_y += inter_loss_weight * loss_tmp

    # Compute final loss
    loss = loss_x + loss_y + 0.1 * kl_loss

    # Store predictions and labels
    prediction_y.append(logit_y.numpy())
    prediction_x.append(logit_x.numpy())
    label_s.append(input_label_std.numpy())

    smooth_loss += loss.numpy()
    step_count += 1

# Compute final statistics
smooth_loss /= step_count
smooth_l1_x /= step_count
smooth_l1_y /= step_count
smooth_kl_loss /= step_count

# Convert predictions to numpy arrays
prediction_y = np.concatenate(prediction_y, axis=0)
prediction_x = np.concatenate(prediction_x, axis=0)
label_s = np.concatenate(label_s, axis=0)

# Compute Mean Absolute Error
mae = np.mean(np.abs(prediction_y - label_s))
mae_x = np.mean(np.abs(prediction_x - label_s))

mae_ori = np.mean(np.abs(prediction_y - label_s) * label_std.numpy())
mae_x_ori = np.mean(np.abs(prediction_x - label_s) * label_std.numpy())

# Print statistics
print("\n********** TESTING STATISTIC ***********")
print(f"total_loss = {smooth_loss:.6f}\t nll_loss = {smooth_l1_y:.6f}\t nll_loss_x = {smooth_l1_x:.6f}\t kl_loss = {smooth_kl_loss:.6f}")
print(f"mae = {mae:.6f}\t mae_x = {mae_x:.6f}\t mae_ori = {mae_ori:.6f}\t mae_x_ori = {mae_x_ori:.6f}")
print("\n*****************************************")


# Load the MPIDs DataFrame
small_nozero_mpids = pd.read_csv('./data/small_nozero_mpids.csv')

# Select testing MPIDs (same as in PyTorch)
testing_mpid = small_nozero_mpids.iloc[test_idx]

# Save the testing MPIDs as CSV
testing_mpid.to_csv(f'./RESULT/testing_mpids.csv', index=False, header=True)

# Convert TensorFlow tensors to NumPy arrays and save them
np.save('./RESULT/label_mean_vector.npy', label_mean.numpy())  # Tensor to NumPy
np.save('./RESULT/label_std_vector.npy', label_std.numpy())  # Tensor to NumPy

# Save predictions and ground truth values
np.save('./RESULT/prediction.npy', np.array(prediction_x))  # Convert list to NumPy array if needed
np.save('./RESULT/label_ground_true.npy', np.array(label_s))  # Convert list to NumPy array if needed







