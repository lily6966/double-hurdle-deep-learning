import tensorflow as tf
import numpy as np
from tqdm import tqdm
from utils_2 import Dataset, THRESHOLDS  
from model import VAE, compute_loss  
from evals import compute_best_metrics, compute_metrics  
import json



class Object(object):
    pass

args = Object()

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
args.lr = 5e-3
args.lr_decay_times = 4
args.mode = 'classification'

##  load dataset  #####
feat_data = np.load(args.feat_data_dir)
feat_data = (feat_data - np.mean(feat_data, axis=0))/(np.std(feat_data, axis=0)+1e-8)
count_label_data = np.load(args.count_label_dir)
binary_label_data = np.load(args.binary_label_dir)

#print(np.isnan(feat_data).any())
#print(np.isnan(count_label_data).any())
#print(np.isnan(binary_label_data).any())
#exit()

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
training_set = Dataset(train_idx, feat_data, count_label_data, binary_label_data)
validation_set = Dataset(val_idx, feat_data, count_label_data, binary_label_data)

train_dataset = training_set.shuffle(buffer_size=10000).batch(args.batch_size).prefetch(tf.data.AUTOTUNE)
val_dataset = validation_set.batch(args.batch_size).prefetch(tf.data.AUTOTUNE)

## build models ##
classifier = VAE(args).to("cuda:0" if tf.cuda.is_available() else "cpu")

## optimizer, metric, scheduler etc ##
optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr, beta_1=0.9, beta_2=0.98, epsilon=1e-6)
criterion = tf.keras.losses.BinaryCrossentropy(from_logits=True)

# Learning rate scheduler
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=args.lr,
    decay_steps=int(np.ceil(len(train_idx) / args.batch_size)),  # equivalent to one_epoch_iters in PyTorch
    decay_rate=args.lr_decay,
    staircase=True
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

best_ebf1_x = 0.0
checkpoint_path = './TRAINED/'

for epoch in range(args.max_epoch):
    # TRAINING-STAGE
    classifier.train()
    for data in tqdm(train_dataset, mininterval=0.5, desc=f'(EPOCH:{epoch} TRAINING)', position=0, leave=True, ascii=True):

        input_feat, input_label_count, input_label_binary = data

        with tf.GradientTape() as tape:
            output = classifier(input_feat, input_label_count, input_label_binary, training=True)
            total_loss, nll_loss, nll_loss_x, kl_loss, cpc_loss, pred_e, pred_x = compute_loss(input_label_binary, input_label_count, output, criterion, mode='classification')

        gradients = tape.gradient(total_loss, classifier.trainable_variables)
        optimizer.apply_gradients(zip(gradients, classifier.trainable_variables))

        smooth_total_loss += total_loss.numpy()
        smooth_nll_loss += nll_loss.numpy()
        smooth_nll_loss_x += nll_loss_x.numpy()
        smooth_kl_loss += kl_loss.numpy()
        smooth_cpc_loss += cpc_loss.numpy()

        prediction_y.append(pred_e.numpy())
        prediction_x.append(pred_x.numpy())
        count_label_gt.append(input_label_count.numpy())
        binary_label_gt.append(input_label_binary.numpy())

        step_count += 1


        smooth_total_loss /= step_count
        smooth_nll_loss /= step_count
        smooth_nll_loss_x /= step_count
        smooth_kl_loss /= step_count
        smooth_cpc_loss /= step_count

        prediction_y = np.concatenate(prediction_y, axis=0)
        prediction_x = np.concatenate(prediction_x, axis=0)
        count_label_gt = np.concatenate(count_label_gt, axis=0)
        binary_label_gt = np.concatenate(binary_label_gt, axis=0)




    print("\n********** TRAINING STATISTIC ***********")
    print("total_loss =%.6f\t nll_loss =%.6f\t nll_loss_x =%.6f\t kl_loss =%.6f\t cpc_loss=%.6f" %
          (smooth_total_loss,smooth_nll_loss,smooth_nll_loss_x,smooth_kl_loss,smooth_cpc_loss))
    print("\n*****************************************")

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

    ## validation phase
    classifier.eval()
    for data in tqdm(val_dataset, mininterval=0.5, desc=f'(EPOCH:{epoch} VALIDATING)', position=0, leave=True, ascii=True):
        input_feat, input_label_count, input_label_binary = data

        output = classifier(input_feat, input_label_count, input_label_binary, training=False)
        total_loss, nll_loss, nll_loss_x, kl_loss, cpc_loss, pred_e, pred_x = compute_loss(input_label_binary, input_label_count, output, criterion, mode='classification')

        smooth_total_loss += total_loss.numpy()
        smooth_nll_loss += nll_loss.numpy()
        smooth_nll_loss_x += nll_loss_x.numpy()
        smooth_kl_loss += kl_loss.numpy()
        smooth_cpc_loss += cpc_loss.numpy()

        prediction_y.append(pred_e.numpy())
        prediction_x.append(pred_x.numpy())
        count_label_gt.append(input_label_count.numpy())
        binary_label_gt.append(input_label_binary.numpy())

        step_count += 1

        smooth_total_loss /= step_count
    smooth_nll_loss /= step_count
    smooth_nll_loss_x /= step_count
    smooth_kl_loss /= step_count
    smooth_cpc_loss /= step_count

    prediction_y = np.concatenate(prediction_y, axis=0)
    prediction_x = np.concatenate(prediction_x, axis=0)
    count_label_gt = np.concatenate(count_label_gt, axis=0)
    binary_label_gt = np.concatenate(binary_label_gt, axis=0)

    train_metric_x = compute_metrics(prediction_x, binary_label_gt, 0.5)
    acc_x, ha_x, ebf1_x, maf1_x, mif1_x = train_metric_x['ACC'], train_metric_x['HA'], train_metric_x['ebF1'], \
                                           train_metric_x['maF1'], train_metric_x['miF1']

    print("\n********** VALIDATION STATISTIC ***********")
    print("total_loss =%.6f\t nll_loss =%.6f\t nll_loss_x =%.6f\t kl_loss =%.6f\t cpc_loss=%.6f" %
          (smooth_total_loss, smooth_nll_loss, smooth_nll_loss_x, smooth_kl_loss, smooth_cpc_loss))
    print("acc_x=%.6f\t ha_x=%.6f\t ebf1_x=%.6f\t maf1_x=%.6f\t mif1_x=%.6f" % (acc_x, ha_x, ebf1_x, maf1_x, mif1_x))
    print("\n*****************************************")

    if best_ebf1_x < ebf1_x:
        best_ebf1_x = ebf1_x
        print("\n********** SAVING MODEL ***********")
        # Save the model using the Keras API (in TensorFlow format)
        classifier.save(checkpoint_path + 'model_classifier')
        print("A new model has been saved to " + checkpoint_path + 'model_classifier')

        # Save args (the model parameters) to a separate file (e.g., as a JSON file)
        with open(checkpoint_path + 'args.json', 'w') as f:
            json.dump(args, f, indent=4)
        print("\n*****************************************")










