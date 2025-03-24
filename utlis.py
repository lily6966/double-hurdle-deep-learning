import tensorflow as tf
import numpy as np
import os
import sys
from operator import attrgetter
from tensorflow.keras.models import Model

THRESHOLDS = [i / 10. for i in range(1, 10)]

def swap_0_1(tensor, on_zero, on_non_zero):
    """
    Swap values in a tensor: replace 0s with `on_zero` and non-zeros with `on_non_zero`
    """
    res = tf.identity(tensor)
    res = tf.where(tf.equal(tensor, 0), tf.constant(on_zero, dtype=tensor.dtype), tf.constant(on_non_zero, dtype=tensor.dtype))
    return res

def save_model(model, epoch_i, current_step, opt, checkpoint_path, save_keys_list=[]):
    """
    Save model's weights and optimizer settings to a checkpoint file
    """
    model_state_dict = {var.name: var.numpy() for var in model.trainable_variables}

    # Filter keys if specified
    if save_keys_list:
        model_state_dict = {key: model_state_dict[key] for key in save_keys_list if key in model_state_dict}

    checkpoint = {
        'model': model_state_dict,
        'settings': opt,
        'epoch': epoch_i,
        'step': current_step
    }

    np.save(checkpoint_path, checkpoint)

def load_model(model, checkpoint_path, keys_to_restore_list=[]):
    """
    Load model weights and optimizer settings from a checkpoint file
    """
    if not os.path.exists(checkpoint_path):
        sys.exit('Checkpoint file not found!')

    checkpoint = np.load(checkpoint_path, allow_pickle=True).item()
    pretrained_dict = checkpoint['model']
    opt = checkpoint['settings']
    step = checkpoint['step']
    epoch = checkpoint['epoch']

    # Filter keys if specified
    if keys_to_restore_list:
        pretrained_dict = {key: pretrained_dict[key] for key in keys_to_restore_list if key in pretrained_dict}

    if pretrained_dict:
        for var in model.trainable_variables:
            if var.name in pretrained_dict:
                var.assign(pretrained_dict[var.name])
    else:
        sys.exit('There is nothing to be loaded!')

    return opt, step, epoch


def freeze_params(model, params_to_freeze_list):
    for str in params_to_freeze_list:
        attr = attrgetter(str)(model)
        attr.requires_grad = False
        attr.grad = None


def freeze_params(model, params_to_freeze_list):
    for param in params_to_freeze_list:
        attr = getattr(model, param, None)
        if attr is not None:
            attr.trainable = False  # In TensorFlow, this prevents updates

def unfreeze_params(model, params_to_unfreeze_list):
    for param in params_to_unfreeze_list:
        attr = getattr(model, param, None)
        if attr is not None:
            attr.trainable = True

class CustomDataset(tf.data.Dataset):
    def __new__(cls, list_IDs, features, count_labels, binary_labels):
        dataset = tf.data.Dataset.from_tensor_slices((
            features[list_IDs], count_labels[list_IDs], binary_labels[list_IDs]
        ))
        return dataset

def sample_generator(G, num_samples, feature):
    generated_data_all = 0
    for _ in range(50):
        latent_samples = tf.random.normal(shape=(num_samples, G.latent_dim))
        generated_data = G(tf.concat([feature, latent_samples], axis=1), training=False)
        generated_data_all += generated_data
    generated_data = generated_data_all / 50.0
    return generated_data





