import tensorflow as tf
import keras
from tensorflow import (
    nn,
    shape,
    eye,
    concat,
    matmul,
    reduce_sum,
    exp,
    random,
    maximum,
)
from keras import Model, layers, Sequential, backend, Variable
import tensorflow as tf
from tensorflow import nn  # Only if you need functions like nn.relu()
from tensorflow.keras import Model, Sequential, layers, backend
from keras.saving import register_keras_serializable

@register_keras_serializable(package="Custom")
class VAE(Model):
    
    def __init__(self, args, training, **kwargs):
        super(VAE, self).__init__(**kwargs)

        # Dropout layer
        self.training = training
        self.args = args
        #feature layers
        input_dim = args.input_dim #+ args.meta_offset
        self.fx1 = layers.Dense(256, input_dim=input_dim, activation=None)
        self.fx2 = layers.Dense(512, input_dim=256, activation=None)
        self.fx3 = layers.Dense(256, input_dim = 512, activation=None)
        self.fx_mu = layers.Dense(args.latent_dim, input_dim = 256,  activation=None)
        self.fx_logvar = layers.Dense(args.latent_dim, input_dim= 256, activation=None)

        self.emb_size = args.emb_size

        self.fd_x1 = layers.Dense(512, input_dim = input_dim + args.latent_dim, activation=None)
        self.fd_x2 = Sequential([
            layers.Dense(self.emb_size, input_dim = 512, activation=None)
        ])
        
        self.feat_mp_mu = layers.Dense(args.label_dim, input_dim = self.emb_size, activation=None)
        
        # Reconstruction layers
        self.recon = Sequential([
            layers.Dense(512, activation=None),
            layers.ReLU(),
            layers.Dense(512, activation=None),
            layers.ReLU(),
            layers.Dense(input_dim, activation=None)
        ])
        
        # Label reconstruction layers
        self.label_recon = Sequential([
            layers.Dense(512, activation=None, input_shape=(args.latent_dim,)),
            layers.ReLU(),
            layers.Dense(self.emb_size, activation=None),
            layers.LeakyReLU()
        ])
        
         
        # Label layers
        self.fe0 = self.fe0 = layers.Dense(self.emb_size, input_dim=args.label_dim, activation=None)
        self.fe1 = layers.Dense(512, activation=None)
        self.fe2 = layers.Dense(256, activation=None)
        self.fe_mu = layers.Dense(args.latent_dim, activation=None)
        self.fe_logvar = layers.Dense(args.latent_dim, activation=None)

        # Feature encoding
        self.fd1 = self.fd_x1
        self.fd2 = self.fd_x2
        self.label_mp_mu = self.feat_mp_mu

        # Bias parameter (in TensorFlow, you typically define a trainable variable)
        self.bias = tf.Variable(tf.zeros([args.label_dim]), trainable=True)

        self.dropout = layers.Dropout(rate=1 - args.keep_prob)
        self.scale_coeff = args.scale_coeff

    # def build(self, input_shape):
    #         """ Define input-dependent layer configurations dynamically """
    #         self.fx1.build(input_shape)
    #         self.fx2.build((None, 256))
    #         self.fx3.build((None, 512))
    #         self.fx_mu.build((None, 256))
    #         self.fx_logvar.build((None, 256))
            
    #         self.fd_x1.build((None, self.args.input_dim + self.args.latent_dim))
    #         self.fd_x2.build((None, 512))
    #         self.feat_mp_mu.build((None, self.emb_size))

    #         self.fe0.build((None, self.args.label_dim))
    #         self.fe1.build((None, self.emb_size))
    #         self.fe2.build((None, 512))
    #         self.fe_mu.build((None, 256))
    #         self.fe_logvar.build((None, 256))

    #         # Call `build()` on sequential models too
    #         self.recon.build((None, self.args.input_dim))
    #         self.label_recon.build((None, 512))

    #         self.built = True  # Mark model as built

        # Label encoder
    def label_encode(self, x):
        h0 = self.dropout(tf.nn.relu(self.fe0(x)))  # [label_dim, emb_size]
        h1 = self.dropout(tf.nn.relu(self.fe1(h0)))  # [label_dim, 512]
        h2 = self.dropout(tf.nn.relu(self.fe2(h1)))  # [label_dim, 256]
        mu = self.fe_mu(h2) * self.scale_coeff  # [label_dim, latent_dim]
        logvar = self.fe_logvar(h2) * self.scale_coeff  # [label_dim, latent_dim]

        fe_output = {
            'fe_mu': mu,
            'fe_logvar': logvar
        }
        return fe_output

        
    def feat_encode(self, x):
        h1 = self.dropout(tf.nn.relu(self.fx1(x)))
        h2 = self.dropout(tf.nn.relu(self.fx2(h1)))
        h3 = self.dropout(tf.nn.relu(self.fx3(h2)))
        mu = self.fx_mu(h3) * self.scale_coeff  # [bs, latent_dim]
        logvar = self.fx_logvar(h3) * self.scale_coeff
        fx_output = {
            'fx_mu': mu,
            'fx_logvar': logvar
        }
        return fx_output
 
    def label_reparameterize(self, mu, logvar):
        std = tf.exp(0.5 * logvar)
        eps = tf.random.normal(shape=tf.shape(std))
        return mu + eps * std

    def feat_reparameterize(self, mu, logvar, coeff=1.0):
        std = tf.exp(0.5 * logvar)
        eps = tf.random.normal(shape=tf.shape(std))
        return mu + eps * std

    def label_decode(self, z):
        d1 = tf.nn.relu(self.fd1(z))
        d2 = tf.nn.leaky_relu(self.fd2(d1))
        d3 = tf.nn.l2_normalize(d2, axis=1)  # Normalize along axis 1 (feature dimension)
        return d3

    def feat_decode(self, z):
        d1 = tf.nn.relu(self.fd_x1(z))
        d2 = tf.nn.leaky_relu(self.fd_x2(d1))
        d3 = tf.nn.l2_normalize(d2, axis=1)  # Normalize along axis 1 (feature dimension)
        return d3

    def label_forward(self, x, feat):  # x is label
        if self.args.reg == "gmvae":
            n_label = tf.shape(x)[1]  # label_dim
            all_labels = tf.eye(n_label, dtype=x.dtype)  # [label_dim, label_dim]
            fe_output = self.label_encode(all_labels)  # map each label to a Gaussian mixture.
        else:
            fe_output = self.label_encode(x)

        mu = fe_output['fe_mu']
        logvar = fe_output['fe_logvar']

        if self.args.reg == "wae" or not self.training:
            if self.args.reg == "gmvae":
                z = tf.linalg.matmul(x, mu) / (tf.reduce_sum(x, axis=1, keepdims=True) + 1e-8)
            else:
                z = mu
        else:
            if self.args.reg == "gmvae":
                z = tf.linalg.matmul(x, mu) / (tf.reduce_sum(x, axis=1, keepdims=True) + 1e-8)  # mu of Gaussian Mixture
            else:
                z = self.label_reparameterize(mu, logvar)
        
        label_emb = self.label_decode(tf.concat([feat, z], axis=1))

        single_label_emb = tf.nn.l2_normalize(self.label_recon(mu), axis=1)  # [label_dim, emb_size]

        fe_output['label_emb'] = label_emb
        fe_output['single_label_emb'] = single_label_emb
        return fe_output

    def feat_forward(self, x):
        fx_output = self.feat_encode(x)
        mu = fx_output['fx_mu']  # [bs, latent_dim]
        logvar = fx_output['fx_logvar']  # [bs, latent_dim]

        if self.args.reg == "wae" or not self.training:
            if self.args.test_sample:
                z = self.feat_reparameterize(mu, logvar)
                z2 = self.feat_reparameterize(mu, logvar)
            else:
                z = mu
                z2 = mu
        else:
            z = self.feat_reparameterize(mu, logvar)  # [bs, latent_dim]
            z2 = self.feat_reparameterize(mu, logvar)  # [bs, latent_dim]

        feat_emb = self.feat_decode(tf.concat([x, z], axis=1))  # [bs, emb_size]
        feat_emb2 = self.feat_decode(tf.concat([x, z2], axis=1))  # [bs, emb_size]

        fx_output['feat_emb'] = feat_emb
        fx_output['feat_emb2'] = feat_emb2

        feat_recon = self.recon(z)
        fx_output['feat_recon'] = feat_recon
        return fx_output
    

    def call(self, input_feat, input_label_count, input_label_binary, training=False):
        # Forward pass through the feature and label encoder
        fe_output = self.label_forward(input_label_binary, input_feat)  # or input_label_count if regression mode
        label_emb, single_label_emb = fe_output['label_emb'], fe_output['single_label_emb']
        
        fx_output = self.feat_forward(input_feat)
        feat_emb, feat_emb2 = fx_output['feat_emb'], fx_output['feat_emb2']
        
        # Apply the decoding steps or further processing as required
        embs = self.fe0.weights[0]
        embs = tf.transpose(embs)
        label_out = tf.linalg.matmul(label_emb, embs)
        single_label_out = tf.linalg.matmul(single_label_emb, embs)
        
        feat_out = tf.linalg.matmul(feat_emb, embs)
        feat_out2 = tf.linalg.matmul(feat_emb2, embs)

        fe_output.update(fx_output)
        output = fe_output
        # Return the outputs from the forward pass
        output = {
            'embs': embs,
            'label_out': label_out,
            'single_label_out': single_label_out,
            'feat_out': feat_out,
            'feat_out2': feat_out2,
            'feat': input_feat
        }
        
        output.update(fe_output)  # Add the feature encoder outputs to the final output
        return output

    


def compute_loss(input_label_binary, input_label_count, output, criterion, args=None, epoch=0,
                 class_weights=None, mode='classification', pred_binary=None):

    def log_sum_exp(x, mask):
        max_x = tf.reduce_max(x, axis=1, keepdims=True)
        new_x = x - max_x
        return tf.squeeze(max_x, axis=1) + tf.math.log(tf.reduce_sum(tf.exp(new_x), axis=1))

    def log_mean_exp(x, mask):
        return log_sum_exp(x, mask) - tf.math.log(tf.reduce_sum(mask, axis=1) + 1e-8)

    def log_normal(x, m, v):
        log_prob = -0.5 * (tf.math.log(v + 1e-8) + tf.square(x - m) / (v + 1e-8))
        return tf.reduce_sum(log_prob, axis=-1)

    def log_normal_mixture(z, m, v, mask=None):
        m = tf.expand_dims(m, axis=0)
        v = tf.expand_dims(v, axis=0)
        m = tf.tile(m, [tf.shape(z)[0], 1, 1])
        v = tf.tile(v, [tf.shape(z)[0], 1, 1])
        
        batch, mix, dim = tf.shape(m)[0], tf.shape(m)[1], tf.shape(m)[2]
        z = tf.reshape(z, (batch, 1, dim))
        z = tf.tile(z, [1, mix, 1])

        indiv_log_prob = log_normal(z, m, v) + tf.ones_like(mask) * (-1e6) * (1.0 - mask)
        log_prob = log_mean_exp(indiv_log_prob, mask)
        return log_prob
    
    fe_out, fe_mu, fe_logvar, label_emb = (output['label_out'], output['fe_mu'], 
                                           output['fe_logvar'], output['label_emb'])
    fx_out, fx_mu, fx_logvar, feat_emb = (output['feat_out'], output['fx_mu'], 
                                          output['fx_logvar'], output['feat_emb'])
    fx_out2, single_label_out = output['feat_out2'], output['single_label_out']
    embs = output['embs']

    feat_recon_loss = 0.0

    fe_sample = tf.linalg.matmul(input_label_binary, fe_mu) / (tf.reduce_sum(input_label_binary, axis=1, keepdims=True) + 1e-8)

    std = tf.exp(0.5 * fx_logvar)
    eps = tf.random.normal(tf.shape(std))
    fx_sample = fx_mu + eps * std
    fx_var = tf.exp(fx_logvar)
    fe_var = tf.exp(fe_logvar)

    kl_loss = tf.reduce_mean(log_normal(fx_sample, fx_mu, fx_var) - log_normal_mixture(fx_sample, fe_mu, fe_var, input_label_binary))

    def supconloss(label_emb, feat_emb, embs, temp=1.0, sample_wise=False):
        if sample_wise:
            loss_func = SupConLoss(temperature=0.1)
            return loss_func(tf.stack([label_emb, feat_emb], axis=1), tf.cast(input_label_binary, tf.float32))

        features = tf.concat([label_emb, feat_emb], axis=0)
        labels = tf.concat([input_label_binary, input_label_binary], axis=0)
        labels = tf.cast(labels, tf.float32)

        n_label = tf.shape(labels)[1]
        emb_labels = tf.eye(n_label)
        mask = tf.linalg.matmul(labels, emb_labels)

        anchor_dot_contrast = tf.linalg.matmul(features, embs) / (temp + 1e-8)
        logits_max = tf.reduce_max(anchor_dot_contrast, axis=1, keepdims=True)
        logits = anchor_dot_contrast - logits_max

        exp_logits = tf.exp(logits)
        log_prob = logits - tf.math.log(tf.reduce_sum(exp_logits, axis=1, keepdims=True) + 1e-8)

        mean_log_prob_pos = tf.reduce_sum(mask * log_prob, axis=1) / (tf.reduce_sum(mask, axis=1) + 1e-8)
        mean_log_prob_neg = tf.reduce_sum((1.0 - mask) * log_prob, axis=1) / (tf.reduce_sum(1.0 - mask, axis=1) + 1e-8)

        loss = -mean_log_prob_pos
        loss = tf.reduce_mean(loss)

        return loss

    

    def compute_BCE(E, input_label_binary):
        # Compute negative log likelihood (BCE loss) for each sample point
        sample_nll = -(tf.math.log(E) * input_label_binary + tf.math.log(1 - E) * (1 - input_label_binary))
        logprob = -tf.reduce_sum(sample_nll, axis=2)

        # Avoid float overflow using the log-sum-exp trick
        maxlogprob = tf.reduce_max(logprob, axis=0)
        Eprob = tf.reduce_mean(tf.exp(logprob - maxlogprob), axis=0)
        nll_loss = tf.reduce_mean(-tf.math.log(Eprob) - maxlogprob)

        return nll_loss

    if mode == 'classification':
        pred_e = tf.sigmoid(fe_out)
        pred_x = tf.sigmoid(fx_out)
        pred_x2 = tf.sigmoid(fx_out2)

        nll_loss = compute_BCE(tf.expand_dims(pred_e, axis=0), input_label_binary)
        nll_loss_x = compute_BCE(tf.expand_dims(pred_x, axis=0), input_label_binary)
        nll_loss_x2 = compute_BCE(tf.expand_dims(pred_x2, axis=0), input_label_binary)

    elif mode == 'regression':
        pred_e = tf.exp(fe_out)
        pred_x = tf.exp(fx_out)
        pred_x2 = tf.exp(fx_out2)
        
        # Compute the loss term using mean squared error or a custom loss
        loss_term_e = tf.reduce_mean(criterion(tf.expand_dims(pred_e, axis=0), input_label_count), axis=-1)  # Reduce shape if needed
        nll_loss = tf.reduce_mean(loss_term_e*tf.expand_dims(input_label_binary, axis=-1)) + 0.01 * tf.reduce_mean(tf.abs(pred_e - input_label_count))
        
        # Similarly for pred_x and pred_x2
        loss_term_x = tf.reduce_mean(criterion(pred_x, input_label_count), axis=-1)
        nll_loss_x = tf.reduce_mean(loss_term_x * tf.expand_dims(input_label_binary, axis=-1)) + 0.01 * tf.reduce_mean(tf.abs(pred_x - input_label_count))

        loss_term_x2 = tf.reduce_mean(criterion(pred_x2, input_label_count), axis=-1)
        nll_loss_x2 = tf.reduce_mean(loss_term_x2 * tf.expand_dims(input_label_binary, axis=-1)) + 0.01 * tf.reduce_mean(tf.abs(pred_x2 - input_label_count))

        
    cpc_loss = supconloss(label_emb, feat_emb, embs, sample_wise=False)
    total_loss = (nll_loss + nll_loss_x + nll_loss_x2) * 10 + kl_loss * 6.0 + 1 * cpc_loss  # + latent_cpc_loss

    return total_loss, nll_loss, nll_loss_x, kl_loss, cpc_loss, pred_e, pred_x
