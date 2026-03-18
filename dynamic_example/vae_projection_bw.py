import pickle
import argparse
from pathlib import Path
from functools import partial
from typing import Callable, List
from matplotlib import pyplot as plt
import jax
# jax.config.update("jax_enable_x64", True)
import jax.nn.initializers as init
import jax.numpy as jnp
import jax.random as jr
import optax
from flax.training import train_state
import numpy as onp
import scipy
import jax.numpy as jnp
from tqdm import tqdm
import flax.linen as nn
from jaxid.common import MLP

# Assuming these are available in your PYTHONPATH
from dataset.input.signals import multisine_signal
import dataset.dynamics.boucwen as dyn
from dataset.simulate import simulate_rk4 as simulate
from dataset.simulate import generate_batch

from neuralss import ss_init, ss_apply
from ae import Encoder, Projector
from lr import create_learning_rate_fn

import wandb

def main(args):
    cfg = argparse.Namespace()
    cfg = argparse.Namespace()
    
    # Misc
    cfg.log_wandb = args.log_wandb
    
    # Meta dataset
    cfg.K = 2               # repetitions from the same system (unused)
    cfg.nu = 1
    cfg.ny = 1
    cfg.seq_len = 1500
    cfg.skip_sim = 500
    cfg.fs = 750.0          # sampling time
    cfg.fh = 150            # highest frequency
    cfg.upsamp = 20         # upsampling for integration
    cfg.input_scale = 50.0
    cfg.output_scale = 7e-4
    
    # Base learner
    cfg.nx = 3
    cfg.hidden_f = 16
    cfg.hidden_g = 16
    
    # Encoder
    cfg.nh = 128
    cfg.nz = 20
    
    # Optimization
    cfg.batch_size = 100    # systems sampled at each meta optimization step
    cfg.iters = 50
    cfg.lr = 1e-4
    cfg.alpha = 1.0         # lr reduction for cosine scheduling
    cfg.clip = 1.0
    cfg.warmup_iters = 0
    cfg.skip_loss = 500     # skipped from the loss computation
    cfg.same_sys = 10       # how many steps to hold the same batch
    MAX_BETA = 0.2

    # Setup JAX device
    jax.config.update("jax_default_device", jax.devices("gpu")[3])

    # Initialize Wandb
    if cfg.log_wandb:
        wandb.init(
            project="sysid-VAE-manifold-meta",
            name=f"run_VAE_bw_beta_{MAX_BETA}",
            # track hyperparameters and run metadata
            config=vars(cfg)
        )

    seed = 12345
    key = jr.key(seed)
    dec_key, proj_key, enc_key, data_train_key,data_test_key, train_key = jr.split(key, 6)

    # Meta dataset definition
    fs_up = cfg.fs * cfg.upsamp
    ts_up = 1.0 / fs_up
    N = cfg.seq_len + cfg.skip_sim
    N_test = 300+ cfg.skip_sim
    N_up = N * cfg.upsamp
    N_test_up = N_test * cfg.upsamp

    input_fn = partial(multisine_signal, seq_len=N_up, fs=fs_up, fh=cfg.fh, scale=cfg.input_scale)
    simulate_fn = jax.jit(partial(simulate, f_xu=dyn.f_xu))
    generate_batch_ = partial(
        generate_batch,
        init_fn=dyn.init_fn,  # random initial state
        input_fn=input_fn,  # random input
        params_fn=dyn.params_fn,  # random system parameters
        simulate_fn=simulate_fn,  # simulation function
    )


    def generate_batches(key, batch_size=cfg.batch_size, K=cfg.K):
        generate_batch_cfg = jax.jit(partial(generate_batch_, systems=batch_size, runs=K))
        while True:
            key, subkey = jr.split(key, 2)
            yield generate_batch_cfg(subkey)


    def preproc_batch(batch):
        batch_u, batch_x, batch_t, batch_params = batch
        batch_y = batch_x[..., [0]]

        batch_u /= cfg.input_scale
        batch_y /= cfg.output_scale

        if cfg.upsamp > 1:
            batch_u = scipy.signal.decimate(batch_u, q=cfg.upsamp, axis=-2)
            batch_y = scipy.signal.decimate(batch_y, q=cfg.upsamp, axis=-2)

        batch_y1 = batch_y[:, 0, cfg.skip_sim:]
        batch_u1 = batch_u[:, 0, cfg.skip_sim:]

        batch_y2 = batch_y[:, 1, cfg.skip_sim:]
        batch_u2 = batch_u[:, 1, cfg.skip_sim:]

        # batch_u = batch_u.astype(jnp.float64)
        # batch_y = batch_y.astype(jnp.float64)
        return batch_y1, batch_u1, batch_y2, batch_u2
    
    # Initialize data loader
    train_dl = generate_batches(data_train_key)
    batch_train = next(iter(train_dl))
    batch_y1, batch_u1, batch_y2, batch_u2 = preproc_batch(batch_train)

    params_dec = ss_init(dec_key, nu=cfg.nu, ny=cfg.ny, nx=cfg.nx)
    params_dec_flat, unflatten_dec = jax.flatten_util.ravel_pytree(params_dec)
    n_params = params_dec_flat.shape[0]
    scalers = {"f": {"lin": 1e-2, "nl": 1e-2}, "g": {"lin": 1e0, "nl": 1e0}}
    #params_dec = {"base_learner": params_dec, "projection": jr.normal(proj_key, shape=(params_dec_flat.shape[0], cfg.nz))}
        
    class Encoder(nn.Module):
        # output_size: int
        mlp_layers: List[int]
        rnn_size: int = 128


        def setup(self):
            self.rnn = nn.Bidirectional(
                nn.RNN(nn.GRUCell(self.rnn_size)), nn.RNN(nn.GRUCell(self.rnn_size))
            )
            self.mlp_mean = MLP(self.mlp_layers)
            self.mlp_logstd = MLP(self.mlp_layers)

        def __call__(self, y, u):
            yu = jnp.concat((y, u), axis=-1)
            rnn_feat = self.rnn(yu).mean(axis=-2)
            # enc_mean = nn.Dense(
            #         self.output_size, 
            #         kernel_init=init.zeros, 
            #         bias_init=init.zeros
            #     )(rnn_feat)
                
            # enc_logstd = nn.Dense(
            #         self.output_size, 
            #         kernel_init=init.zeros, 
            #         bias_init=init.zeros
            #     )(rnn_feat)

            return self.mlp_mean(rnn_feat), self.mlp_logstd(rnn_feat)
            # return enc_mean, enc_logstd

    enc = Encoder(mlp_layers=[cfg.nh, cfg.nz], rnn_size=cfg.nh)
    # enc = Encoder(output_size= cfg.nz, rnn_size=cfg.nh)

    y_dummy = jnp.ones((cfg.seq_len, cfg.ny))
    u_dummy = jnp.ones((cfg.seq_len, cfg.nu))
    params_enc = enc.init(enc_key, y_dummy, u_dummy)
    # This is actually part of the decoder, but we keep it separate...
    class Projector(nn.Module):
        outputs: int
        unflatten: Callable

        def setup(self):
            self.net = nn.Dense(self.outputs, use_bias=False)
        def __call__(self, z):
            return self.unflatten(self.net(z*1e-1))

    z_dummy = jnp.ones((cfg.nz,))
    proj = Projector(outputs=n_params, unflatten=unflatten_dec)
    params_proj = proj.init(enc_key, z_dummy)

    def dec_emb_apply(z, p_dec, p_proj, u):
        """
        z: scalar, (nz,), or (batch, nz)
        u: (seq_len, nu) or (batch, seq_len, nu)
        """
        # Define the base function that processes exactly ONE z and ONE u sequence
        def single_dec_emb_apply(z_single, u_single):
            # Ensure z is at least 1D
            z_single = jnp.atleast_1d(z_single)
            
            # Project the latent into parameters of the decoder
            p_dec_proj = proj.apply(p_proj, z_single) 
            
            # Combine base parameters with projected parameters
            p_new = jax.tree_util.tree_map(lambda x_, y_: x_ + y_, p_dec, p_dec_proj)
            
            # x0 is instantiated per-sample (vmap will naturally batch this)
            x0 = jnp.zeros((cfg.nx, ))
            
            # scalers comes from the outer scope and is automatically unbatched
            y = ss_apply(p_new, scalers, x0, u_single)
            return y


        # 1. Handle scalar z or 1D z (single sample)
        if z.ndim == 0 or z.ndim == 1:
            # If z is unbatched, u must also be unbatched (seq_len, nu)
            if u.ndim == 3:
                raise ValueError("If z is unbatched, u must also be unbatched (seq_len, nu).")
            return single_dec_emb_apply(z, u)

        # Inside dec_emb_apply
        # 2. Batched z (shape: B, nz)
        if z.ndim == 2:
            # If u is unbatched (seq_len, nu), but z is batched, 
            # broadcast u to all z's by using in_axes=(0, None)
            # if u.ndim == 2:
            #     return jax.vmap(single_dec_emb_apply, in_axes=(0, None))(z, u)
                
            # # If u IS batched (B, seq_len, nu), map over both
            # elif u.ndim == 3:
            return jax.vmap(single_dec_emb_apply, in_axes=(0, 0))(z, u)

        
        raise ValueError(f"Unexpected z dimension: {z.ndim}")
    params_log_sigma = jnp.zeros(()) 

    # Add it to your flat tuple of parameters
    params_all = (params_enc, params_dec, params_proj, params_log_sigma)

    def instance_loss_fn(p, y1, u1, y2, u2, key, beta):

        # beta = 1.0 # for beta-vae
        p_enc, p_dec, p_proj, log_sigma_est = p
        # p_enc, p_dec, p_proj = p
        
        # 2. Get the actual variance
        sigma_noise = jnp.exp(log_sigma_est)

        # Process (y1, u1) through the encoder and sample z with reparametrization trick
        enc_mean, enc_logstd = enc.apply(p_enc, y1, u1)
        # enc_logstd = jnp.clip(enc_logstd, -4.0, 2.0)  # FIX 1: clamp logstd
        enc_std = jnp.exp(enc_logstd)
        z = enc_mean + jr.normal(key, enc_mean.shape) * enc_std

        # Project the latent into parameters of the decoder
        # p_dec_proj = proj.apply(p_proj, z) 

        # # Use in the decoder output to define/update the model parameters
        # p = jax.tree.map(lambda x, y: x+y, p_dec, p_dec_proj)


        # x0 = jnp.zeros((cfg.nx, ))
        # y2_hat = model(p, scalers, x0, u2)
        y2_hat = dec_emb_apply(z,p_dec,p_proj,u2)
        # -log p(y | latent) with Gaussian p(y | latent), with fixed and known std
        scaled_err = (y2 - y2_hat)/sigma_noise
        nll_loss = 0.5 * jnp.sum((scaled_err[cfg.skip_loss:] )**2) + y2.shape[0] * log_sigma_est + y2.shape[0]/2 * jnp.log(2*jnp.pi)
        kl_loss = 0.5 * jnp.mean(enc_mean**2 + enc_std**2 - 2 * jnp.log(enc_std) - 1) # KL(N(mean, std^2) || N(0, 1))

        # for better numerical scaling
        scale = 0.5 #* y2.shape[0]
        nll_loss = nll_loss / (scale* y2.shape[0])
        kl_loss = kl_loss / scale

        total_loss = nll_loss + beta * kl_loss

        return total_loss, (nll_loss, kl_loss)



    # batched loss
    def loss_fn(*args):
        loss, (nll_loss, kl_loss) = jax.vmap(instance_loss_fn, in_axes=(None, 0, 0, 0, 0, 0, None))(*args)
        return jnp.mean(loss), (jnp.mean(nll_loss), jnp.mean(kl_loss))

    schedule = optax.cosine_decay_schedule(init_value=5e-3, decay_steps=cfg.iters)
    opt = optax.chain(
    optax.clip(cfg.clip),
    optax.adam(learning_rate=cfg.lr),
    )
    state = train_state.TrainState.create(apply_fn=loss_fn, params=params_all, tx=opt)

    @jax.jit
    def make_step(state, y1, u1, y2, u2, train_key, beta):
            train_key, train_subkey = jr.split(train_key, 2) # subkey to be consumed at the current iteration
            train_keys = jr.split(train_subkey, cfg.batch_size)  # we need one key per sample in the batch
            (loss, (nll_loss, kl_loss)), grads = jax.value_and_grad(state.apply_fn, has_aux=True)(state.params, y1, u1, y2, u2, train_keys, beta)
            state = state.apply_gradients(grads=grads)
            return loss, nll_loss, kl_loss, state, train_key
    
    LOSS = []
    NLL_LOSS = []

    for itr in (pbar := tqdm(range(cfg.iters))):

        if itr % cfg.same_sys == 0: # some speed up
            batch_train = next(iter(train_dl))
            batch_y1, batch_u1, batch_y2, batch_u2 = preproc_batch(batch_train)

        
        # --- NEW: KL Annealing schedule ---
        # Linearly increase beta from 0.0 to 2.0 over the first 1000 iterations
        # current_beta = jnp.minimum(2.0, 2.0 * (itr / cfg.iters))
        current_beta =jnp.minimum(MAX_BETA, jnp.maximum(0.0, MAX_BETA*((itr - 5000) / 15000.0)))


        
        # Pass current_beta to make_step
        loss, nll_loss, kl_loss, state, train_key = make_step(
            state, batch_y1, batch_u1, batch_y2, batch_u2, train_key, current_beta
        )
        
        LOSS.append(loss.item())
        NLL_LOSS.append(nll_loss.item())
        
        if itr % 10 == 0:
            pbar.set_postfix_str(f"loss: {loss.item():.2f} nll: {nll_loss.item():.2f} kl {kl_loss.item():.2f} beta: {current_beta:.2f}")
        
        if cfg.log_wandb:
            if itr % 10 == 0:  # Logging every 10 steps is usually better for performance
                wandb.log({
                    'loss': float(loss),
                    'nll': float(nll_loss),
                    'kl': float(kl_loss),
                    'beta': float(current_beta) # Useful to track your annealing!
                })


        if itr % 5000 == 0:
            params_enc_opt, params_dec_opt, params_proj_opt,log_sigma_est_opt = state.params
            ckpt = {
                "params_enc": params_enc_opt,
                "params_dec": params_dec_opt,
                "params_proj": params_proj_opt,
                "log_sigma_est":log_sigma_est_opt,
                "scalers": scalers,
                "cfg": cfg,
                "LOSS": jnp.array(LOSS),
                "NLL_LOSS": jnp.array(NLL_LOSS),
            }
            tmp_path = Path("tmp") / f"vae_{itr}.p"
            tmp_path.parent.mkdir(exist_ok=True, parents=True)
            pickle.dump(ckpt, open(tmp_path, "wb"))
        
        if itr == cfg.iters:
            break

    params_enc_opt, params_dec_opt, params_proj_opt, log_sigma_est_opt = state.params
    std_opt = jnp.exp(log_sigma_est_opt)

    # Save a checkpoint (using torch utilities)
    LOSS = jnp.array(LOSS)
    NLL_LOSS = jnp.array(NLL_LOSS)
    KL_LOSS = LOSS - NLL_LOSS


    ckpt = {
        "params_enc": params_enc_opt,
        "params_dec": params_dec_opt,
        "params_proj": params_proj_opt,
        "log_sigma_est":log_sigma_est_opt,
        "scalers": scalers,
        "cfg": cfg,
        "LOSS": LOSS,
        "NLL_LOSS": NLL_LOSS
    }

    pickle.dump(ckpt, open("vae{fMAX_BETA}.p", "wb" ))
    #ckpt_load = pickle.load(open("hypernet_bw.p", "rb"))

    if cfg.log_wandb:
        wandb.finish()

    plt.figure()
    plt.plot(LOSS, "k", label="total")
    plt.plot(NLL_LOSS, "b", label="NLL")
    plt.plot(KL_LOSS, "r", label="KL")
    plt.legend()
    plt.savefig(f"vae_loss{MAX_BETA}.png")
    # Process (y1, u1) through the encoder and sample the latent

    enc_mean, enc_logstd = enc.apply(params_enc_opt, batch_y1, batch_u1)
    enc_std = jnp.exp(enc_logstd)
    z = enc_mean + onp.random.randn(*enc_std.shape) * enc_std
    batch_y2_hat =dec_emb_apply(z,params_dec_opt,params_proj_opt,batch_u2)

    print(enc_mean.mean(axis = 0),enc_mean.std(axis = 0))
    # 1. Define the Fit metric function
    def calculate_fit(y_true, y_pred):
        # Ensure shapes match
        y_true = y_true.squeeze()
        y_pred = y_pred.squeeze()
        
        # Calculate the mean of the true signal
        y_mean = jnp.mean(y_true, axis=-1, keepdims=True)
        
        # Calculate L2 norms (or sum of squared errors)
        err_norm = jnp.linalg.norm(y_true - y_pred, axis=-1)
        sig_norm = jnp.linalg.norm(y_true - y_mean, axis=-1)
        
        # Calculate Fit %
        fit = 100.0 * (1.0 - (err_norm / (sig_norm + 1e-8))) # Add 1e-8 to avoid division by zero
        return fit

    # 2. Select index and extract data
    idx = 60
    y1 = batch_y1[idx]
    u1 = batch_u1[idx]
    y2 = batch_y2[idx]
    u2 = batch_u2[idx]

    # 3. Plotting
    plt.figure()
    plt.plot(y2, "k", label="true")
    plt.plot(batch_y2_hat[idx], "b", label="reconstructed")
    plt.plot(y2 - batch_y2_hat[idx], "r", label="reconstruction error")
    plt.legend()
    plt.savefig(f"vae_reconstruction{MAX_BETA}.png")

    # 4. Calculate metrics across the whole batch (skipping the first 0: steps if needed)
    # Assuming you want to skip the first 200 steps like in your other cell: cfg.skip_sim
    skip = 0 # Change this to 200 if you want to exclude transient/warmup
    y_true_batch = batch_y2[:, skip:, :]
    y_pred_batch = batch_y2_hat[:, skip:, :]

    # Compute fits for every sample in the batch
    fits = calculate_fit(y_true_batch, y_pred_batch)

    # Get statistics
    mean_fit = jnp.mean(fits)
    best_fit_idx = jnp.argmax(fits)
    worst_fit_idx = jnp.argmin(fits)

    print(f"Mean RMSE: {jnp.sqrt(jnp.mean((y_pred_batch - y_true_batch)**2)):.4f}")
    print(f"Mean Fit (%): {mean_fit:.2f}%")
    print(f"Best Fit (%): {fits[best_fit_idx]:.2f}% (Index: {best_fit_idx})")
    print(f"Worst Fit (%): {fits[worst_fit_idx]:.2f}% (Index: {worst_fit_idx})")

    # Pass the full meta-test batch through the trained encoder
    # Ensure that 'params_enc' refers to your trained encoder parameters
    # enc_mean, enc_logstd = enc.apply(params_enc, batch_y1, batch_u1)

    # # Convert log standard deviation to actual standard deviation
    # enc_std = jnp.exp(enc_logstd)

    # # 1. Look at the standard deviations for a single test dataset
    # print("Standard Deviations for the first test set (10 dimensions):")
    # print(jnp.round(enc_std[0], 3))

    # # 2. Average the std over the entire meta-test batch 
    # # Dimensions that are shut down will have std ~ 1.0. Active dimensions will have std < 1.0.
    # mean_std_across_datasets = jnp.mean(enc_std, axis=0)

    # print("\nAverage Standard Deviation across all test datasets:")
    # print(jnp.round(mean_std_across_datasets, 3))

    # # 3. Compute the average KL divergence per dimension across the test set
    # # A KL divergence of exactly 0.0 means the dimension is completely shut down (pruned).
    # # Active dimensions will have a KL divergence > 0.
    # kl_per_dim = 0.5 * jnp.mean(enc_mean**2 + enc_std**2 - 2 * jnp.log(enc_std) - 1, axis=0)

    # print("\nAverage KL Divergence per dimension (0 means pruned):")
    # print(jnp.round(kl_per_dim, 3))

    # # Create a dummy test domain for x
    # test_x = jnp.linspace(0, 1200, 1200).reshape(-1, 1)

    # # Base prediction with z entirely zero
    # z_zero = jnp.zeros((batch_u2.shape[0],cfg.nz))
    # base_y =dec_emb_apply(z_zero,params_dec_opt,params_proj_opt,batch_u2)

    # functional_importance = []

    # for i in range(cfg.nz):
    #     # Create a z vector where ONLY dimension i is active
    #     z_perturbed = jnp.zeros((batch_u2.shape[0],cfg.nz))
    #     z_perturbed = z_perturbed.at[i].set(1.0)
        
    #     # Get the network's prediction with this perturbed z
    #     perturbed_y =dec_emb_apply(z_perturbed,params_dec_opt,params_proj_opt,batch_u2)
        
    #     # Measure the Mean Squared Error difference in the actual OUTPUT space
    #     diff = jnp.mean((perturbed_y - base_y)**2)
    #     functional_importance.append(diff)

    # functional_importance = jnp.array(functional_importance)

    # print("Functional importance (MSE change in output per dimension):")
    # print(jnp.round(functional_importance, 6))

    # normalized_func_imp = functional_importance / jnp.max(functional_importance)
    # print("\nRelative functional importance:")
    # print(jnp.round(normalized_func_imp, 3))

    # 1. Grab a REAL multisine input sequence from your test batch
    u_test = batch_u2[0]  # Shape should be (1200, 1)

    # 2. Sample 10 latent vectors from the prior
    z_samples = onp.random.normal(size=(1, cfg.nz))

    # 3. Pass through the neural decoder using your batched function
    ypred = jax.vmap(dec_emb_apply, in_axes=(0, None, None, None))(
        z_samples, params_dec_opt, params_proj_opt, u_test
    )

    # 4. Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(ypred.squeeze(-1).T)
    plt.title("Generated Prior Responses using Real Multisine Input")
    plt.xlabel("Time steps")
    plt.ylabel("Displacement")
    plt.show()
    plt.savefig(f"vae_generated_reconstruction{MAX_BETA}.png")
    

#  ################### TESTING

#     input_fn = partial(multisine_signal, seq_len=N_test_up, fs=fs_up, fh=cfg.fh, scale=cfg.input_scale)
#     simulate_fn = jax.jit(partial(simulate, f_xu=dyn.f_xu))
#     generate_batch_ = partial(
#         generate_batch,
#         init_fn=dyn.init_fn,  # random initial state
#         input_fn=input_fn,  # random input
#         params_fn=dyn.params_fn,  # random system parameters
#         simulate_fn=simulate_fn,  # simulation function
#     )


#     def generate_batches(key, batch_size=cfg.batch_size, K=cfg.K):
#         generate_batch_cfg = jax.jit(partial(generate_batch_, systems=batch_size, runs=K))
#         while True:
#             key, subkey = jr.split(key, 2)
#             yield generate_batch_cfg(subkey)
#     test_dl = generate_batches(data_test_key)
#     batch_test = next(iter(test_dl))
#     batch_test_y1, batch_test_u1, batch_test_y2, batch_test_u2 = preproc_batch(batch_test)
    
#     def test_instance_loss_fn(z, y, u):
#         # z: (latent_dim,) | x: (N, d_x) | y: (N, d_y)
#         y2_hat = dec_emb_apply(z, params_dec_opt, params_proj_opt, u)
#         err = y - y2_hat
#         mse_loss = jnp.mean(err[200:] ** 2)

#         return mse_loss 

#     def test_loss_fn(*args):
#         loss = jax.vmap(test_instance_loss_fn, in_axes=(0, 0, 0))(*args)
#         return jnp.mean(loss)
    
#     z_batch, _ = jax.vmap(enc.apply, in_axes = (None,0,0))(params_enc_opt, batch_test_y1, batch_test_u1)
#     opt = optax.chain(
#     optax.clip(cfg.clip),
#     optax.adam(learning_rate=cfg.lr),
#     )
#     state = train_state.TrainState.create(apply_fn=test_loss_fn, params=z_batch, tx=opt)
#     test_loss = []

#     @jax.jit
#     def make_step(state, y2, u2):
#             loss, grads = jax.value_and_grad(state.apply_fn)(state.params,  y2, u2)
#             state = state.apply_gradients(grads=grads)
#             return loss, state
#     def is_finite(tree):
#         leaves = jax.tree_util.tree_leaves(tree)
#         return jnp.all(jnp.array([jnp.all(jnp.isfinite(l)) for l in leaves]))

#     for itr in (pbar := tqdm(range(int(cfg.iters)))):

#         loss, new_state = make_step(state, batch_test_y1, batch_test_u1)
#         if not jnp.isnan(loss).any() and loss < 2.0 and is_finite(new_state.params):
#             state = new_state

#         test_loss.append(loss.item())
#         if itr % 10 == 0:
#             pbar.set_postfix_str(
#                 f"test loss:{loss.item():.4f}"
#             )
#     latent_opt = state.params

#     test_loss_latent = test_loss
#     plt.figure()
#     plt.plot(test_loss, "k", label="total")
#     plt.legend()
#     plt.savefig(f"vae_test_loss_projection{MAX_BETA}.png")
#     batch_test_y1_hat = dec_emb_apply(latent_opt, params_dec_opt, params_proj_opt, batch_test_u1)
#     batch_test_y2_hat = dec_emb_apply(latent_opt, params_dec_opt, params_proj_opt, batch_test_u2)
#     print('losses:', test_loss_fn(latent_opt,batch_test_y1,batch_test_u1),test_loss_fn(latent_opt,batch_test_y2,batch_test_u2))
#     print(jnp.sqrt(jnp.mean((batch_test_y1_hat - batch_test_y1)**2)), jnp.sqrt(jnp.mean((batch_test_y2_hat - batch_test_y2)**2)))

#     # ---------------------------------------------------------
#     # BULLETPROOF HESSIAN & COVARIANCE COMPUTATION
#     # ---------------------------------------------------------
#         # ---------------------------------------------------------
#     # HESSIAN: MUST BE IDENTICAL TO test_instance_loss_fn
#     # ---------------------------------------------------------

#     print("Computing Hessian...")
#     # 1. Vmap the Hessian computation over the batch dimension
#     vmap_hessian = jax.vmap(jax.hessian(test_instance_loss_fn, argnums=0), in_axes=(0, 0, 0))
#     hessian_scaled = vmap_hessian(latent_opt, batch_test_y1, batch_test_u1) 
#     # hessian_scaled shape: (batch_size, latent_dim, latent_dim)

#     # 2. Vmap the decoder to get MAP predictions for the adaptation set (y1)
#     # vmap_decode = jax.vmap(dec_emb_apply, in_axes=(0, None, None, 0))
#     # y1_hat_map = vmap_decode(latent_opt, params_dec_opt, params_proj_opt, batch_test_u1)

#     # Estimate sigma per trajectory: standard deviation over time (axis 1) and output dims (axis 2)
#     # sigma_est = jnp.std(batch_test_y1 - y1_hat_map, axis=(1, 2)) 
#     # sigma_est shape: (batch_size,)

#     # 3. Mathematically convert the scaled Hessian into the TRUE Bayesian Hessian
#     N = batch_test_y1.shape[1]  # The number of data points (time steps) used in adaptation
#     scale_factors = N / (std_opt ** 2) 
#     # scale_factors shape: (batch_size,)

#     # Multiply each (latent_dim, latent_dim) Hessian by its corresponding scale factor
#     # [:, None, None] adds dummy dimensions so (B,) broadcasts over (B, latent_dim, latent_dim)
#     # print(hessian_scaled.shape,scale_factors.shape)
#     hessian_true = hessian_scaled * scale_factors#[:, None, None]

#     # 4. Add the Prior (Identity matrix) to fix the negative eigenvalues
#     # JAX will automatically broadcast this (latent_dim, latent_dim) matrix to all items in the batch
#     prior_matrix = jnp.eye(latent_opt.shape[-1])
#     hessian_final = hessian_true + prior_matrix

#     # 5. Invert it to get your Covariance!
#     # jnp.linalg.inv natively takes (B, M, M) and returns (B, M, M)
#     posterior_covariances = jnp.linalg.inv(hessian_final)
# ###########
#     # # 2. Define the Hessian function with respect to the first argument (z)
#     # hessian_fn = jax.hessian(test_instance_loss_fn, argnums=0)
    
#     # # 3. Vectorize the Hessian computation across the batch dimension
#     # batched_hessian_fn = jax.vmap(hessian_fn, in_axes=(0, 0, 0))
    
#     # # 4. Compute the Hessian matrices for the whole batch
#     # # Shape of hessian_batch: (batch_size, latent_dim, latent_dim)
#     # hessian_batch = batched_hessian_fn(latent_opt, batch_test_y1, batch_test_u1)
    
#     # # 5. SAFE INVERSION: Eigenvalue Decomposition
#     # eigvals, eigvecs = jnp.linalg.eigh(hessian_batch)
    
#     # # The second derivative of 0.01 * sum(z^2) is exactly 0.02.
#     # # Therefore, we mathematically guarantee no curvature is lower than 0.02!
#     # # This prevents division by zero and entirely eliminates NaNs.
#     # print("eigenvalues_hessian", eigvals)
#     # eigvals = jnp.maximum(eigvals, 1e-2)
    
#     # def make_covariance(v, d):
#     #     return v @ jnp.diag(1.0 / d) @ v.T
        
#     # posterior_covariances = jax.vmap(make_covariance)(eigvecs, eigvals)

#     # # ---------------------------------------------------------
#     # # 4. TEMPERATURE SCALING (THE FIX)
#     # # Because the loss lacks probabilistic constants, the covariance is massive.
#     # # We artificially scale it down by multiplying by a small temperature (e.g., 1e-4 or 1e-5)
#     # # so the ODE solver doesn't explode!
#     # # ---------------------------------------------------------
#     # temperature = 1e-2
#     # posterior_covariances = posterior_covariances * temperature


#     # ---------------------------------------------------------
#     # 6. Sample Predictions
#     def sample_predictions_batched(keys_batch, z_mean_batch, z_cov_batch, x_batch, num_samples=100):
        
#         def sample_predictions_single(key, z_mean, z_cov, x_single):
#             # Sample z: (num_samples, latent_dim)
#             z_samples = jr.multivariate_normal(key, mean=z_mean, cov=z_cov, shape=(num_samples,))
            
#             def predict_single_z(z):
#                 # Shape out: (seq_len, ny)
#                 return dec_emb_apply(z, params_dec_opt, params_proj_opt, x_single)
            
#             # Shape out: (num_samples, seq_len, ny)
#             y_pred_samples = jax.vmap(predict_single_z)(z_samples)
            
#             # CRITICAL: squeeze the feature dimension! (num_samples, seq_len)
#             y_pred_samples = y_pred_samples.squeeze(-1)
            
#             pred_mean = jnp.mean(y_pred_samples, axis=0)
#             pred_std = jnp.std(y_pred_samples, axis=0)
#             return pred_mean, pred_std

#         batch_fn = jax.vmap(sample_predictions_single, in_axes=(0, 0, 0, 0))
        
#         # Execute for the whole batch
#         pred_mean_batch, pred_std_batch = batch_fn(keys_batch, z_mean_batch, z_cov_batch, x_batch)
        
#         return pred_mean_batch, pred_std_batch

#     batch_size = batch_test_u1.shape[0]
#     base_key = jr.PRNGKey(42)
#     keys_batch = jr.split(base_key, batch_size)
#     print("Sampling predictions for target set...")
#     pred_mean_1, pred_std_1 = sample_predictions_batched(keys_batch, latent_opt, posterior_covariances, batch_test_u1)
#     pred_mean_2, pred_std_2 = sample_predictions_batched(keys_batch, latent_opt, posterior_covariances, batch_test_u2)

#     # 7. Evaluate and Plot 
    
#     # HARDCODE IDX to avoid flatline edge-cases
    
#     # Extract flattened sequences
#     y2_test_flat = batch_test_y2[idx].flatten()
#     mean_plot = pred_mean_2[idx].flatten()
#     std_plot = pred_std_2[idx].flatten()
#     map_plot = batch_test_y2_hat[idx].flatten() # The deterministic MAP prediction
#     t_steps = jnp.arange(len(y2_test_flat))

#     plt.figure(figsize=(10, 6))

#     # True points
#     plt.plot(y2_test_flat, "k", label="Target True (y2)")
    
#     # MAP Prediction (This ensures we see exactly what the model learned)
#     plt.plot(map_plot, "g", label="MAP Prediction", linestyle="dashed")
    
#     # Predictive Mean from Sampling
#     plt.plot(mean_plot, "b", label="Sampled Pred Mean")
#     plt.plot(y2_test_flat - mean_plot, "r", label="reconstruction error")
    
#     # Uncertainty Band (± 2 std deviations)
#     plt.fill_between(t_steps, 
#                      mean_plot - 3 * std_plot, 
#                      mean_plot + 3 * std_plot, 
#                      color='blue', alpha=0.3, label="± 2 Std Dev")

#     plt.title(f"Laplace Approximation: Task {idx}")
#     plt.xlabel("Time steps")
#     plt.ylabel("Displacement")
#     plt.legend()
#     plt.show()
#     plt.savefig(f"VAE_test_bw_reconstruction{MAX_BETA}.png")
#     print("Saved VAE_test_bw_reconstruction.png")


#     # 4. Calculate metrics across the whole batch (skipping the first 0: steps if needed)
#     # Assuming you want to skip the first 200 steps like in your other cell: cfg.skip_sim
#     skip = 0 # Change this to 200 if you want to exclude transient/warmup
#     y_true_batch = batch_test_y2[:, skip:, :]
#     y_pred_batch = batch_test_y2_hat[:, skip:, :]

#     # Compute fits for every sample in the batch
#     fits = calculate_fit(y_true_batch, y_pred_batch)

#     # Get statistics
#     mean_fit = jnp.mean(fits)
#     best_fit_idx = jnp.argmax(fits)
#     worst_fit_idx = jnp.argmin(fits)
    
#     # print(mean_plot)
#     print(f"Mean RMSE: {jnp.sqrt(jnp.mean((y_pred_batch - y_true_batch)**2)):.4f}")
#     print(f"TEST Mean Fit (%): {mean_fit:.2f}%")
#     print(f"TEST Best Fit (%): {fits[best_fit_idx]:.2f}% (Index: {best_fit_idx})")
#     print(f"TEST Worst Fit (%): {fits[worst_fit_idx]:.2f}% (Index: {worst_fit_idx})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Meta Train SysID")
    # You can easily toggle wandb from command line by running: python meta_train.py --no_wandb
    parser.add_argument('--no_wandb', action='store_false', dest='log_wandb', help='Disable wandb logging')
    parser.set_defaults(log_wandb=True)
    
    args = parser.parse_args()
    main(args)
