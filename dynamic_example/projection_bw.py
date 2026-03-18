import pickle
import argparse
from pathlib import Path
from functools import partial
from matplotlib import pyplot as plt
import jax
# jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.random as jr
import optax
from flax.training import train_state
import numpy as onp
import scipy
import jax.numpy as jnp
from tqdm import tqdm

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
    cfg.iters = 50_000
    cfg.lr = 1e-4
    cfg.alpha = 1.0         # lr reduction for cosine scheduling
    cfg.clip = 1.0
    cfg.warmup_iters = 0
    cfg.skip_loss = 500     # skipped from the loss computation
    cfg.same_sys = 10       # how many steps to hold the same batch

    # Setup JAX device
    jax.config.update("jax_default_device", jax.devices("gpu")[1])

    # Initialize Wandb
    if cfg.log_wandb:
        wandb.init(
            project="sysid-VAE-manifold-meta",
            name="run_projection_bw",
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

    enc = Encoder(mlp_layers=[cfg.nh, cfg.nz], rnn_size=cfg.nh)
    proj = Projector(outputs=n_params,  unflatten=unflatten_dec)

    params_enc = enc.init(
        enc_key,
        jnp.ones((cfg.seq_len, cfg.ny)),#, dtype=jnp.float64), 
        jnp.ones((cfg.seq_len, cfg.nu))#), dtype=jnp.float64)
    )

    params_proj = proj.init(enc_key, jnp.ones((cfg.nz,)))

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
    # params_log_sigma = jnp.zeros(()) 

    # Add it to your flat tuple of parameters
    params_all = (params_enc, params_dec, params_proj)
    
    def instance_loss_fn(p, y1, u1, y2, u2):

        p_enc, p_ss, p_proj = p

        # Encode (y1, u1) in z and project
        z = enc.apply(p_enc, y1, u1)
        # p_ss_proj = proj.apply(p_proj, z)

        # Sum the base ss parameters
        # p = jax.tree.map(lambda x, y: x+y, p_ss, p_ss_proj)

        # Simulate
        y2_hat = dec_emb_apply(z,p_ss,p_proj,u2)

        # Compute loss
        err = y2 - y2_hat
        loss = jnp.mean(err[cfg.skip_loss:] ** 2)
        return loss


    # batched loss
    def loss_fn(*args):
        loss = jax.vmap(instance_loss_fn, in_axes=(None, 0, 0, 0, 0))(*args)
        return jnp.mean(loss)

    opt = optax.chain(
    optax.clip(cfg.clip),
    optax.adam(learning_rate=cfg.lr),
    )
    state = train_state.TrainState.create(apply_fn=loss_fn, params=params_all, tx=opt)

    @jax.jit
    def make_step(state, y1, u1, y2, u2):
            loss, grads = jax.value_and_grad(state.apply_fn)(state.params, y1, u1, y2, u2)
            state = state.apply_gradients(grads=grads)
            return loss, state
    # Add this helper function before your loop
    def is_finite(tree):
        leaves = jax.tree_util.tree_leaves(tree)
        return jnp.all(jnp.array([jnp.all(jnp.isfinite(l)) for l in leaves]))

    LOSS = []
    loss = jnp.array(jnp.nan)
    #for itr, batch in (pbar := tqdm(enumerate(train_dl), total=cfg.iters)):

    for itr in (pbar := tqdm(range(cfg.iters))):

        if itr % cfg.same_sys == 0: # some speed up
            batch_train = next(iter(train_dl))
            batch_y1, batch_u1, batch_y2, batch_u2 = preproc_batch(batch_train)

        loss, new_state = make_step(state, batch_y1, batch_u1, batch_y2, batch_u2)
        if not jnp.isnan(loss).any() and loss < 2.0 and is_finite(new_state.params):
            state = new_state

        LOSS.append(loss.item())
        if itr % 10 == 0:
            pbar.set_postfix_str(
                f"loss:{loss.item():.4f}"
            )

        #if itr % 100 == 0 and cfg.log_wandb:
        if cfg.log_wandb:
            if itr % 1 == 0:
                wandb.log({"mse": loss.item()})

        if itr % 5000 == 0:
            ckpt = {
                "cfg": cfg,
                "params": state.params,
                "scalers": scalers,
                "LOSS": jnp.array(LOSS),
            }
            ckpt_path = Path("tmp") / f"hypernet2_{itr}.p"
            ckpt_path.parent.mkdir(exist_ok=True, parents=True)
            pickle.dump(ckpt, open(ckpt_path, "wb"))


        if itr == cfg.iters:
            break

    params_enc_opt, params_dec_opt, params_proj_opt = state.params

    # Save the final checkpoint
    ckpt = {
        "cfg": cfg,
        "params": state.params,
        "scalers": scalers,
        "LOSS": jnp.array(LOSS),
    }

    ckpt_path = Path("out") / f"hypernet2.p"
    ckpt_path.parent.mkdir(exist_ok=True, parents=True)
    pickle.dump(ckpt, open(ckpt_path, "wb" ))

    if cfg.log_wandb:
        wandb.finish()
    plt.figure()
    plt.plot(LOSS, "k", label="total")
    plt.legend()
    plt.savefig("loss_projection.png")

    z  = enc.apply(params_enc_opt, batch_y1, batch_u1)
    batch_y1_hat =dec_emb_apply(z,params_dec_opt,params_proj_opt,batch_u1)
    batch_y2_hat =dec_emb_apply(z,params_dec_opt,params_proj_opt,batch_u2)

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
    idx = 20
    y1_train = batch_y2[idx]
    y2_train = batch_y2[idx]

    # 3. Plotting
    fig, axs = plt.subplots(2, 1, figsize=(12, 10))
    axs[0].plot(y1_train, "k", label="true")
    axs[0].plot(batch_y2_hat[idx], "b", label="reconstructed")
    axs[0].plot(y2_train - batch_y2_hat[idx], "r", label="reconstruction error")
    axs[1].plot(y2_train, "k", label="true")
    axs[1].plot(batch_y2_hat[idx], "b", label="reconstructed")
    axs[1].plot(y2_train - batch_y2_hat[idx], "r", label="reconstruction error")
    plt.legend()
    plt.savefig("bw_reconstruction.png")

    # 4. Calculate metrics across the whole batch (skipping the first 0: steps if needed)
    # Assuming you want to skip the first 200 steps like in your other cell: cfg.skip_sim
    skip = 0 # Change this to 200 if you want to exclude transient/warmup
    y_true_batch_1 = batch_y1[:, skip:, :]
    y_pred_batch_1 = batch_y1_hat[:, skip:, :]

    y_true_batch_2 = batch_y2[:, skip:, :]
    y_pred_batch_2 = batch_y2_hat[:, skip:, :]

    # Compute fits for every sample in the batch
    fits = calculate_fit(y_true_batch1, y_pred_batch)

    # Get statistics
    mean_fit = jnp.mean(fits)
    best_fit_idx = jnp.argmax(fits)
    worst_fit_idx = jnp.argmin(fits)

    print(f"Mean RMSE: {jnp.sqrt(jnp.mean((y_pred_batch - y_true_batch)**2)):.4f}")
    print(f"Mean Fit (%): {mean_fit:.2f}%")
    print(f"Best Fit (%): {fits[best_fit_idx]:.2f}% (Index: {best_fit_idx})")
    print(f"Worst Fit (%): {fits[worst_fit_idx]:.2f}% (Index: {worst_fit_idx})")

    ################### TESTING

    input_fn = partial(multisine_signal, seq_len=N_test_up, fs=fs_up, fh=cfg.fh, scale=cfg.input_scale)
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
    test_dl = generate_batches(data_test_key)
    batch_test = next(iter(test_dl))
    batch_test_y1, batch_test_u1, batch_test_y2, batch_test_u2 = preproc_batch(batch_test)
    
    def test_instance_loss_fn(z, y, u):
        # z: (latent_dim,) | x: (N, d_x) | y: (N, d_y)
        y2_hat = dec_emb_apply(z, params_dec_opt, params_proj_opt, u)
        err = y - y2_hat
        mse_loss = jnp.mean(err[200:] ** 2)

        # reg_loss = 0.5 * jnp.sum(z**2) / z.shape[0]
        
        loss = mse_loss #+ reg_loss
        return loss
    def test_loss_fn(*args):
        loss = jax.vmap(test_instance_loss_fn, in_axes=(0, 0, 0))(*args)
        return jnp.mean(loss)
    
    z_batch = jax.vmap(enc.apply, in_axes = (None,0,0))(params_enc_opt, batch_test_y1, batch_test_u1)
    opt = optax.chain(
    optax.clip(cfg.clip),
    optax.adam(learning_rate=cfg.lr),
    )
    state = train_state.TrainState.create(apply_fn=test_loss_fn, params=z_batch, tx=opt)
    test_loss = []

    @jax.jit
    def make_step(state, y2, u2):
            loss, grads = jax.value_and_grad(state.apply_fn)(state.params,  y2, u2)
            state = state.apply_gradients(grads=grads)
            return loss, state
    def is_finite(tree):
        leaves = jax.tree_util.tree_leaves(tree)
        return jnp.all(jnp.array([jnp.all(jnp.isfinite(l)) for l in leaves]))

    for itr in (pbar := tqdm(range(int(cfg.iters)))):

        loss, new_state = make_step(state, batch_test_y1, batch_test_u1)
        if not jnp.isnan(loss).any() and loss < 2.0 and is_finite(new_state.params):
            state = new_state

        test_loss.append(loss.item())
        if itr % 10 == 0:
            pbar.set_postfix_str(
                f"test loss:{loss.item():.4f}"
            )
    latent_opt = state.params

    test_loss_latent = test_loss
    plt.figure()
    plt.plot(test_loss, "k", label="total")
    plt.legend()
    plt.savefig("test_loss_projection.png")
    batch_test_y1_hat = dec_emb_apply(latent_opt, params_dec_opt, params_proj_opt, batch_test_u1)
    batch_test_y2_hat = dec_emb_apply(latent_opt, params_dec_opt, params_proj_opt, batch_test_u2)
    print('losses:', test_loss_fn(latent_opt,batch_test_y1,batch_test_u1),test_loss_fn(latent_opt,batch_test_y2,batch_test_u2))
    print(jnp.sqrt(jnp.mean((batch_test_y1_hat - batch_test_y1)**2)), jnp.sqrt(jnp.mean((batch_test_y2_hat - batch_test_y2)**2)))

    # 2. Select index and extract data
    # idx = 60
    y2_test = batch_test_y2[idx]

    # 3. Plotting
    plt.figure()
    plt.plot(y2_test, "k", label="true")
    plt.plot(batch_test_y2_hat[idx], "b", label="reconstructed")
    plt.plot(y2_test - batch_test_y2_hat[idx], "r", label="reconstruction error")
    plt.legend()
    plt.savefig("TEST_bw_reconstruction.png")

    # 4. Calculate metrics across the whole batch (skipping the first 0: steps if needed)
    # Assuming you want to skip the first 200 steps like in your other cell: cfg.skip_sim
    skip = 0 # Change this to 200 if you want to exclude transient/warmup
    y_true_batch = batch_test_y2[:, skip:, :]
    y_pred_batch = batch_test_y2_hat[:, skip:, :]

    # Compute fits for every sample in the batch
    fits = calculate_fit(y_true_batch, y_pred_batch)

    # Get statistics
    mean_fit = jnp.mean(fits)
    best_fit_idx = jnp.argmax(fits)
    worst_fit_idx = jnp.argmin(fits)

    print(f"Mean RMSE: {jnp.sqrt(jnp.mean((y_pred_batch - y_true_batch)**2)):.4f}")
    print(f"TEST Mean Fit (%): {mean_fit:.2f}%")
    print(f"TEST Best Fit (%): {fits[best_fit_idx]:.2f}% (Index: {best_fit_idx})")
    print(f"TEST Worst Fit (%): {fits[worst_fit_idx]:.2f}% (Index: {worst_fit_idx})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Meta Train SysID")
    # You can easily toggle wandb from command line by running: python meta_train.py --no_wandb
    parser.add_argument('--no_wandb', action='store_false', dest='log_wandb', help='Disable wandb logging')
    parser.set_defaults(log_wandb=True)
    
    args = parser.parse_args()
    main(args)
