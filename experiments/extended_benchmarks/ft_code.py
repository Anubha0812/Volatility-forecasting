import timesfm
import gc
import numpy as np
import pandas as pd
from timesfm import patched_decoder
from timesfm import data_loader

from tqdm import tqdm
import dataclasses
import IPython
import IPython.display
import matplotlib as mpl
import matplotlib.pyplot as plt
from huggingface_hub import snapshot_download
import os

import jax
from jax import numpy as jnp
from praxis import pax_fiddle
from praxis import py_utils
from praxis import pytypes
from praxis import base_model
from praxis import optimizers
from praxis import schedules
from praxis import base_hyperparams
from praxis import base_layer
from paxml import tasks_lib
from paxml import trainer_lib
from paxml import checkpoints
from paxml import learners
from paxml import partitioning
from paxml import checkpoint_types

def train_and_evaluate(checkpoint_pre, CHECKPOINT_DIR, train_batches, val_batches, test_batches,num_ts,batch_size):
    # Step 3: Initialize the TimesFm model
   
    for tbatch in tqdm(train_batches.as_numpy_iterator()):
        pass
        
    tfm = timesfm.TimesFm(
        hparams=timesfm.TimesFmHparams(
            backend="gpu",             # Use "gpu" or "cpu"
            per_core_batch_size=32,    # Batch size
            horizon_len=128,           # Forecast horizon
            num_layers=50,             # Number of layers
            use_positional_embedding=False,  # For v1.0 compatibility
            context_len=512,           # Compatible with both v1.0 and v2.0
        ),
        checkpoint=timesfm.TimesFmCheckpoint(
            path=checkpoint_pre,  # Correct argument to specify the checkpoint path
        ),
    )
    
    # PAX shortcuts
    NestedMap = py_utils.NestedMap
    WeightInit = base_layer.WeightInit
    WeightHParams = base_layer.WeightHParams
    InstantiableParams = py_utils.InstantiableParams
    JTensor = pytypes.JTensor
    NpTensor = pytypes.NpTensor
    WeightedScalars = pytypes.WeightedScalars
    instantiate = base_hyperparams.instantiate
    LayerTpl = pax_fiddle.Config[base_layer.BaseLayer]
    AuxLossStruct = base_layer.AuxLossStruct

    AUX_LOSS = base_layer.AUX_LOSS
    template_field = base_layer.template_field

    # Standard prng key names
    PARAMS = base_layer.PARAMS
    RANDOM = base_layer.RANDOM

    key = jax.random.PRNGKey(seed=1234)   

    model = pax_fiddle.Config(
        patched_decoder.PatchedDecoderFinetuneModel,
        name='patched_decoder_finetune',
        core_layer_tpl=tfm.model_p,
    )


    @pax_fiddle.auto_config
    def build_learner() -> learners.Learner:
        return pax_fiddle.Config(
          learners.Learner,
          name='learner',
          loss_name='avg_qloss',
          optimizer=optimizers.Adam(
              epsilon=1e-7,
              clip_threshold=1e2,
              learning_rate=1e-2,
              lr_schedule=pax_fiddle.Config(
                  schedules.Cosine,
                  initial_value=1e-3,
                  final_value=1e-4,
                  total_steps=40000,
              ),
              ema_decay=0.9999,
          ),
          # Linear probing i.e we hold the transformer layers fixed.
          bprop_variable_exclusion=['.*/stacked_transformer_layer/.*'],
      )


    task_p = tasks_lib.SingleTask(
        name='ts-learn',
        model=model,
        train=tasks_lib.SingleTask.Train(
            learner=build_learner(),
        ),
    )

    task_p.model.ici_mesh_shape = [1, 1, 1]
    task_p.model.mesh_axis_names = ['replica', 'data', 'mdl']

    DEVICES = np.array(jax.devices()).reshape([1, 1, 1])
    MESH = jax.sharding.Mesh(DEVICES, ['replica', 'data', 'mdl'])

    num_devices = jax.local_device_count()
    print(f'num_devices: {num_devices}')
    print(f'device kind: {jax.local_devices()[0].device_kind}')


    jax_task = task_p
    key, init_key = jax.random.split(key)

    # To correctly prepare a batch of data for model initialization (now that shape
    # inference is merged), we take one devices*batch_size tensor tuple of data,
    # slice out just one batch, then run the prepare_input_batch function over it.


    def process_train_batch(batch):
        past_ts = batch[0].reshape(batch_size * num_ts, -1)
        actual_ts = batch[3].reshape(batch_size * num_ts, -1)
        return NestedMap(input_ts=past_ts, actual_ts=actual_ts)


    def process_eval_batch(batch):
        past_ts = batch[0]
        actual_ts = batch[3]
        return NestedMap(input_ts=past_ts, actual_ts=actual_ts)


    jax_model_states, _ = trainer_lib.initialize_model_state(
        jax_task,
        init_key,
        process_train_batch(tbatch),
        checkpoint_type=checkpoint_types.CheckpointType.GDA,
    )

    jax_model_states.mdl_vars['params']['core_layer'] = tfm._train_state.mdl_vars['params']
    jax_vars = jax_model_states.mdl_vars
    gc.collect()

    jax_task = task_p


    def train_step(states, prng_key, inputs):
        return trainer_lib.train_step_single_learner(
          jax_task, states, prng_key, inputs
      )


    def eval_step(states, prng_key, inputs):
        states = states.to_eval_state()
        return trainer_lib.eval_step_single_learner(
          jax_task, states, prng_key, inputs
      )

    key, train_key, eval_key = jax.random.split(key, 3)
    train_prng_seed = jax.random.split(train_key, num=jax.local_device_count())
    eval_prng_seed = jax.random.split(eval_key, num=jax.local_device_count())

    p_train_step = jax.pmap(train_step, axis_name='batch')
    p_eval_step = jax.pmap(eval_step, axis_name='batch')

    replicated_jax_states = trainer_lib.replicate_model_state(jax_model_states)
    replicated_jax_vars = replicated_jax_states.mdl_vars


    best_eval_loss = 1e7
    step_count = 0
    patience = 0
    NUM_EPOCHS = 100
    PATIENCE = 5
    TRAIN_STEPS_PER_EVAL = 1000

    def reshape_batch_for_pmap(batch, num_devices):
        def _reshape(input_tensor):
            bsize = input_tensor.shape[0]
            residual_shape = list(input_tensor.shape[1:])
            nbsize = bsize // num_devices
            return jnp.reshape(input_tensor, [num_devices, nbsize] + residual_shape)

        return jax.tree.map(_reshape, batch)


    for epoch in range(NUM_EPOCHS):
        print(f"__________________Epoch: {epoch}__________________", flush=True)
        train_its = train_batches.as_numpy_iterator()
        if patience >= PATIENCE:
            print("Early stopping.", flush=True)
            break
        for batch in tqdm(train_its):
            train_losses = []
            if patience >= PATIENCE:
                print("Early stopping.", flush=True)
                break
            tbatch = process_train_batch(batch)
            tbatch = reshape_batch_for_pmap(tbatch, num_devices)
            replicated_jax_states, step_fun_out = p_train_step(
                replicated_jax_states, train_prng_seed, tbatch
            )
            train_losses.append(step_fun_out.loss[0])
            if step_count % TRAIN_STEPS_PER_EVAL == 0:
                print(
                    f"Train loss at step {step_count}: {np.mean(train_losses)}",
                    flush=True,
                )
                train_losses = []
                print("Starting eval.", flush=True)
                val_its = val_batches.as_numpy_iterator()
                eval_losses = []
                for ev_batch in tqdm(val_its):
                    ebatch = process_eval_batch(ev_batch)
                    ebatch = reshape_batch_for_pmap(ebatch, num_devices)
                    _, step_fun_out = p_eval_step(
                        replicated_jax_states, eval_prng_seed, ebatch
                    )
                    eval_losses.append(step_fun_out.loss[0])
                mean_loss = np.mean(eval_losses)
                print(f"Eval loss at step {step_count}: {mean_loss}", flush=True)
                if mean_loss < best_eval_loss or np.isnan(mean_loss):
                    best_eval_loss = mean_loss
                    print("Saving checkpoint.")
                    jax_state_for_saving = py_utils.maybe_unreplicate_for_fully_replicated(
                        replicated_jax_states
                    )
                    checkpoints.save_checkpoint(
                        jax_state_for_saving, CHECKPOINT_DIR, overwrite=True
                    )
                    patience = 0
                    del jax_state_for_saving
                    gc.collect()
                else:
                    patience += 1
                    print(f"patience: {patience}")
            step_count += 1


    train_state = checkpoints.restore_checkpoint(jax_model_states, CHECKPOINT_DIR)
    print(train_state.step)
    tfm._train_state.mdl_vars['params'] = train_state.mdl_vars['params']['core_layer']
    tfm.jit_decode()

    mae_losses = []
    mae1_losses = []
    batch_index=1

    forecasts_list1 = []
    forecasts_list2 =[]
    actual_list=[]
    for batch in tqdm(test_batches.as_numpy_iterator()):
        past = batch[0]
        actuals = batch[3]
        _, forecasts1 = tfm.forecast(list(past), [0] * past.shape[0])

        # Save the forecasts for the first column without a suffix
        forecasts = forecasts1[:, 0 : actuals.shape[1], 0]  # First column

        forecast_df = pd.DataFrame(forecasts)
        forecasts_list1.append(forecast_df)

        # Save the forecasts for the remaining 9 columns with suffixes q1 to q9
        #for i in range(1, 10):  # Columns 1 to 9 (q1 to q9)
        i=5
        forecasts = forecasts1[:, 0 : actuals.shape[1], i]
        #forecasts_filename = os.path.join(output_folder, f'vol_2_1_forecasts_q{i}_batch_{batch_index}.csv')
        forecast_df = pd.DataFrame(forecasts)
        forecasts_list2.append(forecast_df)
        #forecast_df.to_csv(forecasts_filename, index=False)

        mae1_losses.append(np.abs(forecasts1[:, 0 : actuals.shape[1], 0] - actuals).mean())
        mae_losses.append(np.abs(forecasts1[:, 0 : actuals.shape[1], 5] - actuals).mean())

        # Save past data
        #past_filename = os.path.join(output_folder, f'index_2_1_input_batch_{batch_index}.csv')
        #past_df = pd.DataFrame(past)
        #past_df.to_csv(past_filename, index=False)

        # Save actuals data

        actuals_df = pd.DataFrame(actuals)
        actual_list.append(actuals_df)

        # Increment batch index
        batch_index += 1
        print(batch_index)

    #df.to_csv(f'stock_{i+1}.csv', index=False)
    combined_forecasts1 = pd.concat(forecasts_list1, ignore_index=True)
    combined_forecasts2 = pd.concat(forecasts_list2, ignore_index=True)
    combined_actuals= pd.concat(actual_list, ignore_index=True)
    return CHECKPOINT_DIR, combined_forecasts1, combined_forecasts2, combined_actuals
