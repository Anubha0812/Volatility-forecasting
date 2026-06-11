## conda activate tfm2
import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['JAX_PMAP_USE_TENSORSTORE'] = 'false'

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

from experiments.extended_benchmarks.ft_code import train_and_evaluate
import concurrent.futures


mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False
# Import necessary libraries and modules
from tensorflow.compat.v1 import ConfigProto, InteractiveSession

# Function to fix GPU configuration
def fix_gpu():
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)

fix_gpu()



##################################################################################################
#os.environ['JAX_PLATFORMS'] = 'cpu'   # Set JAX to use CPU
##################################################################################################
# Define context and prediction length
context_len=128
pred_len = 1


output_folder="incremental_ft"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

#################################################################################################################################################
# Path to the dataset
data_path = "datasets/voldata.csv"

# Step 1: Read the CSV file containing all symbols
df1 = pd.read_csv(data_path)
df1['Date'] = pd.to_datetime(df1['Date'], format='%d/%m/%y')

# Define the cutoff date
cutoff_date = pd.to_datetime('2021-12-31')

# Filter rows based on context
filtered_df1 = df1[df1['Date'] <= cutoff_date]
# Step 2: Get the list of unique symbols
#symbols= ['.KS11','.FCHI']
#symbols=['.AEX', '.AORD', '.BFX', '.BVSP', '.DJI','.FCHI','.FTSE', '.HSI', '.IBEX','.IXIC']
#symbols=['.AEX', '.AORD', '.BFX', '.BVSP', '.DJI','.FCHI','.FTSE','.GDAXI', '.HSI', '.IBEX','.IXIC', '.KS11','.KSE', '.MXX',
#symbols= ['.FTSE', '.GDAXI', '.HSI', '.IBEX','.IXIC', '.KS11'] 
#symbols= ['.KSE', '.MXX', '.N225','.RUT','.SPX', '.SSEC','.SSMI','.STI','.STOXX50E'][]
#.AEX', '.AORD', '.BFX', '.BVSP', '.DJI','.FCHI','.FTSE', '.GDAXI', '.HSI', 
        # '.IBEX','.IXIC', '.KS11','.KSE',
symbols=[ '.KS11','.KSE','.MXX', '.N225','.RUT','.SPX', '.SSEC','.SSMI','.STI','.STOXX50E']
#'.AEX', '.AORD', '.BFX', '.BVSP', '.DJI','.FCHI','.FTSE', '.GDAXI', '.HSI', 
        # '.IBEX','.IXIC', 
# #################################################################################################################################################

def process_symbol(symbol):
    print(f"Processing symbol: {symbol}")
    stop=0
    local_dir = "/home/vrango/tfm"
    cache_dir = "/home/vrango/.cache"
    repo_id = "google/timesfm-2.0-500m-jax"

    # Download the snapshot
    snapshot_download(local_dir=local_dir, cache_dir=cache_dir, repo_id=repo_id)

    # Step 2: Use the checkpoint path
    checkpoint_pre= os.path.join(local_dir, "checkpoints")

    
    combined_forecasts1_all = pd.DataFrame()
    combined_forecasts2_all = pd.DataFrame()
    combined_actuals_all = pd.DataFrame()

    filtered_df = filtered_df1[filtered_df1['Symbol'] == symbol]
    selected_columns = ['Date', 'rv5_ss']
    df = filtered_df[selected_columns]
    #df['rv5_ss'] = np.log(df['rv5_ss'])
    total_data_points = len(df)
    
    symbol_cleaned = symbol.replace('.', '')
    csv_file_path = f'{symbol_cleaned}_var_data.csv'
    df.to_csv(csv_file_path, index=False)

    initial_train_percent = 0.4
    initial_val_percent = 0.1
    initial_test_percent = 0.2

    # Subsequent percentages
    subsequent_test_percent=0.2
    train_val_ratio = 0.8  # Train to validation ratio
    train_end = int(total_data_points * initial_train_percent)
    val_end = train_end + int(total_data_points * initial_val_percent)
    test_end = val_end + int(total_data_points * initial_test_percent)
    j=1
    boundaries=[0, train_end, val_end, test_end]
    while test_end <= total_data_points and stop <= 1:  # defin a loop that will extract the data and do the fine-turning
        # Prepare time series columns and other parameters
        ts_cols = [col for col in df.columns if col != "Date"]
        num_ts = len(ts_cols)
        batch_size = 16
        # Initialize the data loader for this symbol
        dtl = data_loader.TimeSeriesdata(
            data_path=csv_file_path,
            datetime_col="Date",
            num_cov_cols=None,
            cat_cov_cols=None,
            ts_cols=np.array(ts_cols),
            train_range=[boundaries[0], boundaries[1]],
            val_range=[boundaries[1], boundaries[2]],
            test_range=[boundaries[2], boundaries[3]],
            hist_len=context_len,
            pred_len=pred_len,
            batch_size=num_ts,
            freq="D",
            normalize=False,
            epoch_len=None,
            holiday=False,
            permute=True,
          )
        train_batches1 = dtl.tf_dataset(mode="train", shift=1).batch(batch_size)
        for tbatch in tqdm(train_batches1.as_numpy_iterator()):
            pass
        total_batches = sum(1 for _ in train_batches1)
        train_batches = train_batches1.skip(0).take(total_batches - 1) 
        for tbatch in tqdm(train_batches.as_numpy_iterator()):
            pass
        val_batches = dtl.tf_dataset(mode="val", shift=pred_len)
        test_batches = dtl.tf_dataset(mode="test", shift=pred_len)
        
        # Symbol-specific checkpoint directory
        CHECKPOINT_DIR = os.path.join('var_incre', symbol, f'iteration_{j}')
        # Define the directory to be removed
        if j == 3:
            target_dir = os.path.join('var_incre', symbol, 'iteration_1')

            # Check if the directory exists
            if os.path.exists(target_dir):
                try:
                    shutil.rmtree(target_dir)  # Remove the directory and its contents
                    print(f"Directory {target_dir} has been removed.")
                except Exception as e:
                    print(f"Failed to delete {target_dir}. Reason: {e}")
            else:
                print(f"Directory {target_dir} does not exist.")
        j=j+1
        #CHECKPOINT_DIR = "path/to/output/dir" # the directory wwhere you want to save
        print("entering..")
        
                    
        tfm = timesfm.TimesFm(
            hparams=timesfm.TimesFmHparams(
                backend="gpu",             # Use "gpu" or "cpu"
                per_core_batch_size=32,    # Batch size
                horizon_len=128,           # Forecast horizon
                num_layers=50,             # Number of layers
                use_positional_embedding=False,  # For v1.0 compatibility
                context_len=context_len,           # Compatible with both v1.0 and v2.0
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
        checkpoint_pre=CHECKPOINT_DIR
        
        
        
        combined_forecasts1_all = pd.concat([combined_forecasts1_all, combined_forecasts1], ignore_index=True)
        combined_forecasts2_all = pd.concat([combined_forecasts2_all, combined_forecasts2], ignore_index=True)
        combined_actuals_all = pd.concat([combined_actuals_all, combined_actuals], ignore_index=True)        
        
        
        train_start = boundaries[2]  # Start of the previous test
        train_end = train_start + int((boundaries[3] - boundaries[2]) * train_val_ratio)

        val_start = train_end
        val_end = test_end

        test_start = val_end
        test_end = test_start + int(total_data_points * subsequent_test_percent)

        if test_end > total_data_points:
            test_end = total_data_points
            stop=stop+1          
        boundaries= [train_start, train_end, val_end, test_end]
    
    
    forecasts_filename1 = os.path.join(output_folder, f'{symbol}_forecasts_mean.csv')
    forecasts_filename2 = os.path.join(output_folder, f'{symbol}_forecasts_median.csv')
    actuals_filename = os.path.join(output_folder, f'{symbol}_actuals.csv')
    
    combined_forecasts1_all = pd.concat([combined_forecasts1_all, combined_actuals_all], axis=1)  #DERRICK
    combined_forecasts2_all = pd.concat([combined_forecasts2_all, combined_actuals_all], axis=1)   #DERRICK

    combined_forecasts1_all.to_csv(forecasts_filename1, index=False)
    combined_forecasts2_all.to_csv(forecasts_filename2, index=False)
    combined_actuals_all.to_csv(actuals_filename, index=False)

    print(f"Finished processing symbol: {symbol}")
    print("---------------------------------------------------")




with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor: # If this is not working comment this and uncomment the below line and try 
#with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:

    futures = {executor.submit(process_symbol, symbol): symbol for symbol in symbols}
    # Wait for all the futures to complete
    for future in concurrent.futures.as_completed(futures):
        symbol = futures[future]
        try:
            future.result()  # If the future raised an exception, it will be raised here
        except Exception as exc:
            print(f"Symbol {symbol} generated an exception: {exc}")
        else:
            print(f"Symbol {symbol} processed successfully")

        
        