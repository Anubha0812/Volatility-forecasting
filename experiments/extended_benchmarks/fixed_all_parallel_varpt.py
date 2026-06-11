## conda activate tfm2
## mean -- context length=2048   ### it stopped.. if needed then only run otherwise no need
## mean1 -- context length=512
## mean2 -- context length=128
## mean3 -- context length=64
## conda activate tfm2
## mxx is left '.RUT',
import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['JAX_PMAP_USE_TENSORSTORE'] = 'false'

import shutil
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



# Define output folders
output_folder = "var_only_pt"
#output_folder2="var_par_ft"
# Initialize the model
# Path to the dataset
data_path = "datasets/voldata.csv"

# Step 1: Read the CSV file containing all symbols
df1 = pd.read_csv(data_path)
df1['Date'] = pd.to_datetime(df1['Date'], format='%d/%m/%y')

# Define the cutoff date
cutoff_date = pd.to_datetime('2021-12-31')

# Filter rows based on date
filtered_df1 = df1[df1['Date'] <= cutoff_date]
# Step 2: Get the list of unique symbols
#symbols = filtered_df1['Symbol'].unique()
#symbols=['.AORD', '.BFX', '.BSESN','.DJI']
# symbols=['.AEX', '.AORD', '.BFX', '.BVSP', '.DJI',
#          '.FCHI', '.FTSE', '.GDAXI', '.HSI', '.IBEX','.IXIC', '.KS11', 
#          '.KSE', '.MXX', '.N225','.RUT','.SPX', '.SSEC','.SSMI','.STI',
symbols=[ '.SSEC','.SSMI','.STI','.STOXX50E']
#'.AEX', '.AORD', '.BFX', '.BVSP', '.DJI','.FCHI','.FTSE', '.GDAXI', '.HSI', 
        # '.IBEX','.IXIC', '.KS11','.KSE','.MXX', '.N225','.RUT','.SPX',
#symbols=['.AORD', '.BFX', '.BVSP']
###############################################################################################################################
### fine tuning part

import concurrent.futures
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



pred_len = 1


# Step 3: Loop through each symbol and process
def process_symbol(symbol,context_length):
    print(f"Processing symbol: {symbol}")
    context_len=context_length
    
    # Step 1: Download the checkpoint manually
    local_dir = "/home/vrango/tfm"
    cache_dir = "/home/vrango/.cache"
    repo_id = "google/timesfm-2.0-500m-jax"

    # Download the snapshot
    snapshot_download(local_dir=local_dir, cache_dir=cache_dir, repo_id=repo_id)

    # Step 2: Use the checkpoint path
    checkpoint_path = os.path.join(local_dir, "checkpoints")

    # Step 3: Initialize the TimesFm model
    tfm = timesfm.TimesFm(
        hparams=timesfm.TimesFmHparams(
            backend="gpu",             # Use "gpu" or "cpu"
            per_core_batch_size=32,    # Batch size
            horizon_len=128,           # Forecast horizon
            num_layers=50,             # Number of layers
            use_positional_embedding=False,  # For v1.0 compatibility
            context_len=512*4,           # Compatible with both v1.0 and v2.0
        ),
        checkpoint=timesfm.TimesFmCheckpoint(
            path=checkpoint_path,  # Correct argument to specify the checkpoint path
        ),
    )
        # Filter data for the current symbol
    filtered_df = filtered_df1[filtered_df1['Symbol'] == symbol]

    # Select the required columns
    #selected_columns = ['Date', 'rk_th2','medrv','rsv','rv5', 'rk_twoscale','rsv_ss','bv_ss', 
     #                   'rk_parzen','bv', 'rv10', 'rv10_ss','rv5_ss']
    selected_columns = ['Date', 'rv5_ss']
    
    df = filtered_df[selected_columns]
    
    #print(df.head())
    # Check if there's enough data for training, validation, and testing
    total_data_points = len(df)
    symbol_cleaned = symbol.replace('.', '')

    # Construct the file name
    csv_file_path = f'{symbol_cleaned}_var_data.csv'

    # Save to CSV
    df.to_csv(csv_file_path, index=False)
    # Calculate the indices for the boundaries
    train_size = 0.4  # 70% of the total data for training
    val_size = 0.1    # 10% of the total data for validation
    test_size = 0.5   # 20% of the total data for testing

    # Calculate the indices for boundaries
    train_end_index = int(total_data_points * train_size)
    val_end_index = int(total_data_points * (train_size + val_size))
    boundaries = [train_end_index, val_end_index, total_data_points]

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
        train_range=[0, boundaries[0]],
        val_range=[boundaries[0], boundaries[1]],
        test_range=[boundaries[1], boundaries[2]],
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
    #print(tbatch[0].shape,tbatch[3].shape)

    total_batches = sum(1 for _ in train_batches1)

    train_batches = train_batches1.skip(0).take(total_batches - 1) 
    for tbatch in tqdm(train_batches.as_numpy_iterator()):
        pass
    #print(tbatch[0].shape,tbatch[3].shape)

    val_batches = dtl.tf_dataset(mode="val", shift=pred_len)

    test_batches = dtl.tf_dataset(mode="test", shift=pred_len)

    ######################################################### Performing Inference
    mae_losses = []
    mae1_losses = []
    batch_index=1

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

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
        forecast_df=forecast_df.T
        forecasts_list1.append(forecast_df)
        #forecast_df.to_csv(forecasts_filename, index=False)
        # Save the forecasts for the remaining 9 columns with suffixes q1 to q9
        # for i in range(1, 10):  # Columns 1 to 9 (q1 to q9)
        i=5
        forecasts = forecasts1[:, 0 : actuals.shape[1], i]
        #forecasts_filename = os.path.join(output_folder, f'vol_2_1_forecasts_q{i}_batch_{batch_index}.csv')
        forecast_df = pd.DataFrame(forecasts)
        forecast_df=forecast_df.T
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
        actuals_df= actuals_df.T
        actual_list.append(actuals_df)
        # Increment batch index
        batch_index += 1
        #print(batch_index)

    # Combine and save forecasts and actuals
    combined_forecasts1 = pd.concat(forecasts_list1, ignore_index=True)
    combined_forecasts2 = pd.concat(forecasts_list2, ignore_index=True)
    combined_actuals = pd.concat(actual_list, ignore_index=True)

    forecasts_filename1 = os.path.join(output_folder, f'{symbol}_{context_len}_forecasts_mean.csv')
    #forecasts_filename2 = os.path.join(output_folder, f'{symbol}_forecasts_median.csv')
    #actuals_filename = os.path.join(output_folder, f'{symbol}_actuals.csv')

    combined_forecasts1.to_csv(forecasts_filename1, index=False)
    #combined_forecasts2.to_csv(forecasts_filename2, index=False)
    #combined_actuals.to_csv(actuals_filename, index=False)

    # Print Mean Absolute Errors
    #print(f"Symbol: {symbol}")
    #print(f"MAE: {np.mean(mae_losses)}")
    #print(f"MAE1: {np.mean(mae1_losses)}")
    print(f"Finished processing symbol: {symbol}")
    print("---------------------------------------------------")

    
import concurrent.futures

# Define context lengths
context_lengths = [64, 128, 512]

# Using ThreadPoolExecutor or ProcessPoolExecutor
with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
    # If ThreadPoolExecutor doesn't work, try the below ProcessPoolExecutor line
    # with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:

    # Submit tasks for each symbol and context length
    futures = {
        executor.submit(process_symbol, symbol, context_length): (symbol, context_length)
        for symbol in symbols
        for context_length in context_lengths
    }

    # Wait for all the futures to complete
    for future in concurrent.futures.as_completed(futures):
        symbol, context_length = futures[future]
        try:
            future.result()  # If the future raised an exception, it will be raised here
        except Exception as exc:
            print(f"Symbol {symbol} with context_length {context_length} generated an exception: {exc}")
        else:
            print(f"Symbol {symbol} with context_length {context_length} processed successfully")

print("Finetuned...yayyy")
