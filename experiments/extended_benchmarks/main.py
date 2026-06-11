## conda activate tfm2
## mean -- context length=2048   ### it stopped.. if needed then only run otherwise no need
## mean1 -- context length=512
## mean2 -- context length=128
## mean3 -- context length=64

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
context_len = 512
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

# Filter rows based on date
filtered_df1 = df1[df1['Date'] <= cutoff_date]
# Step 2: Get the list of unique symbols
#symbols = filtered_df1['Symbol'].unique()
#symbols= ['.KS11','.FCHI']
symbols=['.AEX', '.AORD', '.BFX', '.BVSP', '.DJI','.FCHI']
#symbols= ['.FTSE', '.GDAXI', '.HSI', '.IBEX','.IXIC', '.KS11'] 
#symbols= ['.KSE', '.MXX', '.N225','.RUT','.SPX', '.SSEC','.SSMI','.STI','.STOXX50E']
# #################################################################################################################################################

def process_symbol(symbol):
    print(f"Processing symbol: {symbol}")
    
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
    total_data_points = len(df)
    
    symbol_cleaned = symbol.replace('.', '')
    csv_file_path = f'{symbol_cleaned}_var_data.csv'
    df.to_csv(csv_file_path, index=False)

    initial_train_percent = 0.4
    initial_val_percent = 0.1
    initial_test_percent = 0.1

    # Subsequent percentages
    subsequent_test_percent=0.1
    train_val_ratio = 0.8  # Train to validation ratio
    train_end = int(total_data_points * initial_train_percent)
    val_end = train_end + int(total_data_points * initial_val_percent)
    test_end = val_end + int(total_data_points * initial_test_percent)
    i=1
    boundaries=[0, train_end, val_end, test_end]
    while test_end < total_data_points:  # defin a loop that will extract the data and do the fine-turning
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
        CHECKPOINT_DIR = os.path.join('var_incre', symbol, f'iteration_{i}')
        i=i+1
        #CHECKPOINT_DIR = "path/to/output/dir" # the directory wwhere you want to save
        print("entering..")
        # Call the fine-tune function
        CHECKPOINT_DIR, combined_forecasts1, combined_forecasts2, combined_actuals = train_and_evaluate(
        checkpoint_pre, CHECKPOINT_DIR, train_batches, val_batches, test_batches,num_ts,batch_size)
        checkpoint_pre=CHECKPOINT_DIR # update the checkpoint directory for next iteration
        
        combined_forecasts1_all = pd.concat([combined_forecasts1_all, combined_forecasts1], ignore_index=True)
        combined_forecasts2_all = pd.concat([combined_forecasts2_all, combined_forecasts2], ignore_index=True)
        combined_actuals_all = pd.concat([combined_actuals_all, combined_actuals], ignore_index=True)        
        
        
        train_start = boundaries[2]  # Start of the previous test
        train_end = train_start + int((boundaries[3] - boundaries[2]) * train_val_ratio)

        val_start = train_end
        val_end = val_start + int((boundaries[3] - boundaries[2]) * (1 - train_val_ratio))

        test_start = val_end
        test_end = test_start + int(total_data_points * subsequent_test_percent)

        if test_end > total_data_points:
            test_end = total_data_points        
        boundaries= [train_start, train_end, val_end, test_end]
    
    
    forecasts_filename1 = os.path.join(output_folder, f'{symbol}_forecasts_mean.csv')
    forecasts_filename2 = os.path.join(output_folder, f'{symbol}_forecasts_median.csv')
    actuals_filename = os.path.join(output_folder, f'{symbol}_actuals.csv')

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
