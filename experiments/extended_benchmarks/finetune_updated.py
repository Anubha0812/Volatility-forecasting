## conda activate tfm2
import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['JAX_PMAP_USE_TENSORSTORE'] = 'false'


import numpy as np
import pandas as pd






import concurrent.futures


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
context_len=64
pred_len = 1



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
symbols=['.AEX']#, '.AORD', '.BFX', '.BVSP', '.DJI','.FCHI','.FTSE', '.GDAXI', '.HSI', 
#        '.IBEX','.IXIC', '.KS11','.KSE', '.MXX', '.N225','.RUT','.SPX', '.SSEC'
#symbols=['.SSMI','.STI','.STOXX50E']
#symbols= ['.FTSE', '.GDAXI', '.HSI', '.IBEX','.IXIC', '.KS11'] 
#symbols= ['.KSE', '.MXX', '.N225','.RUT','.SPX', '.SSEC','.SSMI','.STI','.STOXX50E'][]
# #################################################################################################################################################

def process_symbol(symbol):
    print(f"Processing symbol: {symbol}")
    
    
    stop=0
    filtered_df = filtered_df1[filtered_df1['Symbol'] == symbol]
    selected_columns = ['Date', 'rv5_ss']
    df = filtered_df[selected_columns]
    df['rv5_ss'] = np.log(df['rv5_ss'])
    total_data_points = 1000# len(df)
    
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
        print(boundaries)
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
        print(boundaries)
    
    
    
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

        
        