#!/usr/bin/env python
# coding: utf-8

# In[ ]:


!pip install uv && uv pip install sovai['full'] --system > output.log 2>&1

# You probably have to force this to run once a week or something and push the file.

# In[1]:


import pandas as pd
tick = pd.read_parquet("https://storage.googleapis.com/sovai-public/accounting/tickers_transformed.parq")

# In[2]:


tick

# In[4]:


import pandas as pd

import pandas as pd
import os
from datetime import datetime, timedelta

import os
import pandas as pd
from datetime import datetime, timedelta

def classify_etf_active(category):
    if category is None:
        return "Active"  # or any other default value you see fit
    return "Passive" if "ETF" in category else "Active"

def classify_foreign_domestic(category):
    if category is None:
        return "Domestic"  # or any other default value you see fit
    return "Foreign" if "ADR" in category or "Canadian" in category else "Domestic"

def classify_stock_type(category):
    if category is None:
        return None  # or any other default value you see fit
    if "Common Stock" in category:
        return "Common Stock"
    elif "CEF" in category:
        return "Closed-End Fund"
    elif "ETF" in category:
        return "Exchange-Traded Fund"
    elif "ETN" in category:
        return "Exchange-Traded Note"
    elif "UNIT" in category:
        return None

def rename_tickers(tickers_meta):
    rename_dict = {
        '3 - Small': 'small',
        '2 - Micro': 'micro',
        '4 - Mid': 'medium',
        '5 - Large': 'large',
        '1 - Nano': 'nano',
        '6 - Mega': 'mega'
    }
    tickers_meta['scalemarketcap'] = tickers_meta['scalemarketcap'].replace(rename_dict)
    return tickers_meta

def save_or_update_tickers(output_directory, output_filename, download_url):
    # Create the directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)

    # Full path for the output file
    output_path = os.path.join(output_directory, output_filename)

    # Check if the file exists
    file_exists = os.path.exists(output_path)
    file_is_recent = False

    # If file exists, check if it's older than 7 days
    if file_exists:
        file_age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(output_path))
        file_is_recent = file_age < timedelta(days=7)

    # If the file does not exist or is older than 7 days, download it
    if not file_exists or not file_is_recent:
        print("Downloading and saving new file.")
        tickers_meta = pd.read_parquet(download_url)
        
        # Classify tickers
        tickers_meta["active"] = tickers_meta["category"].apply(classify_etf_active)
        tickers_meta["foreign"] = tickers_meta["category"].apply(classify_foreign_domestic)
        tickers_meta["class"] = tickers_meta["category"].apply(classify_stock_type)
        
        # Remove rows where 'ticker' is None
        tickers_meta = tickers_meta.dropna(subset=['ticker']).drop_duplicates(["ticker"]).dropna(subset=["ticker"])

        # Rename tickers
        tickers_meta = rename_tickers(tickers_meta)

        print(tickers_meta[["scalemarketcap"]].head())
        
        tickers_meta.to_parquet(output_path)
        print(output_path)
    else:
        print(output_path)
        print("File is up-to-date, no need to download.")

# Example usage
# save_or_update_tickers('output_directory', 'output_filename.parquet', 'download_url', 'service_account_info')


# Example usage
output_directory = '../../sovai/assets/'
output_filename = 'tickers.parq'
download_url = "gs://sovai-accounting/dataframes/tickers.parq"

save_or_update_tickers(output_directory, output_filename, download_url)


# In[5]:


import os 
output_directory = 'data'
output_filename = "features_mapping.parq"
output_path = os.path.join(output_directory, output_filename)


# In[6]:


output_path

# In[7]:


import pandas as pd
featies = pd.read_csv("https://docs.google.com/spreadsheets/d/e/2PACX-1vTjJfxQIOXMbW4iExhqjb3T8iygPKmfSNoAvrhu7v8L2txFUCoASZq9iW7ITbpnHaHk-5I3qHzrnX8M/pub?gid=0&single=true&output=csv")
featies.to_parquet(output_path)

