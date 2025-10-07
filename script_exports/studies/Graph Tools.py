#!/usr/bin/env python
# coding: utf-8

# In[ ]:


!pip install uv && uv pip install sovai['full'] --system > output.log 2>&1

# In[1]:


# !pip install edgartools==2.27.5

# In[3]:


import sovai as sov

sov.token_auth(token="visit https://sov.ai/profile for your token")

sov.data("sec/10k",tickers=["NVDA"])

# In[5]:


# data = sov.data("sec/10k",columns=["ticker","date","full_text"])

# In[6]:


data["ticker"].value_counts()

# In[68]:


df_codes = pd.read_parquet("data/codes.parq")

# In[84]:


df_codes.query("ticker =='NVDA'")

# In[115]:


sov.data("sec/10k",tickers=["NVDA"])

# In[104]:


from sovai.utils.client_side_s3 import load_frame_s3

# In[109]:


full_frame = load_frame_s3("sec/10k", tickers=["TSLA"])

# In[110]:


full_frame

# In[97]:


full_frame[full_frame["cik"]==8192]

# In[96]:


full_frame["cik"].value_counts()

# In[79]:


%%timeit
map_identifier(["AMZN","NVDA","MSFT"])

# In[65]:


import pyarrow.dataset as ds
from pyarrow.fs import S3FileSystem
import pyarrow as pa
import pandas as pd
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, as_completed
from sovai.tools.authentication import authentication


@lru_cache(maxsize=2)
def get_cached_s3_filesystem(storage_provider):
    return authentication.get_s3_filesystem_pickle(storage_provider, verbose=True)

def process_partition(storage_provider, base_path, identifier_column, identifier=None, columns=None, filters=None):
    s3 = get_cached_s3_filesystem(storage_provider)
    
    if identifier:
        base_path += f"/{identifier_column}={identifier}"
    
    dataset = ds.dataset(base_path, filesystem=s3, format='parquet')
    
    if filters:
        operator_map = {
            '==': lambda a, b: a == b,
            '!=': lambda a, b: a != b,
            '<': lambda a, b: a < b,
            '<=': lambda a, b: a <= b,
            '>': lambda a, b: a > b,
            '>=': lambda a, b: a >= b,
            'in': lambda a, b: a.isin(b),
            'not in': lambda a, b: ~a.isin(b),
            'like': lambda a, b: a.match_substring(b),
        }
        schema = dataset.schema
        filter_expr = None
        for col, op, val in filters:
            if op not in operator_map:
                raise ValueError(f"Unsupported operator '{op}' in filters.")
            field = ds.field(col)
            field_type = schema.field(col).type
            if pa.types.is_timestamp(field_type) or pa.types.is_date(field_type):
                val = pd.to_datetime(val)
            elif pa.types.is_integer(field_type):
                val = int(val)
            elif pa.types.is_floating(field_type):
                val = float(val)
            condition = operator_map[op](field, val)
            filter_expr = condition if filter_expr is None else filter_expr & condition
    else:
        filter_expr = None
    
    table = dataset.to_table(columns=columns, filter=filter_expr, use_threads=True)
    df = table.to_pandas(use_threads=True)
    
    if identifier:
        df[identifier_column] = identifier
    
    return df

def client_side_s3_frame(config, identifiers, columns, start_date, end_date):
    storage_provider = "digitalocean"  # Default to DigitalOcean
    base_path = config["storage_provider"][storage_provider]
    identifier_column = config.get("identifier_column", "cik")
    
    filters = []
    if start_date:
        filters.append(('date', '>=', start_date))
    if end_date:
        filters.append(('date', '<=', end_date))

    if not identifiers:
        # Load the entire database
        return process_partition(storage_provider, base_path, identifier_column, columns=columns, filters=filters)
    
    if isinstance(identifiers, str) or (isinstance(identifiers, list) and len(identifiers) == 1):
        # Single identifier
        identifier = identifiers if isinstance(identifiers, str) else identifiers[0]
        return process_partition(storage_provider, base_path, identifier_column, identifier=identifier, columns=columns, filters=filters)
    
    # Multiple identifiers
    with ThreadPoolExecutor() as executor:
        futures = []
        for identifier in identifiers:
            futures.append(executor.submit(process_partition, storage_provider, base_path, identifier_column, identifier, columns, filters))
        
        results = []
        for future in as_completed(futures):
            results.append(future.result())
    
    df = pd.concat(results, ignore_index=True)
    # if 'filing_date' in df.columns:
    #     df['date'] = pd.to_datetime(df['filing_date'])
    # elif 'date' not in df.columns:
    #     raise ValueError("Neither 'filing_date' nor 'date' column found in the data")
    
    # df.set_index([identifier_column, 'date'], inplace=True)
    # df.sort_index(inplace=True)
    
    return df


def load_frame_s3(endpoint, identifiers, columns, start_date, end_date):

    endpoint_config = {
        "sec/10k": {
            "storage_provider": {
                "digitalocean": "sovai/sovai-filings/tenk",
                "wasabi": "sovai-filings/tenk"
            },
            "identifier_column": "cik",
            "partitioned": True,
        },
        "future/enpdoint": {
            "storage_provider": {
                "digitalocean": "sovai/sovai-filings/tenk",
                "wasabi": "sovai-filings/tenk"
            },
            "identifier_column": "cik",
            "partitioned": True,
        }
    }
    
    if endpoint not in endpoint_config:
        raise ValueError(f"Invalid endpoint: {endpoint}")
    
    config = endpoint_config[endpoint]
    
    if endpoint == "sec/10k":
        df_frame = client_side_s3_frame(
            config,
            identifiers,
            columns,
            start_date,
            end_date
        )
    else:
        # Implement logic for other endpoints if needed
        raise NotImplementedError(f"Endpoint {endpoint} is not yet implemented")
    
    if HAS_CUSTOM_DATAFRAME:
        return CustomDataFrame(df_frame)
    else:
        return df_frame  # Returns a regular pandas DataFrame if CustomDataFrame is not available



# In[66]:


result

# In[64]:


%%timeit
result = client_side_s3_frame(config, identifiers, columns, start_date, end_date)

# In[45]:


# client_side_s3.py
import pandas as pd
import pyarrow.dataset as ds
import pyarrow as pa
from pyarrow.fs import S3FileSystem
from functools import lru_cache
import sovai as sov
from sovai.tools.authentication import authentication
from concurrent.futures import ThreadPoolExecutor, as_completed

sov.token_auth(token="visit https://sov.ai/profile for your token")

@lru_cache(maxsize=2)
def get_cached_s3_filesystem(storage_provider):
    return authentication.get_s3_filesystem_pickle(storage_provider, verbose=True)

def process_s3_partition(storage_provider, base_path, identifier_column, identifier_value, columns=None, filters=None):
    s3 = get_cached_s3_filesystem(storage_provider)
    
    dataset = ds.dataset(base_path, filesystem=s3, format='parquet')
    
    if filters is None:
        filters = []
    filters.append((identifier_column, '==', identifier_value))
    
    operator_map = {
        '==': lambda a, b: a == b,
        '!=': lambda a, b: a != b,
        '<': lambda a, b: a < b,
        '<=': lambda a, b: a <= b,
        '>': lambda a, b: a > b,
        '>=': lambda a, b: a >= b,
        'in': lambda a, b: a.isin(b),
        'not in': lambda a, b: ~a.isin(b),
        'like': lambda a, b: a.match_substring(b),
    }
    
    schema = dataset.schema
    filter_expr = None
    for col, op, val in filters:
        if op not in operator_map:
            raise ValueError(f"Unsupported operator '{op}' in filters.")
        field = ds.field(col)
        field_type = schema.field(col).type
        if pa.types.is_timestamp(field_type) or pa.types.is_date(field_type):
            val = pd.to_datetime(val)
        elif pa.types.is_integer(field_type):
            val = int(val)
        elif pa.types.is_floating(field_type):
            val = float(val)
        condition = operator_map[op](field, val)
        filter_expr = condition if filter_expr is None else filter_expr & condition
    
    table = dataset.to_table(columns=columns, filter=filter_expr, use_threads=True)
    return table.to_pandas(use_threads=True)

def process_identifier(args):
    storage_provider, base_path, identifier_column, identifier, columns, filters = args
    df = process_s3_partition(storage_provider, base_path, identifier_column, identifier, columns, filters)
    if 'filing_date' in df.columns:
        df['date'] = pd.to_datetime(df['filing_date'])
    elif 'date' not in df.columns:
        raise ValueError(f"Neither 'filing_date' nor 'date' column found in the data for {identifier}")
    return df

def client_side_s3_frame(config, identifiers, columns, start_date, end_date):
    storage_provider = "digitalocean"  # Default to DigitalOcean
    base_path = config["storage_provider"][storage_provider]
    identifier_column = config.get("identifier_column", "cik")
    
    filters = []
    if start_date:
        filters.append(('filing_date', '>=', start_date))
    if end_date:
        filters.append(('filing_date', '<=', end_date))
    
    with ThreadPoolExecutor() as executor:
        futures = []
        for identifier in identifiers:
            args = (storage_provider, base_path, identifier_column, identifier, columns, filters)
            futures.append(executor.submit(process_identifier, args))
        
        results = []
        for future in as_completed(futures):
            results.append(future.result())
    
    df = pd.concat(results, ignore_index=True)
    df.set_index([identifier_column, 'date'], inplace=True)
    df.sort_index(inplace=True)
    
    return df


config = {
    "s3": True,
    "storage_provider": {
        "digitalocean": "sovai/sovai-filings/tenk",
        "wasabi": "sovai-filings/tenk"
    },
    "identifier_column": "cik"
}
identifiers = ["1018724"]  # Example CIKs
columns = None
start_date = "2020-01-01"
end_date = None

result = client_side_s3_frame(config, identifiers, columns, start_date, end_date)


# In[46]:


result = client_side_s3_frame(config, identifiers, columns, start_date, end_date)

# In[36]:


# client_side_s3.py
import pandas as pd
import pyarrow.dataset as ds
import pyarrow as pa
from pyarrow.fs import S3FileSystem
from functools import lru_cache
import sovai as sov
from sovai.tools.authentication import authentication

sov.token_auth(token="visit https://sov.ai/profile for your token")

@lru_cache(maxsize=2)
def get_cached_s3_filesystem(storage_provider):
    return authentication.get_s3_filesystem_pickle(storage_provider, verbose=True)

def process_s3_partition(storage_provider, base_path, identifier_column=None, identifier_values=None, columns=None, filters=None):
    s3 = get_cached_s3_filesystem(storage_provider)
    
    dataset = ds.dataset(base_path, filesystem=s3, format='parquet')
    
    if identifier_column and identifier_values:
        if filters is None:
            filters = []
        filters.append((identifier_column, 'in', identifier_values))
    
    if filters:
        operator_map = {
            '==': lambda a, b: a == b,
            '!=': lambda a, b: a != b,
            '<': lambda a, b: a < b,
            '<=': lambda a, b: a <= b,
            '>': lambda a, b: a > b,
            '>=': lambda a, b: a >= b,
            'in': lambda a, b: a.isin(b),
            'not in': lambda a, b: ~a.isin(b),
            'like': lambda a, b: a.match_substring(b),
        }
        schema = dataset.schema
        filter_expr = None
        for col, op, val in filters:
            if op not in operator_map:
                raise ValueError(f"Unsupported operator '{op}' in filters.")
            field = ds.field(col)
            field_type = schema.field(col).type
            if op not in ['in', 'not in']:
                if pa.types.is_timestamp(field_type) or pa.types.is_date(field_type):
                    val = pd.to_datetime(val)
                elif pa.types.is_integer(field_type):
                    val = int(val)
                elif pa.types.is_floating(field_type):
                    val = float(val)
            condition = operator_map[op](field, val)
            filter_expr = condition if filter_expr is None else filter_expr & condition
    else:
        filter_expr = None
    
    table = dataset.to_table(columns=columns, filter=filter_expr, use_threads=True)
    return table.to_pandas(use_threads=True)

def client_side_s3_frame(config, identifiers, columns, start_date, end_date):
    storage_provider = "digitalocean"  # Default to DigitalOcean
    base_path = config["storage_provider"][storage_provider]
    identifier_column = config.get("identifier_column", "cik")
    
    filters = []
    if start_date:
        filters.append(('filing_date', '>=', start_date))
    if end_date:
        filters.append(('filing_date', '<=', end_date))
    
    df = process_s3_partition(
        storage_provider, 
        base_path, 
        identifier_column=identifier_column, 
        identifier_values=identifiers, 
        columns=columns, 
        filters=filters
    )
    
    # if 'filing_date' in df.columns:
    #     df['date'] = pd.to_datetime(df['filing_date'])  # Ensure 'date' column exists
    # elif 'date' not in df.columns:
    #     raise ValueError("Neither 'filing_date' nor 'date' column found in the data")
    
    # df.set_index([identifier_column, 'date'], inplace=True)
    # df.sort_index(inplace=True)
    
    return df

# Example usage
config = {
    "s3": True,
    "storage_provider": {
        "digitalocean": "sovai/sovai-filings/tenk",
        "wasabi": "sovai-filings/tenk"
    },
    "identifier_column": "cik"
}
identifiers = ["1018724", "1045810"]  # Example CIKs
columns = None
start_date = "2020-01-01"
end_date = None


# In[39]:


%%timeit
result = client_side_s3_frame(config, identifiers, columns, start_date, end_date)

# In[38]:


result.sort_index()

# In[3]:


s3fs_object = authentication.get_s3_filesystem_pickle("digitalocean",verbose=True)

# In[47]:


import pyarrow.dataset as ds
from pyarrow.fs import S3FileSystem
import pyarrow as pa
import pandas as pd
from functools import lru_cache

@lru_cache(maxsize=2)  # Cache the last 2 S3FileSystem objects (one for each provider)
def get_cached_s3_filesystem(storage_provider):
    return authentication.get_s3_filesystem_pickle(storage_provider, verbose=True)

def process_partition(storage_provider, cik=None, columns=None, filters=None):
    s3 = get_cached_s3_filesystem(storage_provider)
    
    if storage_provider.lower() == 'digitalocean':
        base_path = "sovai/sovai-filings/tenk"
    else:  # Wasabi
        base_path = "sovai-filings/tenk"
    
    if cik:
        base_path += f"/cik={cik}"
    
    dataset = ds.dataset(base_path, filesystem=s3, format='parquet')
    
    if filters:
        operator_map = {
            '==': lambda a, b: a == b,
            '!=': lambda a, b: a != b,
            '<': lambda a, b: a < b,
            '<=': lambda a, b: a <= b,
            '>': lambda a, b: a > b,
            '>=': lambda a, b: a >= b,
            'in': lambda a, b: a.isin(b),
            'not in': lambda a, b: ~a.isin(b),
            'like': lambda a, b: a.match_substring(b),
        }
        schema = dataset.schema
        filter_expr = None
        for col, op, val in filters:
            if op not in operator_map:
                raise ValueError(f"Unsupported operator '{op}' in filters.")
            field = ds.field(col)
            field_type = schema.field(col).type
            # Convert the value to the appropriate type based on the column type
            if pa.types.is_timestamp(field_type) or pa.types.is_date(field_type):
                val = pd.to_datetime(val)
            elif pa.types.is_integer(field_type):
                val = int(val)
            elif pa.types.is_floating(field_type):
                val = float(val)
            # For string and other types, keep the value as is
            condition = operator_map[op](field, val)
            filter_expr = condition if filter_expr is None else filter_expr & condition
    else:
        filter_expr = None
    
    table = dataset.to_table(columns=columns, filter=filter_expr, use_threads=True)
    return table.to_pandas(use_threads=True)

# Example usage
storage_provider = 'wasabi'  # or 'wasabi'
cik = 1018724
columns = None
filters = [('filing_date', '>=', '2020-01-01')]
final_df = process_partition(storage_provider, cik, columns, filters)

# In[48]:


%%timeit
final_df = process_partition(storage_provider, cik, columns, filters)

# In[16]:


# Example usage
storage_provider = 'digitalocean'
cik = 1018724
columns = None
filters = [('filing_date', '>=', '2020-01-01')]


# In[18]:


%%timeit
final_df = process_partition(storage_provider, cik, columns, filters)

# In[20]:


process_partition('wasabi', cik, columns, filters)

# In[19]:


%%timeit
final_df = process_partition('wasabi', cik, columns, filters)

# In[17]:


final_df

# In[9]:


import pyarrow.dataset as ds
from pyarrow.fs import S3FileSystem
import pyarrow as pa
import pandas as pd


storage_provider = 'digitalocean'
cik = '1018724'
# cik= None
columns = None
# filters = [('filing_date', '>=', '2020-01-01')]


s3 = s3fs_object

base_path = f"sovai/{'sovai-filings/' if storage_provider.lower() == 'digitalocean' else ''}tenk"
if cik:
    base_path += f"/cik={cik}"

dataset = ds.dataset(base_path, filesystem=s3, format='parquet')

if filters:
    operator_map = {
        '==': lambda a, b: a == b,
        '!=': lambda a, b: a != b,
        '<': lambda a, b: a < b,
        '<=': lambda a, b: a <= b,
        '>': lambda a, b: a > b,
        '>=': lambda a, b: a >= b,
        'in': lambda a, b: a.isin(b),
        'not in': lambda a, b: ~a.isin(b),
        'like': lambda a, b: a.match_substring(b),
    }

    schema = dataset.schema

    filter_expr = None
    for col, op, val in filters:
        if op not in operator_map:
            raise ValueError(f"Unsupported operator '{op}' in filters.")

        field = ds.field(col)
        field_type = schema.field(col).type

        # Convert the value to the appropriate type based on the column type
        if pa.types.is_timestamp(field_type) or pa.types.is_date(field_type):
            val = pd.to_datetime(val)
        elif pa.types.is_integer(field_type):
            val = int(val)
        elif pa.types.is_floating(field_type):
            val = float(val)
        # For string and other types, keep the value as is

        condition = operator_map[op](field, val)
        filter_expr = condition if filter_expr is None else filter_expr & condition
else:
    filter_expr = None

table = dataset.to_table(columns=columns, filter=filter_expr, use_threads=True).to_pandas()




# In[10]:


table

# In[ ]:


import pyarrow.dataset as ds
from pyarrow.fs import S3FileSystem
import pyarrow as pa
import pandas as pd

def get_s3_filesystem(storage_provider):
    credentials = {
        'wasabi': {
            'endpoint': 'https://s3.wasabisys.com',
                'access_key': 'KI07G4DK9XP5_fake_id_ECd0NH1VU',
                'secret_key': 'kXLnJylZJ4ai_fake_id_DuA5BtbEq9N53dC1eFUmyVqsnCzRn'
        },
        'digitalocean': {
            'endpoint': 'https://nyc3.digitaloceanspaces.com',
            'access_key': 'DO00U3NVdRFE2P8_fake_id_3JPDU2',
            'secret_key': 'AuYt4sW4F02yR+ERdht_fake_id_g0n9dGX2Qk5xaLtZTUY+V+yQo'
        }
    }

    cred = credentials.get(storage_provider.lower())
    if not cred:
        raise ValueError("Invalid storage provider. Choose 'wasabi' or 'digitalocean'.")

    return S3FileSystem(access_key=cred['access_key'],
                        secret_key=cred['secret_key'],
                        endpoint_override=cred['endpoint'])

def process_partition(storage_provider, cik=None, columns=None, filters=None):
    s3 = get_s3_filesystem(storage_provider)
    base_path = f"sovai/{'sovai-filings/' if storage_provider.lower() == 'digitalocean' else ''}tenk"
    if cik:
        base_path += f"/cik={cik}"

    dataset = ds.dataset(base_path, filesystem=s3, format='parquet')

    if filters:
        operator_map = {
            '==': lambda a, b: a == b,
            '!=': lambda a, b: a != b,
            '<': lambda a, b: a < b,
            '<=': lambda a, b: a <= b,
            '>': lambda a, b: a > b,
            '>=': lambda a, b: a >= b,
            'in': lambda a, b: a.isin(b),
            'not in': lambda a, b: ~a.isin(b),
            'like': lambda a, b: a.match_substring(b),
        }

        schema = dataset.schema

        filter_expr = None
        for col, op, val in filters:
            if op not in operator_map:
                raise ValueError(f"Unsupported operator '{op}' in filters.")

            field = ds.field(col)
            field_type = schema.field(col).type

            # Convert the value to the appropriate type based on the column type
            if pa.types.is_timestamp(field_type) or pa.types.is_date(field_type):
                val = pd.to_datetime(val)
            elif pa.types.is_integer(field_type):
                val = int(val)
            elif pa.types.is_floating(field_type):
                val = float(val)
            # For string and other types, keep the value as is

            condition = operator_map[op](field, val)
            filter_expr = condition if filter_expr is None else filter_expr & condition
    else:
        filter_expr = None

    table = dataset.to_table(columns=columns, filter=filter_expr, use_threads=True)
    return table.to_pandas(use_threads=True)

# Example usage
storage_provider = 'digitalocean'
cik = '1018724'
cik= None
columns = None
filters = [('filing_date', '>=', '2020-01-01')]

final_df = process_partition(storage_provider, cik, columns, filters)

