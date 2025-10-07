#!/usr/bin/env python
# coding: utf-8

# In[ ]:


!pip install uv && uv pip install sovai['full'] --system > output.log 2>&1

# ## Breakout Prediction

# In[3]:


import time
import sys
import importlib # Using importlib for a slightly cleaner import mechanism

# --- List of Imports Extracted from Provided Snippets ---
# This list includes standard libraries, third-party libraries,
# and specific modules from the 'sovai' package identified in your code.
imports_to_test = [
    # Standard Library
    "typing",
    "re",
    "time",
    "functools",
    "datetime",
    "hashlib",
    "json",
    "logging",
    "collections",
    "io",
    "asyncio",
    "concurrent.futures",
    "pickle",
    "warnings", # Added from __init__.py
    "importlib.metadata", # Added from __init__.py

    # Third-Party Core Libraries
    "pandas",
    "numpy",
    "requests",
    "requests.exceptions", # Often loaded with requests, but test explicitly
    "boto3",
    "polars",
    "pyarrow", # Base library for pyarrow.parquet
    "pyarrow.parquet",
    "plotly", # Base plotly library
    "plotly.graph_objects",
    "plotly.io",
    "plotly.offline",
    "IPython",
    "IPython.display",

    # SovAI Specific Modules (based on imports in your code)
    "sovai", # For `from sovai import data` and the package itself
    "sovai.api_config",
    "sovai.basic_auth", # Added from __init__.py
    "sovai.token_auth", # Added from __init__.py
    "sovai.errors.sovai_errors",
    "sovai.utils.converter",
    "sovai.utils.stream",
    "sovai.utils.datetime_formats",
    "sovai.utils.client_side",
    "sovai.utils.client_side_s3",
    "sovai.utils.client_side_s3_part_high",
    "sovai.utils.verbose_utils",
    "sovai.utils.file_management", # Added from __init__.py
    "sovai.utils.plot", # From try/except block and explicit import
    "sovai.extensions.pandas_extensions", # From try/except block

    # SovAI Plotting Submodules (assuming relative paths resolved from sovai.plot)
    "sovai.plot.plots.bankruptcy.bankruptcy_plots",
    "sovai.plot.plots.breakout.breakout_plots",
    "sovai.plot.plots.accounting.accounting_plots",
    "sovai.plot.plots.ratios.ratios_plots",
    "sovai.plot.plots.institutional.institutional_plots",
    "sovai.plot.plots.news.news_plots",
    "sovai.plot.plots.corp_risk.corp_risk_plots",
    "sovai.plot.plots.insider.insider_plots",
    "sovai.plot.plots.allocation.allocation_plots",
    "sovai.plot.plots.earnings_surprise.earnings_surprise_plots",

    # SovAI Reporting Submodules (assuming relative paths resolved from sovai.plot)
    "sovai.plot.reports.bankruptcy.bankruptcy_monthly_top",
    "sovai.plot.reports.accounting.accounting_balance_sheet",
    "sovai.plot.reports.general.general_plots",
    "sovai.plot.reports.news.news_econometric_analysis",

    # SovAI Extensions Tools (assuming relative paths resolved from sovai.extensions)
    "sovai.extensions.tools.sec.sec_edgar_search",
    "sovai.extensions.tools.sec.sec_10_k_8_k_filings",
    "sovai.extensions.tools.sec.llm_code_generator",
    "sovai.extensions.tools.sec.graphs",

    # SovAI Compute Submodules (assuming relative paths resolved from sovai.plot)
    "sovai.plot.computations.functions",

    # Modules that might be lazy-loaded but are good to test independently if used elsewhere
    "sovai.get_data", # Underlying module for lazy-loaded data()
    "sovai.get_plots", # Underlying module for lazy-loaded plot()
    "sovai.get_reports", # Underlying module for lazy-loaded report()
    "sovai.get_compute", # Underlying module for lazy-loaded compute()
    "sovai.studies.nowcasting", # Underlying module for lazy-loaded nowcast()
    "sovai.get_tools", # Underlying module for lazy-loaded sec_search(), etc.
]

# Remove duplicates just in case
imports_to_test = sorted(list(set(imports_to_test)))

# --- Import Timing Logic ---
results = []
failed_imports = []
other_errors = []

# Add current directory to sys.path if needed for local sovai development
# sys.path.insert(0, ".")

print("Starting import timing test...")
print(f"Testing {len(imports_to_test)} unique modules.")

# --- Pre-run check for sovai ---
# Let's try importing the base package first to catch immediate issues
try:
    importlib.import_module("sovai")
    print("\nBase 'sovai' package imported successfully. Proceeding with detailed timing...\n")
except ImportError as e:
    print(f"\nCRITICAL ERROR: Could not import the base 'sovai' package: {e}", file=sys.stderr)
    print("Please ensure 'sovai' is installed correctly in this environment before running timing tests.", file=sys.stderr)
    sys.exit(1) # Exit if the base package can't be found
except Exception as e:
    print(f"\nCRITICAL ERROR: An unexpected error occurred while importing the base 'sovai' package: {e}", file=sys.stderr)
    sys.exit(1)
# --- End Pre-run check ---


for module_name in imports_to_test:
    start_time = time.time()
    try:
        # Use importlib.import_module for robust importing
        importlib.import_module(module_name)
        end_time = time.time()
        duration = end_time - start_time
        results.append((module_name, duration))
        # Optionally reduce print frequency for long lists
        # if duration > 0.01 or module_name.startswith("sovai"):
        print(f"  Successfully imported {module_name} in {duration:.6f}s")
    except ImportError:
        failed_imports.append(module_name)
        print(f"  ERROR: Could not import {module_name} (ImportError)", file=sys.stderr)
    except Exception as e:
        other_errors.append((module_name, e))
        print(f"  ERROR: Exception during import of {module_name}: {e}", file=sys.stderr)

print("\nImport timing test finished.")

# Sort successful results by duration (longest first)
results.sort(key=lambda item: item[1], reverse=True)

print("\n--- Import Time Results (Successful Imports, seconds) ---")
if results:
    for module_name, duration in results:
        print(f"{module_name}: {duration:.6f}")
else:
    print("No modules imported successfully.")

if failed_imports:
    print("\n--- Failed Imports (ImportError) ---")
    for module_name in failed_imports:
        print(f"{module_name}")

if other_errors:
    print("\n--- Failed Imports (Other Errors) ---")
    for module_name, error in other_errors:
        print(f"{module_name}: {error}")

print("\n--- End of Report ---")

# In[1]:


import time
import sys
import importlib # Using importlib for a slightly cleaner import mechanism

# --- List of Imports Extracted from Provided Snippets ---
# This list includes standard libraries, third-party libraries,
# and specific modules from the 'sovai' package identified in your code.
imports_to_test = [
    # Standard Library
    "typing",
    "re",
    "time",
    "functools",
    "datetime",
    "hashlib",
    "json",
    "logging",
    "collections",
    "io",
    "asyncio",
    "concurrent.futures",
    "pickle",

    # Third-Party Core Libraries
    "pandas",
    "numpy",
    "requests",
    "requests.exceptions", # Often loaded with requests, but test explicitly
    "boto3",
    "polars",
    "pyarrow", # Base library for pyarrow.parquet
    "pyarrow.parquet",
    "plotly", # Base plotly library
    "plotly.graph_objects",
    "plotly.io",
    "plotly.offline",
    "IPython",
    "IPython.display",

    # SovAI Specific Modules (based on imports in your code)
    "sovai", # For `from sovai import data`
    "sovai.api_config",
    "sovai.errors.sovai_errors",
    "sovai.utils.converter",
    "sovai.utils.stream",
    "sovai.utils.datetime_formats",
    "sovai.utils.client_side",
    "sovai.utils.client_side_s3",
    "sovai.utils.client_side_s3_part_high",
    "sovai.utils.verbose_utils",
    "sovai.extensions.pandas_extensions", # From try/except block
    "sovai.utils.plot", # From try/except block and explicit import

    # SovAI Plotting Submodules (assuming relative paths resolved from sovai.plot)
    "sovai.plot.plots.bankruptcy.bankruptcy_plots",
    "sovai.plot.plots.breakout.breakout_plots",
    "sovai.plot.plots.accounting.accounting_plots",
    "sovai.plot.plots.ratios.ratios_plots",
    "sovai.plot.plots.institutional.institutional_plots",
    "sovai.plot.plots.news.news_plots",
    "sovai.plot.plots.corp_risk.corp_risk_plots",
    "sovai.plot.plots.insider.insider_plots",
    "sovai.plot.plots.allocation.allocation_plots",
    "sovai.plot.plots.earnings_surprise.earnings_surprise_plots",

    # SovAI Reporting Submodules (assuming relative paths resolved from sovai.plot)
    "sovai.plot.reports.bankruptcy.bankruptcy_monthly_top",
    "sovai.plot.reports.accounting.accounting_balance_sheet",
    "sovai.plot.reports.general.general_plots",
    "sovai.plot.reports.news.news_econometric_analysis",

    # SovAI Extensions Tools (assuming relative paths resolved from sovai.extensions)
    "sovai.extensions.tools.sec.sec_edgar_search",
    "sovai.extensions.tools.sec.sec_10_k_8_k_filings",
    "sovai.extensions.tools.sec.llm_code_generator",
    "sovai.extensions.tools.sec.graphs",

    # SovAI Compute Submodules (assuming relative paths resolved from sovai.plot)
    "sovai.plot.computations.functions",
]

# --- Import Timing Logic ---
results = []
failed_imports = []
other_errors = []

# Add current directory to sys.path if needed for local sovai development
# sys.path.insert(0, ".")

print("Starting import timing test...")

for module_name in imports_to_test:
    start_time = time.time()
    try:
        # Use importlib.import_module for robust importing
        importlib.import_module(module_name)
        end_time = time.time()
        duration = end_time - start_time
        results.append((module_name, duration))
        print(f"  Successfully imported {module_name} in {duration:.6f}s")
    except ImportError:
        failed_imports.append(module_name)
        print(f"  ERROR: Could not import {module_name} (ImportError)", file=sys.stderr)
    except Exception as e:
        other_errors.append((module_name, e))
        print(f"  ERROR: Exception during import of {module_name}: {e}", file=sys.stderr)

print("\nImport timing test finished.")

# Sort successful results by duration (longest first)
results.sort(key=lambda item: item[1], reverse=True)

print("\n--- Import Time Results (Successful Imports, seconds) ---")
if results:
    for module_name, duration in results:
        print(f"{module_name}: {duration:.6f}")
else:
    print("No modules imported successfully.")

if failed_imports:
    print("\n--- Failed Imports (ImportError) ---")
    for module_name in failed_imports:
        print(f"{module_name}")

if other_errors:
    print("\n--- Failed Imports (Other Errors) ---")
    for module_name, error in other_errors:
        print(f"{module_name}: {error}")

print("\n--- End of Report ---")

# In[1]:


import time
import sys

imports_to_test = [
    "pandas",
    "numpy",
    "plotly.express",
    "numpy.linalg",
    "sovai.extensions.shapley_global_importance",
    "sovai.extensions.shapley_importance",
    "sovai.extensions.feature_neutralizer",
    "sovai.extensions.fractional_differencing",
    "sovai.extensions.anomalies",
    "sovai.extensions.pairwise",
    "sovai.extensions.clustering",
    "sovai.extensions.nowcasting",
    "sovai.extensions.change_point_generator",
    "sovai.extensions.regime_change",
    "sovai.extensions.regime_change_pca",
    "sovai.extensions.time_decomposition",
    "sovai.extensions.feature_extraction",
    "sovai.extensions.dimensionality_reduction",
    "sovai.extensions.feature_importance",
    "sovai.extensions.weight_optimization",
    "sovai.extensions.signal_evaluation",
    "sovai.extensions.technical_indicators",
    "warnings",
    "dateutil",
    "typing",
    "sovai.extensions.ask_df_llm",
    "sovai.extensions.filter_df",
    "functools",
    "re",
    "polars",
]

results = []

# Add current directory to sys.path to find local modules
sys.path.insert(0, ".")

for module_name in imports_to_test:
    start_time = time.time()
    try:
        # Use __import__ with level=0 for absolute imports
        __import__(module_name, level=0)
        end_time = time.time()
        duration = end_time - start_time
        results.append((module_name, duration))
    except ImportError:
        results.append((module_name, -1)) # Indicate import failed
    except Exception as e:
        results.append((module_name, -2)) # Indicate other error
        print(f"Error importing {module_name}: {e}", file=sys.stderr)


# Sort results by duration (longest first), putting failed imports at the end
results.sort(key=lambda item: item[1] if item[1] >= 0 else float('inf'), reverse=True)

print("--- Import Time Results (seconds) ---")
for module_name, duration in results:
    if duration >= 0:
        print(f"{module_name}: {duration:.6f}")
    elif duration == -1:
        print(f"{module_name}: Import Failed (Module not found)")
    else:
        print(f"{module_name}: Import Failed (Other error)")


# You can run the following commands to retrieve data using `sov.data`:
# 
# To fetch the **latest data** for a specific query:
# 
# ```python
# sov.data("query")
# ```
# 
# To fetch the **full historical data** for a specific query:
# 
# ```python
# sov.data("query", full_history=True)
# ```
# 
# To fetch the **full data** multiple **tickers** or identifiers like **cusip** and **openfigi**:
# 
# ```python
# sov.data("query", tickers=["9033434", "IB94343", "43432", "AAPL"])
# ```
# 
# To filter **any dataframe** just write some queries:
# 
# ```python
# df_accounting.filter(["cash_short_term > 10m","start with ticker A","negative profits" ])
# ```
# 

# In[ ]:


import sovai as sov

sov.token_auth(token="visit https://sov.ai/profile for your token")

# ### Grab historical predictions - Large File (2 mins)

# In[ ]:


df_breakout = sov.data("breakout"); df_breakout

# Let's look at the latest data, although the model is explicity trained on the long-side, the short-side could also contain some positive signal.

# In[ ]:


df_breakout.sort_values("slope")

# It is sometimes advised to remove the top and bottom 1% of the data as they could be related to noise or firms with no liquidity. 

# In[ ]:


df_breakout = df_breakout[(df_breakout["prediction"] > df_breakout["prediction"].quantile(0.01)) & (df_breakout["prediction"] < df_breakout["prediction"].quantile(0.99))]

# In[ ]:


df_breakout

# Let's plot a simple prediction over time.

# Let's also add some confidence intervals:
# (1) Change of slope is a **strong** indicator, (2) change of slope + above/below 50% is a **very strong** indicator. 

# In[ ]:


df_msft = sov.data("breakout", start_date="2025-01-01")

# In[ ]:


df_msft = sov.data("breakout", tickers=["MSFT"])

# In[ ]:


import sovai as sov

sov.token_auth(token="visit https://sov.ai/profile for your token")
sov.plot("breakout", chart_type="predictions", df=df_msft)

# In[ ]:


sov.plot("breakout", chart_type="accuracy", df=df_msft)
