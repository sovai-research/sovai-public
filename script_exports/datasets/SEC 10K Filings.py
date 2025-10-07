#!/usr/bin/env python
# coding: utf-8

# In[ ]:


!pip install uv && uv pip install sovai['full'] --system > output.log 2>&1

# ## SEC 10-K filings

# You can run the following commands to retrieve data (`df`) using `sov.data`:
# 
# To fetch the **latest data** for a specific query:
# 
# ```python
# df = sov.data("query")
# ```
# 
# To fetch the **full historical data** for a specific query:
# 
# ```python
# df = sov.data("query", full_history=True)
# ```
# 
# To fetch the **full data** multiple **tickers** or identifiers like **cusip** and **openfigi**:
# 
# ```python
# df = sov.data("query", tickers=["9033434", "IB94343", "43432", "AAPL"])
# ```
# 
# To filter **any dataframe** just write some queries:
# 
# ```python
# df.filter(["cash_short_term > 10m","start with ticker A","negative profits" ])
# ```
# 

# In[5]:


import sovai as sov
import pandas as pd

sov.token_auth(token="visit https://sov.ai/profile for your token")

data = sov.data("sec/10k", tickers=["AAPL"], limit=1).reset_index()

# In[6]:


data

# You don't have to downlaod the entire dataset, instead look at the companies that filed that week, and simply do an analysis of them. Future edits would include sentiment analysis.

# In[7]:


import pandas as pd

# Define mandatory sections
mandatory_sections = [
    "FINANCIAL_STATEMENTS",
    "RISK_FACTORS",
    "MANAGEMENT_DISCUSSION",
    "MANAGEMENT",
    "COMPENSATION",
    "EXHIBITS",
    "BUSINESS",
    "CONTROLS_AND_PROCEDURES",
    "PRINCIPAL_STOCKHOLDERS",
    "ACCOUNTING_FEES"
]

# 1. Filter for mandatory sections using isin (vectorized operation)
mandatory_data = data[data['section'].isin(mandatory_sections)].copy()

# 2. Extract year from the 'date' column
mandatory_data['year'] = mandatory_data['date'].dt.year

# 3. Calculate word count using vectorized string operations
mandatory_data['word_count'] = mandatory_data['full_text'].str.count(r'\w+')

# 4. Group by 'ticker', 'year', and 'section', then sum the 'word_count'
word_count_agg = mandatory_data.groupby(['ticker', 'year', 'section'], as_index=False)['word_count'].sum()

# 5. Pivot the DataFrame to create a multi-index with 'ticker' and 'year'
word_count_pivot = word_count_agg.pivot_table(
    index=['ticker', 'year'],
    columns='section',
    values='word_count',
    fill_value=0  # Fill missing values with 0
)

# Optional: Flatten the column index if needed (not necessary in multi-index)
# word_count_pivot.columns = word_count_pivot.columns.get_level_values(0)

# 6. Reset index to ensure 'ticker' and 'year' are part of the index
word_count_pivot = word_count_pivot.reset_index().set_index(['ticker', 'year'])


# In[8]:


word_count_pivot

# In[9]:


sov.data("sec/10k", tickers=["META"], limit=1)
