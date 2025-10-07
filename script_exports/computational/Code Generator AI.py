#!/usr/bin/env python
# coding: utf-8

# In[ ]:


!pip install uv && uv pip install sovai['full'] --system > output.log 2>&1

# ## Asset Allocation

# In[1]:


Actually this looks pretty good 

https://docs.llamaindex.ai/en/stable/getting_started/starter_example_local/

https://docs.llamaindex.ai/en/stable/examples/data_connectors/GithubRepositoryReaderDemo/

# In[8]:


import sovai as sov
import pandas as pd
# Set the option to display all columns
pd.set_option('display.max_columns', None)

sov.token_auth(token="visit https://sov.ai/profile for your token")
# And then continue working with get some data from API and manipulating them

# #### Processed Dataset

# In[ ]:


df_allocate = sov.data("I want data from different regimes asset classes")

# In[14]:


df_allocate = sov.data("regime/allocation")

# In[10]:


df_allocate = sov.data("regime/returns")

# In[15]:


df_allocate

# In[5]:


df_allocate

# In[11]:


df_wiki = sov.data("wikipedia/views")

# In[12]:


df_wiki.head()

# In[29]:


df_news.query("ticker == 'AAPL'").reset_index().set_index("date")[["sentiment","tone"]].plot()

# In[30]:


df_news_sent = sov.data("news/sentiment")

# In[37]:


df_news_sent[["MSFT","TSLA", "AMZN", "AAPL", "FB", "GOOG"]].corr()

# In[19]:


df_institute = sov.data("institutional/trading")

# In[20]:


df_institute.head()

# In[ ]:


df_price = pd.read_parquet(f"gs://sovai-accounting/dataframes/prices.parq", storage_options={'token': service_account_info})
