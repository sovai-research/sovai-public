#!/usr/bin/env python
# coding: utf-8

# In[ ]:


!pip install uv && uv pip install sovai['full'] --system > output.log 2>&1

# In[2]:


import sovai as sov

sov.token_auth(token="visit https://sov.ai/profile for your token")

# In[7]:


graph_df = sov.sec_graph("AAPL", date="2024", ontology_type="causal", sentiment_filter=0.3, verbose=False)

# In[3]:


# Usage example:
G, dfg, summary = analyze_10k_graph("MSFT", date="2024", ontology_type="temporal", sentiment_filter=0.6)
