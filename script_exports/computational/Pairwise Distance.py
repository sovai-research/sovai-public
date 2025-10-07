#!/usr/bin/env python
# coding: utf-8

# In[ ]:


!pip install uv && uv pip install sovai['full'] --system > output.log 2>&1

# ## Pairwise Interactions

# ### Interactions
# 
# 1. Cross-Sectional - The cross-sectional information between two tickers are compared.
#    1. Mean, Std, Skew, First Diff, Zero Crossing, Turning Points
#    3. Spectral Centroid, Hjorst, Hurst, Time Reversal
# 2. Time-Series - The time-series information between two tickers are compared.
#    1. DTW (Dynamic Time Warping)
#    2. Pearson Correlation. 
# 3. Panel Interactions - The entire dataset are compared on different axis.
#    1. Tucker Decomposition
# 

# In[26]:


import sovai as sov

sov.token_auth(token="visit https://sov.ai/profile for your token")

# In[31]:


# Load ratios - takes around 
df_factors = sov.data("factors/accounting", start_date="2020-01-26", purge_cache=True); df_factors.head()

# In[32]:


df_factors.query("ticker == 'AAPL'").align(df_factors.query("ticker == 'AMZN'"), join='inner', axis=0)

# In[33]:


df_slice = df_factors.select_stocks("mega").date_range("2020-01-01")

# ### Standard Cross-Sectional Distance (Mean, Cosine)

# In[8]:


dist_matrix = df_slice.distance()

# In[9]:


dist_matrix

# In[10]:


dist_matrix.sort_values("AAPL")[["AAPL"]].T

# #### Other Cross-Sectional Calculations

# The extracted features include `mean`, `ske`, `std` (standard deviation), `diffm` (first difference mean), `zcr` (zero crossing rate), `mac` (mean absolute change), `sc` (spectral centroid), `tp` (turning points), `acl1` (autocorrelation at lag 1), `hjorthm` (Hjorth mobility), `hurst` (Hurst exponent), `hist` (histogram mode with 5 bins), and `timerev` (time reversibility statistic). These features capture various aspects of the time series, including central tendency, variability, complexity, and temporal characteristics.

# In[11]:


features = ['mean', 'skew', 'std', 'diffm', 'zcr', 'mac', 'sc', 'tp', 'acl1', 'hjorthm', 'hurst', 'hist', 'timerev']

dist_matrix_all = df_slice.distance(orient="cross-sectional", on='ticker', distance='cosine', calculations=features)

# In[12]:


dist_matrix_all

# In[13]:


dist_matrix_all.sort_values("AAPL")[["AAPL"]].T

# *If you don't want to use the cosine distance to calculate the matrix, you can also use `euclidean` distance:

# In[14]:


dist_matrix_all = df_slice.distance(orient="cross-sectional", distance='euclidean', calculations=features)

# In[15]:


dist_matrix_all.sort_values("AAPL")[["AAPL"]].T

# ### Time Series Distance 

# #### Pearson Correlation (fast)

# You could also try `spearman` for a non-linear monotonic correlation measure.

# In[16]:


dist_matrix_pearson = df_slice.distance(orient="time-series", metric="pearson")

# In[17]:


dist_matrix_pearson.sort_values("AAPL")[["AAPL"]].T

# #### Dynamic Time Warping (slow)

# In[18]:


dist_matrix_dtw = df_slice.distance(orient="time-series", metric="dtw")

# In[23]:


dist_matrix_dtw.sort_values("AAPL")[["AAPL"]].T

# #### Various Other Measures
# 
# The distance metrics encompass a diverse range of approaches for comparing time series, capturing various aspects of similarity and difference:
# 
# 1. `dtw`: Dynamic Time Warping, which allows for non-linear alignment of time series.
# 2. `pearson` and `spearman`: Correlation-based distances, measuring linear and rank-based relationships respectively.
# 3. `euclidean` and `euclidean_int`: Euclidean distances, with the latter using interpolation for different-length series.
# 4. `pec`: Power Envelope Correlation, capturing similarities in the energy distribution of signals.
# 5. `frechet`: Fréchet distance, measuring similarity between curves.
# 6. `kl_divergence`: Kullback-Leibler divergence, comparing probability distributions.
# 7. `wasserstein`: Wasserstein distance, also known as Earth Mover's distance.
# 8. `jaccard`: Jaccard distance, comparing set similarity.
# 9. `bray_curtis`: Bray-Curtis dissimilarity, often used in ecology.
# 10. `hausdorff`: Hausdorff distance, measuring how far two subsets of a metric space are from each other.
# 11. `manhattan`: Manhattan distance, summing the absolute differences.
# 12. `chi2`: Chi-squared distance, useful for comparing histograms.
# 13. `hellinger`: Hellinger distance, measuring the similarity between probability distributions.
# 14. `canberra`: Canberra distance, weighted version of Manhattan distance.
# 15. `shannon_entropy`: Distance based on Shannon entropy, measuring information content.
# 16. `sample_entropy` and `approx_entropy`: Complexity measures based on repeating patterns.
# 17. `jensen_shannon`: Jensen-Shannon divergence, measuring similarity between probability distributions.
# 18. `renyi_entropy` and `tsallis_entropy`: Generalized entropy measures.
# 19. `mutual_information`: Distance based on mutual information, measuring mutual dependence.
# 
# These metrics capture a wide range of time series characteristics, including shape, statistical properties, complexity, and information content. They allow for comprehensive comparison of time series data, accommodating different aspects of similarity and difference in temporal patterns.

# In[20]:


dist_matrix_dtw = df_slice.distance(orient="time-series", metric="tsallis_entropy")

# In[21]:


dist_matrix_dtw.sort_values("AAPL")[["AAPL"]].T

# ### Panel Interactions

# The Tucker decomposition is our only panel distance metric, it can be seen as a generalisation of the CP decomposition: it decomposes the tensor into a small core tensor and factor matrices.

# In[34]:


dist_matrix_tucker = df_slice.distance(orient="panel")

# In[35]:


dist_matrix_tucker

# In[36]:


dist_matrix_tucker.sort_values("AAPL")[["AAPL"]].T

# ## Date Distance Matrices
# 
# Every distance calculation above was done be default accross `on="ticker"` we can also specify `on="date"`.

# In[37]:


## All previous functions converted to "date"
dist_matrix_mean = df_slice.distance(on="date")
dist_matrix_cos = df_slice.distance(orient="cross-sectional", on="date", distance='cosine', calculations=features)
dist_matrix_euc = df_slice.distance(orient="cross-sectional", on="date", distance='euclidean', calculations=features)
dist_matrix_pearson = df_slice.distance(orient="time-series", on="date", metric="pearson")
dist_matrix_dtw = df_slice.distance(orient="time-series", on="date", metric="dtw")
dist_matrix_tsent = df_slice.distance(orient="time-series", on="date", metric="tsallis_entropy")
dist_matrix_tucker = df_slice.distance(orient="panel", on="date")

# In[38]:


dist_matrix_cos

# In[39]:


date = dist_matrix_cos.index.max()
dist_matrix_cos.sort_values(date)[[date]].T
