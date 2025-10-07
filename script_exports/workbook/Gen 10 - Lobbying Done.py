#!/usr/bin/env python
# coding: utf-8

# In[ ]:


!pip install uv && uv pip install sovai['full'] --system > output.log 2>&1

# In[4]:


from datetime import datetime
import requests
import json

# Set up Notion credentials (hardcoded as per your request)
NOTION_TOKEN = "your_notion_token_here"  # **Ensure this token is kept secure!**
DATABASE_ID = "your_database_id_here"
NOTION_VERSION = "2022-06-28"

headers = {
    "Authorization": f"Bearer {NOTION_TOKEN}",
    "Content-Type": "application/json",
    "Notion-Version": NOTION_VERSION,
}

def create_page(title, database_id, children):
    """
    Creates a new page in the specified Notion database.

    Args:
        title (str): The title of the page.
        database_id (str): The ID of the Notion database.
        children (list): A list of block objects to include in the page.

    Returns:
        dict: The response from the Notion API.
    """
    page_data = {
        "parent": {"database_id": database_id},
        "properties": {
            "Title": {
                "title": [
                    {
                        "text": {
                            "content": title
                        }
                    }
                ]
            },
        },
        "children": children
    }

    response = requests.post("https://api.notion.com/v1/pages", headers=headers, json=page_data)
    return response


def find_page_by_title(database_id, title):
    """
    Searches the Notion database for a page with the specified title.

    Args:
        database_id (str): The ID of the Notion database.
        title (str): The title to search for.

    Returns:
        dict or None: The page object if found, else None.
    """
    query_url = f"https://api.notion.com/v1/databases/{database_id}/query"
    query_data = {
        "filter": {
            "property": "Title",
            "title": {
                "equals": title
            }
        }
    }

    response = requests.post(query_url, headers=headers, json=query_data)
    
    if response.status_code != 200:
        print("Failed to query database:")
        print(json.dumps(response.json(), indent=2))
        return None

    results = response.json().get("results")
    if results:
        return results[0]  # Assuming titles are unique
    return None


def append_to_page(page_id, children):
    """
    Appends new blocks to an existing Notion page.

    Args:
        page_id (str): The ID of the page to append to.
        children (list): A list of block objects to append.

    Returns:
        dict: The response from the Notion API.
    """
    append_url = f"https://api.notion.com/v1/blocks/{page_id}/children"
    append_data = {
        "children": children
    }
    response = requests.patch(append_url, headers=headers, json=append_data)
    return response


def build_content_from_dict(content_dict):
    """
    Builds Notion content blocks from a dictionary.

    Args:
        content_dict (dict): A dictionary containing content definitions.

    Returns:
        list: A list of Notion block objects.
    """
    children = []

    # Add Heading
    if "heading" in content_dict and content_dict["heading"]:
        children.append(
            {
                "object": "block",
                "type": "heading_2",
                "heading_2": {
                    "rich_text": [
                        {
                            "type": "text",
                            "text": {
                                "content": content_dict["heading"]
                            }
                        }
                    ]
                },
            }
        )

    # Add Content
    if "content" in content_dict and content_dict["content"]:
        children.append(
            {
                "object": "block",
                "type": "paragraph",
                "paragraph": {
                    "rich_text": [
                        {
                            "type": "text",
                            "text": {
                                "content": content_dict["content"]
                            }
                        }
                    ]
                },
            }
        )

        # Add List Items (Bullet Points)
    if "list" in content_dict and content_dict["list"]:
        list_blocks = build_bullet_list(content_dict["list"])
        children.extend(list_blocks)
        
    # Add URL as a Link
    if "url" in content_dict and content_dict["url"]:
        children.append(
            {
                "object": "block",
                "type": "paragraph",
                "paragraph": {
                    "rich_text": [
                        {
                            "type": "text",
                            "text": {
                                "content": content_dict["url"],
                                "link": {"url": content_dict["url"]}
                            }
                        }
                    ]
                },
            }
        )



    return children


def build_bullet_list(items):
    """
    Builds Notion bullet list blocks from a list of items.

    Args:
        items (list): A list of strings representing bullet points.

    Returns:
        list: A list of Notion bulleted list item block objects.
    """
    bullet_blocks = []
    for item in items:
        bullet_blocks.append(
            {
                "object": "block",
                "type": "bulleted_list_item",
                "bulleted_list_item": {
                    "rich_text": [
                        {
                            "type": "text",
                            "text": {
                                "content": item
                            }
                        }
                    ]
                },
            }
        )
    return bullet_blocks


def build_children_from_sections(content_sections):
    """
    Iterates through the content sections dictionary and builds the children blocks.

    Args:
        content_sections (dict): Dictionary containing all content sections.

    Returns:
        list: A list of Notion block objects.
    """
    children = []
    for key in sorted(content_sections.keys()):
        section = content_sections[key]
        section_blocks = build_content_from_dict(section)
        children.extend(section_blocks)
    return children


def handle_page_creation_or_append(title, database_id, content_sections):
    """
    Handles the logic to either create a new page or append content to an existing page.

    Args:
        title (str): The title of the page.
        database_id (str): The ID of the Notion database.
        content_sections (dict): Dictionary containing all content sections.

    Returns:
        None
    """
    current_date = datetime.now().strftime("%Y-%m-%d")
    full_title = f"{title} - {current_date}"

    # Build the content blocks
    children = build_children_from_sections(content_sections)

    # Check if the page already exists
    existing_page = find_page_by_title(database_id, full_title)

    if existing_page:
        print(f"Page '{full_title}' already exists. Appending new content to it.")
        page_id = existing_page["id"]
        response = append_to_page(page_id, children)
        
        if response.status_code == 200:
            print("New content appended successfully.")
            # Construct the page URL manually
            # Note: Notion page URLs follow the format https://www.notion.so/{workspace}/{page_id}
            # However, constructing the exact URL might require additional steps.
            # Here, we'll provide a placeholder.
            page_url = f"https://www.notion.so/{page_id.replace('-', '')}"
            print(f"View your page here: {page_url}")
        else:
            print("Failed to append new content:")
            print(json.dumps(response.json(), indent=2))
    else:
        print(f"Page '{full_title}' does not exist. Creating a new page with the new content.")
        response = create_page(full_title, database_id, children)
        
        # Handle the response
        if response.status_code == 200:
            page_url = response.json().get("url", "No URL returned")
            print("Page created successfully with the new content.")
            print(f"View your page here: {page_url}")
        else:
            print("Failed to create page:")
            print(json.dumps(response.json(), indent=2))


# In[1]:


import sovai as sov
import pandas as pd
import numpy as np
import ast

sov.token_auth(token="visit https://sov.ai/profile for your token")

tickers_meta = pd.read_parquet("data/tickers.parq")

df_lobbying = sov.data("lobbying/public",verbose=True, full_history=True)

df_lobbying = df_lobbying.reset_index()


# Optional: For Time Series Anomaly Detection
# Uncomment the following lines if you plan to use Prophet
# !pip install prophet
# from prophet import Prophet

# ---------------------------------------------
# Step 1: Load and Preprocess Data
# ---------------------------------------------

# Load your DataFrame
# Replace 'your_data.csv' with your actual data source
# Example:
# df_lobbying = pd.read_csv('your_data.csv')

# For demonstration purposes, let's assume df_lobbying is already loaded
# Uncomment and modify the following line as needed:
# df_lobbying = pd.read_csv('your_data.csv')

# Convert 'date' column to datetime
df_lobbying['date'] = pd.to_datetime(df_lobbying['date'], errors='coerce')

# Drop rows with invalid dates
df_lobbying.dropna(subset=['date'], inplace=True)

# Determine the maximum date in the dataset
max_date = df_lobbying['date'].max()

# Calculate the cutoff date (3 years before max_date)
three_years = pd.DateOffset(years=3)
cutoff_date = max_date - three_years

# Filter data for the last three years
recent_data = df_lobbying[df_lobbying['date'] >= cutoff_date].copy()

# Reset index if necessary
recent_data.reset_index(drop=True, inplace=True)

# ---------------------------------------------
# Step 2: Handle List-like Columns
# ---------------------------------------------

def safe_convert(x):
    """
    Safely convert string representations of lists to actual lists.
    Uses ast.literal_eval for security.
    """
    if isinstance(x, str):
        try:
            return ast.literal_eval(x)
        except (ValueError, SyntaxError):
            return []
    elif isinstance(x, (list, np.ndarray, tuple)):
        return list(x)
    else:
        return []

# List of columns that contain list-like data
list_like_columns = [
    'government_entity_details',
    'issue_codes',
    'previous_goverment_positions',
    'lobbyist_new_statuses',
    'lobbyist_full_names',
    'lobbyist_ids',
    'registrant_contact_name',
    'registrant_house_registrant_id',
    'registrant_contact_telephone',
    'match'
]

# Apply the conversion to list-like columns
for col in list_like_columns:
    if col in recent_data.columns:
        recent_data[col] = recent_data[col].apply(safe_convert)

# ---------------------------------------------
# Step 3: Weekly Aggregation Per Ticker
# ---------------------------------------------

# Define aggregation functions for scalar columns
scalar_aggregations = {
    'spend': ['sum', 'mean', 'median'],
    'transaction_type': 'count',
    'client': 'nunique'
    # 'government_entity_details': 'nunique'  # Removed due to list-like entries
}

# Group by 'ticker' and resample weekly
weekly_scalar = recent_data.groupby(
    ['ticker', pd.Grouper(key='date', freq='W')]
).agg(scalar_aggregations)

# Flatten MultiIndex columns
weekly_scalar.columns = ['_'.join(col).strip() for col in weekly_scalar.columns.values]

# Handle 'government_entity_details' separately
def count_unique_government_entities(series):
    unique_entities = set()
    for entry in series:
        if isinstance(entry, (list, tuple, np.ndarray)):
            unique_entities.update(entry)
        elif pd.notnull(entry):
            unique_entities.add(entry)
    return len(unique_entities)

weekly_gov_entities = recent_data.groupby(
    ['ticker', pd.Grouper(key='date', freq='W')]
)['government_entity_details'].apply(count_unique_government_entities).rename('Unique_Government_Entities')

# Group by 'ticker' and week, then aggregate spend by transaction type
spend_by_type = recent_data.groupby(
    ['ticker', pd.Grouper(key='date', freq='W'), 'transaction_type']
)['spend'].sum().unstack(fill_value=0).rename(columns=lambda x: f"Spend_by_{x}")

# Combine scalar aggregations with unique government entities
weekly_stats = weekly_scalar.join(weekly_gov_entities)

# Combine with spend by transaction type
weekly_stats = weekly_stats.join(spend_by_type)

# Reset index to turn 'ticker' and 'date' into columns
weekly_stats_reset = weekly_stats.reset_index()

# Rename columns for clarity
weekly_stats_reset.rename(columns={
    'spend_sum': 'Total_Spend',
    'spend_mean': 'Average_Spend',
    'spend_median': 'Median_Spend',
    'transaction_type_count': 'Transaction_Count',
    'client_nunique': 'Unique_Clients'
    # 'Unique_Government_Entities' is already appropriately named
}, inplace=True)

# Handle missing values (if any)
weekly_stats_reset.fillna(0, inplace=True)

# ---------------------------------------------
# Step 4: Anomaly Detection Methods
# ---------------------------------------------

# Group data by 'ticker' for calculations
grouped = weekly_stats_reset.groupby('ticker')

# --- Method 1: Z-Score ---
weekly_stats_reset['Mean_Spend'] = grouped['Total_Spend'].transform('mean')
weekly_stats_reset['Std_Spend'] = grouped['Total_Spend'].transform('std')
weekly_stats_reset['Z_Score'] = (weekly_stats_reset['Total_Spend'] - weekly_stats_reset['Mean_Spend']) / weekly_stats_reset['Std_Spend']
weekly_stats_reset['Z_Score'].fillna(0, inplace=True)  # Handle NaN std (e.g., std=0)

# Define Z-Score threshold
threshold_z = 2

# Flag anomalies based on Z-Score
weekly_stats_reset['Anomaly_Z_Score'] = weekly_stats_reset['Z_Score'] > threshold_z

# --- Method 2: Moving Average and Standard Deviation ---
window_size = 4  # e.g., 4 weeks
n_std = 2  # Number of standard deviations

# Sort data by 'ticker' and 'date' to ensure correct rolling calculations
weekly_stats_reset.sort_values(['ticker', 'date'], inplace=True)

# Calculate rolling mean and std
weekly_stats_reset['Rolling_Mean'] = grouped['Total_Spend'].transform(lambda x: x.rolling(window=window_size, min_periods=1).mean())
weekly_stats_reset['Rolling_Std'] = grouped['Total_Spend'].transform(lambda x: x.rolling(window=window_size, min_periods=1).std())
weekly_stats_reset['Rolling_Std'].fillna(0, inplace=True)  # Handle NaN std

# Calculate excess spend over the threshold
weekly_stats_reset['Excess_Spend_Moving_Avg'] = weekly_stats_reset['Total_Spend'] - (weekly_stats_reset['Rolling_Mean'] + n_std * weekly_stats_reset['Rolling_Std'])

# Replace negative excess spends with 0 (only consider positive deviations)
weekly_stats_reset['Excess_Spend_Moving_Avg'] = weekly_stats_reset['Excess_Spend_Moving_Avg'].apply(lambda x: x if x > 0 else 0)

# Flag anomalies based on Moving Average
weekly_stats_reset['Anomaly_Moving_Avg'] = weekly_stats_reset['Excess_Spend_Moving_Avg'] > 0

# --- Method 3: Percentile-Based ---
percentile = 0.90  # 90th percentile

# Calculate the 90th percentile for each ticker
weekly_stats_reset['Spend_90th_Percentile'] = grouped['Total_Spend'].transform(lambda x: x.quantile(percentile))

# Calculate excess spend over the percentile
weekly_stats_reset['Excess_Spend_Percentile'] = weekly_stats_reset['Total_Spend'] - weekly_stats_reset['Spend_90th_Percentile']

# Replace negative excess spends with 0 (only consider positive deviations)
weekly_stats_reset['Excess_Spend_Percentile'] = weekly_stats_reset['Excess_Spend_Percentile'].apply(lambda x: x if x > 0 else 0)

# Flag anomalies based on Percentile
weekly_stats_reset['Anomaly_Percentile'] = weekly_stats_reset['Excess_Spend_Percentile'] > 0


# ---------------------------------------------
# Step 5: Continuous Anomaly Scoring
# ---------------------------------------------

# Compute Percentile Ranks for each anomaly detection method within each ticker

# --- Percentile Rank for Z-Score ---
weekly_stats_reset['Z_Score_Pct'] = grouped['Z_Score'].transform(lambda x: x.rank(pct=True))

# --- Percentile Rank for Moving Average Excess Spend ---
weekly_stats_reset['Excess_Spend_Moving_Avg_Pct'] = grouped['Excess_Spend_Moving_Avg'].transform(lambda x: x.rank(pct=True))

# --- Percentile Rank for Percentile-Based Excess Spend ---
weekly_stats_reset['Excess_Spend_Percentile_Pct'] = grouped['Excess_Spend_Percentile'].transform(lambda x: x.rank(pct=True))


# Calculate the mean of the percentile ranks to get a consolidated anomaly score
weekly_stats_reset['Anomaly_Score'] = (
    weekly_stats_reset['Z_Score_Pct'] +
    weekly_stats_reset['Excess_Spend_Moving_Avg_Pct'] +
    weekly_stats_reset['Excess_Spend_Percentile_Pct']
    # + weekly_stats_reset['Anomaly_Prophet_Pct']  # Uncomment if using Prophet
) / 3  # Adjust denominator based on the number of methods used



import pandas as pd
import numpy as np
import ast

# ---------------------------------------------
# Step 1: Load and Preprocess Data
# ---------------------------------------------

# Load your DataFrame
# Replace 'your_data.csv' with your actual data source
# Example:
# df_lobbying = pd.read_csv('your_data.csv')

# For demonstration purposes, let's assume df_lobbying is already loaded
# Uncomment and modify the following line as needed:
# df_lobbying = pd.read_csv('your_data.csv')

# Ensure 'df_lobbying' is loaded. If not, raise an error.
try:
    df_lobbying
except NameError:
    raise NameError("DataFrame 'df_lobbying' is not loaded. Please load your data before proceeding.")

# Convert 'date' column to datetime
df_lobbying['date'] = pd.to_datetime(df_lobbying['date'], errors='coerce')

# Drop rows with invalid dates
df_lobbying.dropna(subset=['date'], inplace=True)

# Determine the maximum date in the dataset
max_date = df_lobbying['date'].max()

# Calculate the cutoff date (3 years before max_date)
three_years = pd.DateOffset(years=3)
cutoff_date = max_date - three_years

# Compute Total Spend over the entire dataset per ticker
total_spend_all = df_lobbying.groupby('ticker')['spend'].sum().reset_index().rename(columns={'spend': 'Total_Spend_All'})

# Filter data for the last three years
recent_data = df_lobbying[df_lobbying['date'] >= cutoff_date].copy()

# Reset index if necessary
recent_data.reset_index(drop=True, inplace=True)

# ---------------------------------------------
# Step 2: Handle List-like Columns
# ---------------------------------------------

def safe_convert(x):
    """
    Safely convert string representations of lists to actual lists.
    Uses ast.literal_eval for security.
    """
    if isinstance(x, str):
        try:
            return ast.literal_eval(x)
        except (ValueError, SyntaxError):
            return []
    elif isinstance(x, (list, np.ndarray, tuple)):
        return list(x)
    else:
        return []

# List of columns that contain list-like data
list_like_columns = [
    'government_entity_details',
    'issue_codes',
    'previous_goverment_positions',
    'lobbyist_new_statuses',
    'lobbyist_full_names',
    'lobbyist_ids',
    'registrant_contact_name',
    'registrant_house_registrant_id',
    'registrant_contact_telephone',
    'match'
]

# Apply the conversion to list-like columns
for col in list_like_columns:
    if col in recent_data.columns:
        recent_data[col] = recent_data[col].apply(safe_convert)

# ---------------------------------------------
# Step 3: Weekly Aggregation Per Ticker
# ---------------------------------------------

# Define aggregation functions for scalar columns
scalar_aggregations = {
    'spend': ['sum', 'mean', 'median'],
    'transaction_type': 'count',
    'client': 'nunique'
    # 'government_entity_details': 'nunique'  # Removed due to list-like entries
}

# Group by 'ticker' and resample weekly
weekly_scalar = recent_data.groupby(
    ['ticker', pd.Grouper(key='date', freq='W')]
).agg(scalar_aggregations)

# Flatten MultiIndex columns
weekly_scalar.columns = ['_'.join(col).strip() for col in weekly_scalar.columns.values]

# Handle 'government_entity_details' separately
def count_unique_government_entities(series):
    unique_entities = set()
    for entry in series:
        if isinstance(entry, (list, tuple, np.ndarray)):
            unique_entities.update(entry)
        elif pd.notnull(entry):
            unique_entities.add(entry)
    return len(unique_entities)

weekly_gov_entities = recent_data.groupby(
    ['ticker', pd.Grouper(key='date', freq='W')]
)['government_entity_details'].apply(count_unique_government_entities).rename('Unique_Government_Entities')

# Handle 'Unique_Lobbyists' by counting unique lobbyist_ids
def count_unique_lobbyists(series):
    unique_lobbyists = set()
    for entry in series:
        if isinstance(entry, (list, tuple, np.ndarray)):
            unique_lobbyists.update(entry)
        elif pd.notnull(entry):
            unique_lobbyists.add(entry)
    return len(unique_lobbyists)

weekly_unique_lobbyists = recent_data.groupby(
    ['ticker', pd.Grouper(key='date', freq='W')]
)['lobbyist_ids'].apply(count_unique_lobbyists).rename('Unique_Lobbyists')

# Group by 'ticker' and week, then aggregate spend by transaction type
spend_by_type = recent_data.groupby(
    ['ticker', pd.Grouper(key='date', freq='W'), 'transaction_type']
)['spend'].sum().unstack(fill_value=0).rename(columns=lambda x: f"Spend_by_{x}")

# Combine scalar aggregations with unique government entities and unique lobbyists
weekly_stats = weekly_scalar.join([weekly_gov_entities, weekly_unique_lobbyists])

# Combine with spend by transaction type
weekly_stats = weekly_stats.join(spend_by_type)

# Reset index to turn 'ticker' and 'date' into columns
weekly_stats_reset = weekly_stats.reset_index()

# Rename columns for clarity
weekly_stats_reset.rename(columns={
    'spend_sum': 'Total_Spend',
    'spend_mean': 'Average_Spend',
    'spend_median': 'Median_Spend',
    'transaction_type_count': 'Transaction_Count',
    'client_nunique': 'Unique_Clients'
    # 'Unique_Government_Entities' and 'Unique_Lobbyists' are already appropriately named
}, inplace=True)

# Handle missing values (if any)
weekly_stats_reset.fillna(0, inplace=True)

# ---------------------------------------------
# Step 4: Anomaly Detection Methods
# ---------------------------------------------

# Group data by 'ticker' for calculations
grouped = weekly_stats_reset.groupby('ticker')

# --- Method 1: Z-Score ---
weekly_stats_reset['Mean_Spend'] = grouped['Total_Spend'].transform('mean')
weekly_stats_reset['Std_Spend'] = grouped['Total_Spend'].transform('std')
weekly_stats_reset['Z_Score'] = (weekly_stats_reset['Total_Spend'] - weekly_stats_reset['Mean_Spend']) / weekly_stats_reset['Std_Spend']
weekly_stats_reset['Z_Score'].fillna(0, inplace=True)  # Handle NaN std (e.g., std=0)

# --- Method 2: Moving Average and Standard Deviation ---
window_size = 4  # e.g., 4 weeks
n_std = 2  # Number of standard deviations

# Sort data by 'ticker' and 'date' to ensure correct rolling calculations
weekly_stats_reset.sort_values(['ticker', 'date'], inplace=True)

# Calculate rolling mean and std
weekly_stats_reset['Rolling_Mean'] = grouped['Total_Spend'].transform(lambda x: x.rolling(window=window_size, min_periods=1).mean())
weekly_stats_reset['Rolling_Std'] = grouped['Total_Spend'].transform(lambda x: x.rolling(window=window_size, min_periods=1).std())
weekly_stats_reset['Rolling_Std'].fillna(0, inplace=True)  # Handle NaN std

# Calculate excess spend over the threshold
weekly_stats_reset['Excess_Spend_Moving_Avg'] = weekly_stats_reset['Total_Spend'] - (weekly_stats_reset['Rolling_Mean'] + n_std * weekly_stats_reset['Rolling_Std'])

# Replace negative excess spends with 0 (only consider positive deviations)
weekly_stats_reset['Excess_Spend_Moving_Avg'] = weekly_stats_reset['Excess_Spend_Moving_Avg'].apply(lambda x: x if x > 0 else 0)

# Flag anomalies based on Moving Average
weekly_stats_reset['Anomaly_Moving_Avg'] = weekly_stats_reset['Excess_Spend_Moving_Avg'] > 0

# --- Method 3: Percentile-Based ---
percentile = 0.90  # 90th percentile

# Calculate the 90th percentile for each ticker
weekly_stats_reset['Spend_90th_Percentile'] = grouped['Total_Spend'].transform(lambda x: x.quantile(percentile))

# Calculate excess spend over the percentile
weekly_stats_reset['Excess_Spend_Percentile'] = weekly_stats_reset['Total_Spend'] - weekly_stats_reset['Spend_90th_Percentile']

# Replace negative excess spends with 0 (only consider positive deviations)
weekly_stats_reset['Excess_Spend_Percentile'] = weekly_stats_reset['Excess_Spend_Percentile'].apply(lambda x: x if x > 0 else 0)

# Flag anomalies based on Percentile
weekly_stats_reset['Anomaly_Percentile'] = weekly_stats_reset['Excess_Spend_Percentile'] > 0

# ---------------------------------------------
# Step 5: Continuous Anomaly Scoring
# ---------------------------------------------

# Compute Percentile Ranks for each anomaly detection method within each ticker

# --- Percentile Rank for Z-Score ---
weekly_stats_reset['Z_Score_Pct'] = grouped['Z_Score'].transform(lambda x: x.rank(pct=True))

# --- Percentile Rank for Moving Average Excess Spend ---
weekly_stats_reset['Excess_Spend_Moving_Avg_Pct'] = grouped['Excess_Spend_Moving_Avg'].transform(lambda x: x.rank(pct=True))

# --- Percentile Rank for Percentile-Based Excess Spend ---
weekly_stats_reset['Excess_Spend_Percentile_Pct'] = grouped['Excess_Spend_Percentile'].transform(lambda x: x.rank(pct=True))

# --- Percentile Rank for Total Spend ---
# Incorporate Total_Spend as an additional factor
weekly_stats_reset['Total_Spend_Pct'] = grouped['Total_Spend'].transform(lambda x: x.rank(pct=True))

# Calculate the mean of the percentile ranks to get a consolidated anomaly score
# Now, including Total_Spend_Pct, so denominator becomes 4
weekly_stats_reset['Anomaly_Score'] = (
    weekly_stats_reset['Z_Score_Pct'] +
    weekly_stats_reset['Excess_Spend_Moving_Avg_Pct'] +
    weekly_stats_reset['Excess_Spend_Percentile_Pct'] +
    weekly_stats_reset['Total_Spend_Pct']
) / 4  # Adjust denominator based on the number of methods used

# ---------------------------------------------
# Step 6: Flagging Anomalies Based on Anomaly_Score
# ---------------------------------------------

# Define a threshold for anomaly score (e.g., top 95th percentile)
overall_threshold = weekly_stats_reset['Anomaly_Score'].quantile(0.95)

# Flag anomalies where Anomaly_Score exceeds the threshold
weekly_stats_reset['Anomaly_Flag'] = weekly_stats_reset['Anomaly_Score'] > overall_threshold

# ---------------------------------------------
# Step 7: Compute Quarterly and Annual Aggregates
# ---------------------------------------------

# Add 'year' and 'quarter' columns
weekly_stats_reset['year'] = weekly_stats_reset['date'].dt.year
weekly_stats_reset['quarter'] = weekly_stats_reset['date'].dt.to_period('Q')

# --- Quarterly Aggregates ---
quarterly_aggregates = weekly_stats_reset.groupby(['ticker', 'quarter']).agg({
    'Total_Spend': ['sum', 'mean', 'median'],
    'Transaction_Count': 'sum',
    'Unique_Clients': 'sum',  # Assuming sum makes sense; else use 'nunique'
    'Unique_Government_Entities': 'sum'  # Assuming sum makes sense
}).reset_index()

# Flatten MultiIndex columns
quarterly_aggregates.columns = ['ticker', 'quarter',
                                'Q_TS',  # Quarterly_Total_Spend
                                'Q_AS',  # Quarterly_Average_Spend
                                'Q_MS',  # Quarterly_Median_Spend
                                'Q_TC',  # Quarterly_Transaction_Count
                                'Q_UC',  # Quarterly_Unique_Clients
                                'Q_UGE']  # Quarterly_Unique_Government_Entities

# --- Annual Aggregates ---
annual_aggregates = weekly_stats_reset.groupby(['ticker', 'year']).agg({
    'Total_Spend': ['sum', 'mean', 'median'],
    'Transaction_Count': 'sum',
    'Unique_Clients': 'sum',  # Assuming sum makes sense; else use 'nunique'
    'Unique_Government_Entities': 'sum'  # Assuming sum makes sense
}).reset_index()

# Flatten MultiIndex columns
annual_aggregates.columns = ['ticker', 'year',
                             'A_TS',  # Annual_Total_Spend
                             'A_AS',  # Annual_Average_Spend
                             'A_MS',  # Annual_Median_Spend
                             'A_TC',  # Annual_Transaction_Count
                             'A_UC',  # Annual_Unique_Clients
                             'A_UGE']  # Annual_Unique_Government_Entities

# ---------------------------------------------
# Step 8: Merge Quarterly and Annual Aggregates with Weekly Data
# ---------------------------------------------

# Merge quarterly aggregates
weekly_stats_reset = weekly_stats_reset.merge(
    quarterly_aggregates,
    on=['ticker', 'quarter'],
    how='left'
)

# Merge annual aggregates
weekly_stats_reset = weekly_stats_reset.merge(
    annual_aggregates,
    on=['ticker', 'year'],
    how='left'
)

# Merge Total_Spend_All into weekly_stats_reset
weekly_stats_reset = weekly_stats_reset.merge(
    total_spend_all,
    on='ticker',
    how='left'
)

# ---------------------------------------------
# Step 9: Select Only the Latest Week's Data
# ---------------------------------------------

# Identify the latest week date in the dataset
latest_week_date = weekly_stats_reset['date'].max()

# Filter the DataFrame to include only the latest week's data
latest_week_data = weekly_stats_reset[weekly_stats_reset['date'] == latest_week_date].copy()

# ---------------------------------------------
# Step 10: Clean Up and Retain Only Relevant Columns with Descriptive Names
# ---------------------------------------------

# Define the columns to retain with more descriptive and self-explanatory names
# Selected 10 key columns:
# 1. Ticker
# 2. Date
# 3. Anomaly_Score
# 4. Total_Spend (Week Spend)
# 5. Quarterly_Spend (Quarter Spend)
# 6. Annual_Spend (Year Spend)
# 7. Total_Spend_All (Total Spend)
# 8. Transaction_Count (Transactions)
# 9. Unique_Lobbyists (Lobbyist)
# 10. Unique_Government_Entities (Gov Entities)

columns_to_keep = [
    'ticker',                      # Ticker
    'date',                        # Date
    'Anomaly_Score',               # Anomaly
    'Total_Spend',                 # Week Spend
    'Q_TS',                        # Quarter Spend
    'A_TS',                        # Year Spend
    'Total_Spend_All',             # Total Spend
    'Transaction_Count',           # Transactions
    'Unique_Lobbyists',            # Lobbyist
    'Unique_Government_Entities'   # Gov Entities
]

# Verify which of these columns exist in the DataFrame
existing_columns = [col for col in columns_to_keep if col in latest_week_data.columns]

# Create a mapping for shorter and more descriptive names
column_mapping = {
    'ticker': 'Ticker',
    'date': 'Date',
    'Anomaly_Score': 'Anomaly',
    'Total_Spend': 'Week Spend',
    'Q_TS': 'Quarter Spend',
    'A_TS': 'Year Spend',
    'Total_Spend_All': 'Total Spend',
    'Transaction_Count': 'Transaction Counts',
    'Unique_Lobbyists': 'Lobbyist Counts',
    'Unique_Government_Entities': 'Gov Entities'
}

# Filter the DataFrame
final_output = latest_week_data[existing_columns].copy()

# Rename the columns to more descriptive names
final_output.rename(columns=column_mapping, inplace=True)

# ---------------------------------------------
# Step 11: Scale Spending Columns by 1000
# ---------------------------------------------

# List of spending columns to scale
spend_columns = ['Week Spend', 'Quarter Spend', 'Year Spend', 'Total Spend']

# Check if these columns exist before scaling
spend_columns_existing = [col for col in spend_columns if col in final_output.columns]

# Scale the spending columns by dividing by 1000
final_output[spend_columns_existing] = final_output[spend_columns_existing] / 1000

# Optionally, round the spending columns for better readability
final_output[spend_columns_existing] = final_output[spend_columns_existing].round(3)

# ---------------------------------------------
# Step 12: Final Output
# ---------------------------------------------

# Define the final column order as specified
final_output = final_output[[
    "Ticker", 
    "Date",
    "Anomaly", 
    "Week Spend",
    "Quarter Spend", 
    "Year Spend", 
    "Total Spend", 
    "Transaction Counts",
    "Lobbyist Counts",
    "Gov Entities"
]]

# Sort the final output by Anoma


final_output.shape

final_output = final_output.sort_values("Anomaly",ascending=False)



# Create Yahoo Finance links for tickers
final_output['Ticker'] = final_output['Ticker'].apply(
    lambda x: f"[{x}](https://finance.yahoo.com/quote/{x})"
)

# Remove the Date column
final_output = final_output.drop(columns=['Date'])


# Configure visualization
# Configure visualization
# First rename the column in your DataFrame
final_output = final_output.rename(columns={'Transaction Counts': 'Trans. Counts',"Lobbyist Counts":'Lobby Accounts'})


final_output = final_output.drop(columns=["Trans. Counts"])

# In[2]:


final_output

# In[5]:


import datetime

import locale

# Set locale to US English
locale.setlocale(locale.LC_TIME, 'en_US.UTF-8')


def get_week_ending_label(reference_date=None):
    """
    Returns a formatted string indicating the week ending on the last Friday relative to the reference date.

    Args:
        reference_date (datetime.date, optional): The date to reference. Defaults to today.

    Returns:
        str: Formatted string like "Week ending Friday 25th October, 2024"
    """
    if reference_date is None:
        reference_date = datetime.date.today()
    
    def get_ordinal(n):
        if 11 <= n % 100 <= 13:
            suffix = 'th'
        else:
            suffix = {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')
        return f"{n}{suffix}"
    
    days_since_friday = (reference_date.weekday() - 4) % 7
    last_friday = reference_date - datetime.timedelta(days=days_since_friday)
    day_with_ordinal = get_ordinal(last_friday.day)
    formatted_date = f"Week ending {last_friday.strftime('%A')} {day_with_ordinal} {last_friday.strftime('%B')}, {last_friday.year}"
    
    return formatted_date

# Usage
formatted_week_label = get_week_ending_label()


# In[6]:


from datawrapper import Datawrapper


# Initialize Datawrapper
dw = Datawrapper(access_token="your_token")

# Create chart
chart = dw.create_chart(
    title="Company Lobbying Activity",
    chart_type="tables"
)

# Add data
dw.add_data(chart['id'], data=final_output)

# Configure visualization
# Configure visualization
# First rename the column in your DataFrame
final_output = final_output.rename(columns={'Transactions': 'Trans. Counts',"Lobbyist Counts":'Lobby Accounts'})

# Updated configuration
properties = {
    "visualize": {
        "dark-mode-invert": True,
        "columns": {
            "Ticker": {
                "align": "left",
                "title": "Company",
                "width": "100",
                "markdown": True,
                "fixedWidth": False
            },
            "Anomaly": {
                "style": {"fontSize": 1},
                "title": "Anomaly",
                "width": 0.45,
                "format": "0.0%",
                "barStyle": "slim",
                "showAsBar": True,
                "borderLeft": "none",
                "fixedWidth": True,
                "customBarColor": False,
                "barColorNegative": "#ff4444",
                "barColorPositive": "#44bb77",
                "customBarColorBy": "Ticker"
            },
            "Week Spend": {
                "title": "Week Spend",
                "width": "100",
                "format": "$0,0.0",
                "fixedWidth": False
            },
            "Quarter Spend": {
                "title": "Quarter Spend",
                "width": "100",
                "format": "$0,0.0",
                "fixedWidth": False
            },
            "Year Spend": {
                "title": "Year Spend",
                "width": "100",
                "format": "$0,0.0",
                "fixedWidth": False
            },
            "Total Spend": {
                "title": "Total Spend",
                "width": "100",
                "format": "$0,0.0",
                "fixedWidth": False
            },
            "Lobby Accounts": {
                "title": "Lobby.",
                "width": "80",
                "format": "0",
                "fixedWidth": False,
                "includeInHeatmap": True
            },
            "Gov Entities": {
                "title": "Gov. Ent.",
                "width": "80",
                "format": "0",
                "fixedWidth": False,
                "includeInHeatmap": True
            }
        },
        "header": {
            "style": {
                "bold": True,
                "color": "#494949",
                "italic": False,
                "fontSize": 0.9,
                "background": False
            },
            "borderTop": "none",
            "borderBottom": "2px",
            "borderTopColor": "#333333",
            "borderBottomColor": "#333333"
        },
        "heatmap": {
            "enabled": True,
            "mode": "continuous",
            "stops": "equidistant",
            "colors": [
                {"color": "#f0f9e8", "position": 0},
                {"color": "#b6e3bb", "position": 0.16666666666666666},
                {"color": "#75c8c5", "position": 0.3333333333333333},
                {"color": "#4ba8c9", "position": 0.5},
                {"color": "#2989bd", "position": 0.6666666666666666},
                {"color": "#0a6aad", "position": 0.8333333333333334},
                {"color": "#254b8c", "position": 1}
            ],
            "palette": 0,
            "rangeMax": "13",
            "rangeMin": "1",
            "stopCount": 5,
            "hideValues": False,
            "customStops": [],
            "rangeCenter": "4",
            "categoryOrder": [],
            "interpolation": "equidistant",
            "categoryLabels": {},
            "columns": ["Trans. Counts", "Lobbyist", "Gov Entities"]  # Specify columns to include in heatmap
        },
        "perPage": 15,
        "striped": True,
        "markdown": True,
        "showHeader": True,
        "compactMode": True,
        "firstRowIsHeader": False,
        "firstColumnIsSticky": True,
        "searchable": True,
        "search": {
            "enabled": True,
            "placeholder": "Search companies..."
        }
    },
    "describe": {
        "intro": ("Analysis of company lobbying activities. Data shows spending across different time periods (in thousands of dollars), number of transactions, lobbyists involved, and government entities engaged. The anomaly score indicates unusual lobbying activity patterns."
                 f" {formatted_week_label}."
                 " Derived from <a href='https://docs.sov.ai/realtime-datasets/equity-datasets/lobbying-data/'>Sov.ai™ Lobbying</a> datasets."),
        "byline": "",
        "source-name": "Lobbying Data",
        "hide-title": False
    },
    "publish": {
        "embed-width": 700,
        "embed-height": 714,
        "blocks": {
            "logo": {"enabled": False},
            "embed": False,
            "download-pdf": False,
            "download-svg": False,
            "get-the-data": False,
            "download-image": True
        },
        "autoDarkMode": False,
        "chart-height": 582,
        "force-attribution": False
    }
}

# Update and publish chart
dw.update_chart(
    chart['id'],
    metadata=properties
)

# Publish the chart
dw.publish_chart(chart['id'])

# Get the published URL
published_url = dw.get_chart_display_urls(chart['id'])
print("Published Chart URL:", published_url)

# In[8]:


from datetime import datetime
# Define title
page_title = "Predict a Mockingbird"

# Define content sections using the content_sections dictionary
content_sections = {
    "section_1": {
        "heading": "Corporate lobbying tracker",
        "content": (
            "Lobbiest have important roles to play in the American political system "
            " and are often responible for many of the bills being passed and the policies implemented."
            
        ),
        "url": published_url[0]["url"],
        "list": None
    }

    # Add more sections as needed
}

# Handle page creation or append
handle_page_creation_or_append(page_title, DATABASE_ID, content_sections)

