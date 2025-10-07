#!/usr/bin/env python
# coding: utf-8

# In[ ]:


!pip install uv && uv pip install sovai['full'] --system > output.log 2>&1

# In[ ]:


### Consumer Financial Complaints is very mislabelled.

# In[3]:


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


# In[23]:


import sovai as sov
import pandas as pd
import numpy as np
from scipy.stats import zscore
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import seaborn as sns

# Authenticate with Sovai
sov.token_auth(token="visit https://sov.ai/profile for your token") 

# Load data
df_complaints = sov.data("complaints/public", verbose=True, full_history=True)

# Data Preprocessing
df_complaints['date'] = pd.to_datetime(df_complaints['date'])
today = pd.Timestamp.now()
thirteen_months_ago = today - pd.DateOffset(months=14)
df_complaints = df_complaints[df_complaints['date'] >= thirteen_months_ago]
max_date = df_complaints['date'].max()

def get_monthly_end_date(dt):
    return dt.replace(day=max_date.day) if dt.day <= max_date.day else (
        dt + pd.offsets.MonthEnd(0) - pd.Timedelta(days=max_date.day-1)
    )

df_complaints['month_end'] = df_complaints['date'].apply(get_monthly_end_date)
df_complaints.set_index('month_end', inplace=True)

df_monthly = df_complaints.groupby('ticker').resample('M', label='right', closed='right').agg({
    'company': 'first',
    'ticker': 'count',
    'culpability_score': 'mean',
    'complaint_score': 'mean',
    'grievance_score': 'mean',
    'total_risk_rating': 'mean',
    'similarity': 'mean'
})

df_monthly.rename(columns={'ticker': 'complaint_count'}, inplace=True)
df_monthly.reset_index(inplace=True)
df_monthly.sort_values(['month_end', 'ticker'], ascending=[False, True], inplace=True)

df_monthly['month_end'] = df_monthly['month_end'].apply(
    lambda x: x.replace(day=max_date.day) if x.month != max_date.month or x.year != max_date.year
    else max_date
)

# Handle missing values
missing_values = df_monthly.isnull().sum()
print("Missing Values in Each Column:\n", missing_values)

numerical_cols = ['complaint_count', 'culpability_score', 'complaint_score',
                  'grievance_score', 'total_risk_rating', 'similarity']
df_monthly[numerical_cols] = df_monthly[numerical_cols].fillna(df_monthly[numerical_cols].median())

df_monthly['ticker'] = df_monthly['ticker'].fillna('UNKNOWN')
df_monthly['company'] = df_monthly['company'].fillna('UNKNOWN')

# Define the Anomaly Detection Function
def detect_anomalies(df, metrics, historical_months=12, contamination=0.05):
    """
    Detect anomalies in the latest month's data based on historical data.

    Parameters:
    - df (pd.DataFrame): The monthly aggregated data DataFrame.
    - metrics (list): List of metric column names to use for anomaly detection.
    - historical_months (int): Number of months to consider as historical data.
    - contamination (float): The proportion of anomalies in the data set for Isolation Forest.

    Returns:
    - pd.DataFrame: DataFrame with anomaly scores and labels.
    """
    # Ensure 'month_end' is in datetime format
    df['month_end'] = pd.to_datetime(df['month_end'])

    # Sort data by 'month_end' to ensure chronological order
    df = df.sort_values('month_end')

    # Identify the latest month in the data
    latest_month = df['month_end'].max()

    # Define the historical period
    historical_start_date = latest_month - pd.DateOffset(months=historical_months)

    # Extract historical data (12 months before the latest month)
    historical_data = df[(df['month_end'] >= historical_start_date) & 
                         (df['month_end'] < latest_month)]

    # Extract current month data
    current_month_data = df[df['month_end'] == latest_month]

    # Check if historical_data is empty
    if historical_data.empty:
        raise ValueError("Historical data is empty. Adjust the historical_months parameter or check the data.")

    # a. Calculate historical statistics (mean and std) for each metric per ticker
    stats = historical_data.groupby('ticker')[metrics].agg(['mean', 'std']).reset_index()

    # Flatten MultiIndex columns
    stats.columns = ['ticker'] + [f"{metric}_{stat}" for metric in metrics for stat in ['mean', 'std']]

    # b. Merge historical statistics with current month data
    merged_data = current_month_data.merge(stats, on='ticker', how='left')

    # c. Handle missing statistics
    for metric in metrics:
        mean_col = f"{metric}_mean"
        std_col = f"{metric}_std"
        merged_data[mean_col] = merged_data[mean_col].fillna(merged_data[metric])
        merged_data[std_col] = merged_data[std_col].fillna(0)

    # d. Calculate Z-Scores for each metric
    for metric in metrics:
        z_col = f"{metric}_z"
        mean_col = f"{metric}_mean"
        std_col = f"{metric}_std"
        # Avoid division by zero by replacing 0 std with a small number
        merged_data[z_col] = (merged_data[metric] - merged_data[mean_col]) / merged_data[std_col].replace(0, 1e-6)
        merged_data[z_col] = merged_data[z_col].fillna(0)

    # e. Combine Z-Scores into a Signed Composite Anomaly Score
    # Instead of using Euclidean distance, sum the z-scores to preserve the direction
    merged_data['anomaly_score'] = merged_data[[f"{metric}_z" for metric in metrics]].sum(axis=1)
    
    # Alternatively, you can use the mean if you prefer normalization
    # merged_data['anomaly_score'] = merged_data[[f"{metric}_z" for metric in metrics]].mean(axis=1)

    # f. Isolation Forest for advanced anomaly detection (Optional)
    feature_cols = [f"{metric}_z" for metric in metrics]
    X = merged_data[feature_cols].fillna(0)

    # Initialize Isolation Forest
    iso_forest = IsolationForest(contamination=contamination, random_state=42)

    # Fit the model and predict anomalies
    iso_forest.fit(X)
    merged_data['anomaly_label_iso'] = iso_forest.predict(X)

    # Anomaly labels: -1 for anomalies, 1 for normal
    merged_data['is_anomaly_iso'] = merged_data['anomaly_label_iso'].apply(lambda x: 1 if x == -1 else 0)

    return merged_data

# Define Metrics and Detect Anomalies
metrics = [
    'complaint_count', 
    'culpability_score', 
    'complaint_score', 
    'grievance_score', 
    'total_risk_rating', 
    # 'similarity'
]

# Call the anomaly detection function
try:
    analyzed_data = detect_anomalies(df_monthly, metrics)
except ValueError as ve:
    print(f"Error: {ve}")
    analyzed_data = pd.DataFrame()  # Create an empty DataFrame in case of error

# Representation and Visualization
final_representation = analyzed_data[[
    'month_end', 'ticker', 'company', 'complaint_count', 'culpability_score',
    'complaint_score', 'grievance_score', 'total_risk_rating',
    'similarity', 'anomaly_score', 'is_anomaly_iso'
]]



# In[25]:


final_representation

# In[26]:


final_representation = final_representation[final_representation["complaint_count"]>3]

# In[27]:


final_representation = final_representation.drop(columns=["is_anomaly_iso"])

# In[31]:


final_representation.sort_values("anomaly_score",ascending=False)

# In[32]:


# Rename and reorder columns
df_risk = final_representation.copy()
selected_columns = [
    'ticker', 
    'company',
    'complaint_count',
    'total_risk_rating',
    'anomaly_score',
    'culpability_score',
    'complaint_score',
    'grievance_score',
    'similarity'
]

df_risk = df_risk[selected_columns].rename(columns={'anomaly_score': 'risk_change'})

# In[36]:


df_risk[["total_risk_rating","culpability_score","complaint_score","grievance_score"]] = df_risk[["total_risk_rating","culpability_score","complaint_score","grievance_score"]]*100

# In[38]:


df_risk = df_risk.sort_values('risk_change', ascending=False)


# In[43]:


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


# In[45]:


from datawrapper import Datawrapper
import pandas as pd

# Initialize Datawrapper
dw = Datawrapper(access_token="your_token")

# Create the chart
chart = dw.create_chart(
    title="Financial Institution Risk Analysis",
    chart_type="tables"
)

# Add the data to the chart
dw.add_data(chart['id'], data=df_risk)

# Configure visualization properties
properties = {
    "visualize": {
        "dark-mode-invert": True,
        "perPage": 10,
        "columns": {
            "ticker": {
                "align": "left",
                "title": "Ticker",
                "width": "100",
                "fixedWidth": False
            },
            "company": {
                "align": "left",
                "title": "Company",
                "width": "200",
                "fixedWidth": False
            },
            "complaint_count": {
                "title": "Complaints",
                "align": "right",
                "format": "0,0"
            },
            "total_risk_rating": {
                "title": "Risk Rating",
                "align": "right",
                "format": "+0.000"
            },
            "risk_change": {
                "title": "Risk Change",
                "align": "right",
                "width": 0.66,
                "format": "+0.000",
                "showAsBar": True,
                "barColor": 7,
                "barColorNegative": 1,
                "fixedWidth": True,
                "minWidth": 35
            },
            "culpability_score": {
                "title": "Culpability",
                "align": "right",
                "format": "0.000"
            },
            "complaint_score": {
                "title": "Complaint",
                "align": "right",
                "format": "+0.000"
            },
            "grievance_score": {
                "title": "Grievance",
                "align": "right",
                "format": "+0.000"
            },
            "similarity": {
                "title": "Similarity",
                "align": "right",
                "format": "0.000"
            }
        },
        "header": {
            "style": {
                "bold": True,
                "fontSize": 0.9,
                "color": "#494949"
            },
            "borderBottom": "2px",
            "borderBottomColor": "#333333"
        },
        "pagination": {
            "enabled": True,
            "position": "bottom",
            "pagesPerScreen": 10
        },
        "striped": True,
        "markdown": False,
        "showHeader": True,
        "compactMode": True,
        "firstRowIsHeader": False,
        "firstColumnIsSticky": True,
        "mergeEmptyCells": False,
        "sortBy": "risk_change",
        "sortDirection": "desc"
    },
    "describe": {
        "intro": (f"Analysis of financial institutions' risk metrics and complaint data.. Sorted by risk change (highest to lowest). {formatted_week_label}"
                 " Derived from <a href='https://docs.sov.ai/realtime-datasets/sectorial-datasets/cfpb-complaints'>Sov.ai™ Complaints</a> datasets."),
        
        "byline": "",
        "source-name": "Consumer Complaints Database",
        "hide-title": False
    },
    "publish": {
        "embed-width": 1000,
        "embed-height": 800,
        "blocks": {
            "logo": {"enabled": False},
            "embed": False,
            "download-pdf": False,
            "download-svg": False,
            "get-the-data": True,
            "download-image": False
        }
    }
}

# Update the chart with the properties
dw.update_chart(
    chart['id'],
    metadata=properties
)

# Publish the chart
dw.publish_chart(chart['id'])

# Get the published URL
published_url = dw.get_chart_display_urls(chart['id'])
print("Published Chart URL:", published_url)

# In[ ]:


# Define title
page_title = "Predict a Mockingbird"

# Define content sections using the content_sections dictionary
content_sections = {
    "section_1": {
        "heading": "Consumer Financial Complaints",
        "content": (
            "Here we investigate recent changes in risk for financial firms based on complaints they have received."
            " The mapping from ticker to names are not yet solved for this dataset, so use with caution."
            
        ),
        "url": published_url[0]["url"],
        "list": None
    }

    # Add more sections as needed
}

# Handle page creation or append
handle_page_creation_or_append(page_title, DATABASE_ID, content_sections)

