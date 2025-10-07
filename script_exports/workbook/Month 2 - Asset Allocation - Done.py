#!/usr/bin/env python
# coding: utf-8

# In[ ]:


!pip install uv && uv pip install sovai['full'] --system > output.log 2>&1

# In[8]:


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

sov.token_auth(token="visit https://sov.ai/profile for your token")

# In[3]:


import pandas as pd


df_allocate = sov.data("allocation/all"); df_allocate
df_past = sov.data("allocation/past"); df_past
df_future = sov.data("allocation/future"); df_future

performance = df_future.head(2).mean()-df_past.tail(2).mean()


# Recommended changes in asset allocation (relative basis)
recommended_changes = performance

# Scaling factor to convert relative changes to percentage points
scaling_factor = 1000  # Adjust this factor as needed to achieve desired change magnitudes

# Scale the recommended changes
scaled_changes = recommended_changes * scaling_factor

# Verify that the sum of scaled changes is approximately zero
if not abs(scaled_changes.sum()) < 1e-6:
    raise ValueError("The sum of scaled changes must be zero to maintain total allocation at 100%.")

# Initialize an equal-weighted portfolio
assets = scaled_changes.index.tolist()
num_assets = len(assets)
initial_allocation = pd.Series(100 / num_assets, index=assets)

def reallocate_assets(initial_alloc, changes):
    """
    Reallocate assets based on recommended changes.

    Parameters:
    - initial_alloc (pd.Series): Initial allocation percentages.
    - changes (pd.Series): Recommended changes in allocation percentages.

    Returns:
    - pd.Series: New asset allocations in percentages.
    """
    # Apply changes to all assets
    new_allocation = initial_alloc + changes

    # Validate that no allocation is negative
    if (new_allocation < 0).any():
        negative_assets = new_allocation[new_allocation < 0].index.tolist()
        raise ValueError(f"Allocations for assets {negative_assets} have become negative. Adjust the recommended changes or scaling factor.")

    # Calculate total allocation after changes
    total_allocation = new_allocation.sum()

    # Check if total allocation is approximately 100%
    if not abs(total_allocation - 100) < 1e-6:
        # Normalize allocations to ensure the total sums to 100%
        new_allocation = (new_allocation / total_allocation) * 100

    # Round allocations to three decimal places for precision
    new_allocation = new_allocation.round(3)

    return new_allocation

def generate_summary(initial_alloc, new_alloc, category_order=None):
    """
    Generate a summary report of the portfolio reallocation.

    Parameters:
    - initial_alloc (pd.Series): Initial allocation percentages.
    - new_alloc (pd.Series): New allocation percentages after reallocation.
    - category_order (list): Desired order of assets.

    Returns:
    - pd.DataFrame: Summary table with Asset, Initial Allocation, Change, New Allocation, Change Direction.
    """
    # Calculate the percentage point difference
    change = new_alloc - initial_alloc

    # Create a DataFrame for the summary
    summary_df = pd.DataFrame({
        'Initial Allocation (%)': initial_alloc,
        'Change (%)': change,
        'New Allocation (%)': new_alloc
    })

    # Determine Change Direction
    summary_df['Change Direction'] = summary_df['Change (%)'].apply(
        lambda x: 'Increased' if x > 0 else ('Decreased' if x < 0 else 'No Change')
    )

    # Reset index to turn asset names into a column
    summary_df = summary_df.reset_index().rename(columns={'index': 'Asset'})

    # If a category order is provided, reorder the DataFrame accordingly
    if category_order is not None:
        # Define 'Asset' as a categorical type with the specified order
        summary_df['Asset'] = pd.Categorical(summary_df['Asset'], categories=category_order, ordered=True)
        # Sort the DataFrame based on the categorical order
        summary_df = summary_df.sort_values('Asset').reset_index(drop=True)
    else:
        # If no specific order is provided, maintain the current order
        pass

    # Rearrange columns to place 'Asset' as the first column
    summary_df = summary_df[['Asset', 'Initial Allocation (%)', 'Change (%)', 'New Allocation (%)', 'Change Direction']]

    # Round 'Change (%)' to three decimal places for consistency
    summary_df['Change (%)'] = summary_df['Change (%)'].round(3)

    return summary_df

# Define the desired category order
category_order = ["equities", "bonds", "commodities", "real_estate", "dollar"]

# Perform reallocation
try:
    new_allocations = reallocate_assets(initial_allocation, scaled_changes)
except ValueError as ve:
    print(f"Reallocation Error: {ve}")
    exit(1)

# Generate the summary report with the specified order
summary_table = generate_summary(initial_allocation, new_allocations, category_order=category_order)


# In[5]:


import pandas as pd
from datawrapper import Datawrapper
from IPython.display import IFrame

# Initialize Datawrapper with your API key
dw = Datawrapper(access_token="your_token")


from datetime import datetime

# Get the current year and month in a natural language format
current_year_month = datetime.now().strftime("%B %Y")

# Update the title string
title = f"Portfolio Reallocation Summary - {current_year_month}"


# Create a new chart
chart = dw.create_chart(
    title=title,
    chart_type="tables",
    data=summary_table
)

# Get the chart ID
chart_id = chart['id']

# Update description
dw.update_description(
    chart_id,
    source_name="Portfolio Analysis",
    byline="Generated by Python script",
    intro="""This is a fast moving monthly reallocation model to take advantage of short-term moves | <a href="https://sov.ai">Sov.ai™</a>"""
)

# Customize table appearance
properties = {
    'visualize': {
        'columns': {
            'Initial Allocation (%)': {
                'format': 'number',
                'decimals': 3,
                'colorScale': {
                    'min': 0,
                    'max': 100,
                    'colors': ['#fee8c8', '#e34a33']
                },
                'width': 'auto'
            },
            'Change (%)': {
                'format': 'number',
                'decimals': 3,
                'colorScale': {
                    'min': summary_table['Change (%)'].min(),
                    'max': summary_table['Change (%)'].max(),
                    'colors': ['#d7191c', '#ffffbf', '#2c7bb6']
                },
                'width': 'auto'
            },
            'New Allocation (%)': {
                'format': 'number',
                'decimals': 3,
                'colorScale': {
                    'min': 0,
                    'max': 100,
                    'colors': ['#fee8c8', '#e34a33']
                },
                'width': 'auto'
            },
            'Change Direction': {
                'type': 'text',
                'cellColor': {
                    'Increased': '#a1d99b',
                    'Decreased': '#fc9272',
                    'No Change': '#d9d9d9'
                },
                'width': 'auto'
            }
        },
        'header-row': True,
        'border-width': 1,
        'stripe-rows': False,
        'even-odd': True,
        'header-color': '#333333',
        'background-color': '#ffffff',
        'text-color': '#333333',
        'first-row-bold': True,
        'column-separator': True,
        'row-separator': True,
        'show-row-hover': True,
        'row-hover-color': '#f5f5f5',
        'auto-width': True,
        'dark-mode': {
            'enabled': False,
            'auto': True,
        }
    },
    'publish': {
        'auto-layout': True
    }
}

# Update metadata
dw.update_metadata(chart_id, properties)

# Publish the chart
publish_result = dw.publish_chart(chart_id)

# Get the published chart URL
published_url = dw.get_chart_display_urls(chart_id)

print(published_url)
# Get the public URL from the publish result
public_url = publish_result['data']['publicUrl']

# Display the chart in the notebook
IFrame(src=public_url, width=800, height=600)

# In[6]:


from datawrapper import Datawrapper
import pandas as pd
from datetime import datetime, timedelta

# Initialize Datawrapper with your API token
# dw = Datawrapper(access_token="your_token")

df_allocation_new = sov.data("allocation/future")
# Deduct 0.2 from all the columns
df_allocation_new = df_allocation_new - 0.15
# Perform 0-1 normalization
# df_allocation_normalized = (df_allocation_new - df_allocation_new.min()) / (df_allocation_new.max() - df_allocation_new.min())
# Current date
today = datetime.now()
df_melt = df_allocation_new.reset_index().melt(id_vars='date', var_name='Category', value_name='Value')


# Pivot the data
# Assuming df_melt is your melted DataFrame with 'date', 'Category', and 'Value' columns
# Convert date to datetime if it's not already
df_melt['date'] = pd.to_datetime(df_melt['date'])

# Format date as 'Month Year'
df_melt['date_formatted'] = df_melt['date'].dt.strftime('%b %Y')



# Step 1: Extract the Year from the 'date' column
df_melt['year'] = pd.to_datetime(df_melt['date']).dt.year

# Step 2: Group by 'year' and 'Category', then sum the 'Value'
df_yearly_sum = df_melt.groupby(['year', 'Category'])['Value'].sum().reset_index()

# Step 3: Normalize the 'Value' so that for each year, the sum of allocations equals 1
df_yearly_sum['Value_normalized'] = df_yearly_sum.groupby('year')['Value'].transform(lambda x: x / x.sum())

# Step 4: Select and rename columns as needed
df_yearly_normalized = df_yearly_sum[['year', 'Category', 'Value_normalized']]

# Optional: Rename 'Value_normalized' to 'Allocation' for clarity
df_yearly_normalized = df_yearly_normalized.rename(columns={'Value_normalized': 'Allocation'})


dw_data = df_yearly_normalized.pivot(index='year', columns='Category', values='Allocation')
dw_data = dw_data.reset_index()

dw_data = dw_data[["year","equities", "bonds", "commodities", "real_estate", "dollar"]]

# In[9]:


from datawrapper import Datawrapper
from IPython.display import IFrame

# Initialize Datawrapper with your API token
dw = Datawrapper(access_token="your_token")

# Create the chart
chart = dw.create_chart(
    title="Risk Parity Asset Allocation Over Time",
    chart_type="d3-bars-stacked",
    data=dw_data
)

# Update chart metadata
metadata = {
    "visualize": {
        "base-color": 5,
        "custom-colors": {
            "equities": "#d62728",
            "bonds": "#1f77b4",
            "commodities": "#ff7f0e",
            "real_estate": "#9467bd",
            "dollar": "#2ca02c"
        },
        "color-key": {
            "enabled": True,
            "position": "top",
            "label_values": False,
            "horizontal": True
        },
        "legend": {
            "enabled": True,
            "position": "top",
            "horizontal": True 
        },
        "show-color-key": True,
        "x-grid": "on",
        "y-grid": "on",
        "rotate-labels": -45,
        "bar-padding": 60,
        "valueLabels": {
            "show": "hover",
            "format": "0.0%",
            "enabled": True,
            "placement": "outside"
        },
        "yAxisLabels": {
            "enabled": True,
            "alignment": "left",
            "placement": "outside"
        },
        "show-tooltips": True,
        "y-grid-format": "0%",
        "y-grid-labels": "auto",
        "stack-color-legend": False,
        "color-by-column": False,
        "base_color": 5,
        "color-category": {
            "map": {
                "equities": "#d62728",
                "bonds": "#1f77b4",
                "commodities": "#ff7f0e",
                "real_estate": "#9467bd",
                "dollar": "#2ca02c"
            },
            "palette": ["#d62728", "#1f77b4", "#ff7f0e", "#9467bd", "#2ca02c"],
            "categoryOrder": ["equities", "bonds", "commodities", "real_estate", "dollar"]
        },
        'dark-mode': {
            'enabled': False,
            'auto': True,
            'background-color': '#1e1e1e',
            'text-color': '#ffffff',
            'header-color': '#ffffff',
            'row-hover-color': '#2a2a2a'
        }
    },
    "axes": {
        "x": {
            "title": "Month and Year",
            "ticks": {
                "color": "#ffffff"
            }
        },
        "y": {
            "title": "Allocation Percentage",
            "range": [0, 1],
            "format": "0%"
        }
    },
    "describe": {
        "intro": "Asset allocation model over the long-term, 20% is the expected risk-parity percentage",
        "byline": "Your Name",
        "source-name": "Macro Machine Learning Prediction",
        "source-url": "https://yourdatasource.com"
    },
    "annotate": {
        "notes": "Data normalized. Hover over bars for detailed information."
    }
}

# Update the chart metadata
dw.update_metadata(chart['id'], metadata)

# Publish the chart
publish_result = dw.publish_chart(chart['id'])

# Get the public URL from the publish result
public_url_parity = publish_result['data']['publicUrl']

# Display the chart in the notebook
IFrame(src=public_url, width=800, height=600)

# In[12]:


# Define title
page_title = "Predict a Monthly Bird"

# Define content sections using the content_sections dictionary
content_sections  = {
    "section_1": {
        "heading": "Recommended Asset Shift",
        "content": "This is a purely data-centric recommendation model for changes in risk-parity allocations.",
        "url": public_url,
        "list": None
    },
    "section_2": {
        "heading": "Recommended Future Allocations",
        "content": "These recommendations are dynamic and change on a monthly basis, generally on the 3rd of every month",
        "url": public_url_parity,
        "list": None
    },

}

# Handle page creation or append
handle_page_creation_or_append(page_title, DATABASE_ID, content_sections)

# In[11]:


public_url

# In[ ]:


public_url
