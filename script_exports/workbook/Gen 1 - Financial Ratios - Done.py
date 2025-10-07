#!/usr/bin/env python
# coding: utf-8

# In[ ]:


!pip install uv && uv pip install sovai['full'] --system > output.log 2>&1

# In[ ]:


# !pip install notion-client

# In[78]:


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


# In[58]:


import sovai as sov
import pandas as pd

sov.token_auth(token="visit https://sov.ai/profile for your token")

# In[59]:


tickers_meta = pd.read_parquet("data/tickers.parq")

# In[60]:


import pandas as pd
from sklearn.decomposition import PCA

df_ratios = sov.data("ratios/relative", frequency="summary",verbose=True, parquet=False)
df_data = df_ratios[df_ratios["lagged"].isin(["0_week","1_month"])].drop(columns=["lagged"])

# Define the feature mapping dictionary
feature_mapping = {
    'closeadj': {'characteristic': 'Price', 'direction': 1},
    'volume': {'characteristic': 'Volume', 'direction': 1},
    'market_cap': {'characteristic': 'Valuation', 'direction': 1},
    'evebit_ttm': {'characteristic': 'Valuation', 'direction': -1},
    'evebitda_ttm': {'characteristic': 'Valuation', 'direction': -1},
    'current_ratio': {'characteristic': 'Liquidity', 'direction': 1},
    'quick_ratio': {'characteristic': 'Liquidity', 'direction': 1},
    'cash_ratio': {'characteristic': 'Liquidity', 'direction': 1},
    'operating_cash_flow_ratio': {'characteristic': 'Liquidity', 'direction': 1},
    'net_working_capital_ratio': {'characteristic': 'Liquidity', 'direction': 1},
    'acid_test_ratio': {'characteristic': 'Liquidity', 'direction': 1},
    'excess_cash_margin_ratio': {'characteristic': 'Cash Flow', 'direction': 1},
    'earnings_per_share': {'characteristic': 'Profitability', 'direction': 1},
    'eps_usd': {'characteristic': 'Profitability', 'direction': 1},
    'eps_diluted': {'characteristic': 'Profitability', 'direction': 1},
    'gross_profit_margin': {'characteristic': 'Profitability', 'direction': 1},
    'operating_profit_margin': {'characteristic': 'Profitability', 'direction': 1},
    'ebitda_margin': {'characteristic': 'Profitability', 'direction': 1},
    'net_profit_margin': {'characteristic': 'Profitability', 'direction': 1},
    'return_on_assets': {'characteristic': 'Profitability', 'direction': 1},
    'return_on_equity': {'characteristic': 'Profitability', 'direction': 1},
    'return_on_net_assets': {'characteristic': 'Profitability', 'direction': 1},
    'roce_sub_cash': {'characteristic': 'Profitability', 'direction': 1},
    'roce_with_cash': {'characteristic': 'Profitability', 'direction': 1},
    'fcf_roce_with_cash': {'characteristic': 'Cash Flow', 'direction': 1},
    'fcf_roce_sub_cash': {'characteristic': 'Cash Flow', 'direction': 1},
    'income_dividend_payout_ratio': {'characteristic': 'Profitability', 'direction': -1},
    'return_on_invested_capital': {'characteristic': 'Profitability', 'direction': 1},
    'asset_turnover': {'characteristic': 'Efficiency', 'direction': 1},
    'inventory_turnover': {'characteristic': 'Efficiency', 'direction': 1},
    'days_sales_outstanding': {'characteristic': 'Efficiency', 'direction': -1},
    'days_inventory_outstanding': {'characteristic': 'Efficiency', 'direction': -1},
    'days_payable_outstanding': {'characteristic': 'Efficiency', 'direction': 1},
    'cash_conversion_cycle': {'characteristic': 'Efficiency', 'direction': -1},
    'total_asset_efficiency': {'characteristic': 'Efficiency', 'direction': 1},
    'working_capital_turnover_ratio': {'characteristic': 'Efficiency', 'direction': 1},
    'gross_operating_cycle': {'characteristic': 'Efficiency', 'direction': -1},
    'sg_and_gross_profit_ratio': {'characteristic': 'Profitability', 'direction': -1},
    'depreciation_revenue_ratio': {'characteristic': 'Efficiency', 'direction': -1},
    'depreciation_cfo_ratio': {'characteristic': 'Efficiency', 'direction': -1},
    'debt_ratio': {'characteristic': 'Solvency', 'direction': -1},
    'equity_multiplier': {'characteristic': 'Solvency', 'direction': -1},
    'interest_coverage_ratio': {'characteristic': 'Solvency', 'direction': 1},
    'debt_to_capital': {'characteristic': 'Solvency', 'direction': -1},
    'debt_service_coverage': {'characteristic': 'Solvency', 'direction': 1},
    'liabilities_equity_ratio': {'characteristic': 'Solvency', 'direction': -1},
    'debt_ebitda_ratio': {'characteristic': 'Solvency', 'direction': -1},
    'debt_ebitda_minus_capex_ratio': {'characteristic': 'Solvency', 'direction': -1},
    'debt_equity_ratio': {'characteristic': 'Solvency', 'direction': -1},
    'ebitda_interest_coverage': {'characteristic': 'Solvency', 'direction': 1},
    'ebitda_minus_capex_interest_coverage': {'characteristic': 'Solvency', 'direction': 1},
    'interest_to_cfo_plus_interest_coverage': {'characteristic': 'Solvency', 'direction': -1},
    'debt_to_total_capital': {'characteristic': 'Solvency', 'direction': -1},
    'debt_cfo_ratio': {'characteristic': 'Solvency', 'direction': -1},
    'ltdebt_cfo_ratio': {'characteristic': 'Solvency', 'direction': -1},
    'ltdebt_earnings_ratio': {'characteristic': 'Solvency', 'direction': -1},
    'cash_flow_to_debt_ratio': {'characteristic': 'Cash Flow', 'direction': 1},
    'cash_flow_coverage_ratio': {'characteristic': 'Cash Flow', 'direction': 1},
    'operating_cash_flow_to_sales': {'characteristic': 'Cash Flow', 'direction': 1},
    'free_cash_flow_conversion_ratio': {'characteristic': 'Cash Flow', 'direction': 1},
    'rough_dividend_payout_ratio': {'characteristic': 'Profitability', 'direction': -1},
    'dividends_cfo_ratio': {'characteristic': 'Profitability', 'direction': -1},
    'preferred_cfo_ratio': {'characteristic': 'Profitability', 'direction': -1},
    'cash_flow_reinvestment_ratio': {'characteristic': 'Cash Flow', 'direction': 1},
    'free_cashflow_ps': {'characteristic': 'Cash Flow', 'direction': 1},
    'enterprise_value': {'characteristic': 'Valuation', 'direction': 1},
    'price_to_earnings': {'characteristic': 'Valuation', 'direction': -1},
    'price_to_book': {'characteristic': 'Valuation', 'direction': -1},
    'price_to_sales': {'characteristic': 'Valuation', 'direction': -1},
    'dividend_yield': {'characteristic': 'Valuation', 'direction': 1},
    'market_to_book_ratio': {'characteristic': 'Valuation', 'direction': -1},
    'ev_opinc_ratio': {'characteristic': 'Valuation', 'direction': -1},
    'rough_ffo': {'characteristic': 'Cash Flow', 'direction': 1},
    'dividend_payout_ratio_pref': {'characteristic': 'Profitability', 'direction': -1},
    'dividend_payout_ratio': {'characteristic': 'Profitability', 'direction': -1},
    'retention_ratio': {'characteristic': 'Profitability', 'direction': 1},
    'greenblatt_earnings_yield': {'characteristic': 'Valuation', 'direction': 1},
    'enterprise_value_to_revenue': {'characteristic': 'Valuation', 'direction': -1},
    'enterprise_value_to_ebitda': {'characteristic': 'Valuation', 'direction': -1},
    'enterprise_value_to_ebit': {'characteristic': 'Valuation', 'direction': -1},
    'enterprise_value_to_invested_capital': {'characteristic': 'Valuation', 'direction': -1},
    'enterprise_value_to_free_cash_flow': {'characteristic': 'Valuation', 'direction': -1},
    'cash_productivity_ratio': {'characteristic': 'Efficiency', 'direction': 1},
    'debt_to_market_ratio': {'characteristic': 'Solvency', 'direction': -1},
    'net_debt_to_price_ratio': {'characteristic': 'Solvency', 'direction': -1},
    'cash_flow_to_price_ratio': {'characteristic': 'Valuation', 'direction': 1},
    'rd_to_market_ratio': {'characteristic': 'Efficiency', 'direction': 1},
    'book_to_market_enterprise_value_ratio': {'characteristic': 'Valuation', 'direction': 1},
    'equity_payout_yield': {'characteristic': 'Valuation', 'direction': 1},
    'equity_net_payout_yield': {'characteristic': 'Valuation', 'direction': 1},
    'ebitda_to_mev_ratio': {'characteristic': 'Valuation', 'direction': 1}
}

# Convert the dictionary to a pandas DataFrame
df_mapping = pd.DataFrame.from_dict(feature_mapping, orient='index').reset_index()
df_mapping = df_mapping.rename(columns={'index': 'feature'})

# Display the DataFrame
print(df_mapping)


feature_to_characteristic = df_mapping.set_index('feature')['characteristic']
feature_to_direction = df_mapping.set_index('feature')['direction']

# 5. Adjust `df_data` Based on Direction
# Create a copy to avoid modifying the original dataframe
df_adjusted = df_data.copy()

# Identify features with direction -1
negative_features = feature_to_direction[feature_to_direction == -1].index.tolist()

# Check which negative_features are present in df_adjusted
negative_features_present = [feature for feature in negative_features if feature in df_adjusted.columns]

# Transform the negative features
df_adjusted[negative_features_present] = 1 - df_adjusted[negative_features_present]

df_adjusted = df_adjusted.sort_index()


df_adjusted["Factor"] = PCA(n_components=1).fit_transform(df_adjusted.select_dtypes(include=['float64', 'int64']).fillna(df_adjusted.select_dtypes(include=['float64', 'int64']).mean()))

df_factor = df_adjusted["Factor"].to_frame()

del df_adjusted["Factor"]

# 6. Map Features to Characteristics
# Create a mapping dictionary from feature to characteristic
feature_char_map = feature_to_characteristic.to_dict()

# Rename the dataframe columns to their respective characteristics
# This will result in duplicate column names for features under the same characteristic
df_adjusted_renamed = df_adjusted.rename(columns=feature_char_map)

# 7. Compute the Mean Across Characteristics
# Group by the new column names (characteristics) and calculate the mean
# This assumes that after renaming, columns with the same name belong to the same characteristic
df_characteristic_means = df_adjusted_renamed.groupby(df_adjusted_renamed.columns, axis=1).mean()

# 8. (Optional) Reset Index if Needed
# If you want 'ticker' and 'date' as columns instead of the index
df_characteristic_means = df_characteristic_means.reset_index()


df_characteristic_means = df_characteristic_means.sort_values(["ticker","date"]).reset_index(drop=True)

df_factor = df_factor["Factor"].rank(pct=True)

df_characteristic_means = pd.merge(df_characteristic_means.set_index(["ticker","date"]), df_factor, left_index=True, right_index=True, how="right").reset_index()

new_df = df_characteristic_means.set_index(["ticker","date"]).groupby("ticker").diff().dropna()

new_df


# In[61]:



# Assume new_df is your existing DataFrame
# Initialize a list to hold the selected DataFrames
selected_dfs = []

# Iterate over each column in the DataFrame
for column in new_df.columns:
    # Select top 30 highest values for the current column
    top_30 = new_df.nlargest(60, column)
    # Select top 30 smallest values for the current column
    bottom_30 = new_df.nsmallest(60, column)
    # Append both selections to the list
    selected_dfs.extend([top_30, bottom_30])

# Concatenate all selected DataFrames into one final DataFrame
final_df = pd.concat(selected_dfs)

# In[62]:


# final_df = new_df.copy()

# (Optional) Reset index if needed
final_df.reset_index(inplace=True)


final_df = final_df.drop_duplicates()

final_df["Composite"] = final_df.drop(columns=["ticker","date","Factor","Volume"]).mean(axis=1)


# Define the renaming mapping and process dataframe as before
rename_mapping = {
    'ticker': 'Ticker',
    'date': 'Date',
    'Composite': 'Comp',
    'Cash Flow': 'CF',
    'Efficiency': 'Eff',
    'Liquidity': 'Liq',
    'Price': 'Price',
    'Profitability': 'Prof',
    'Solvency': 'Solv',
    'Valuation': 'Value',
    'Volume': 'Vol',
    'Factor': 'Fac',
}

final_df.rename(columns=rename_mapping, inplace=True)
final_df = (final_df.set_index(["Ticker","Date"])*100).round(0).astype(int).reset_index()
final_df.sort_values("Fac")
final_df = final_df.drop(columns=["Date"]).sort_values("Fac")


# In[63]:


cols = ['Ticker'] + ['Comp'] + [col for col in final_df.columns if col not in ['Ticker', 'Comp']]
final_df = final_df[cols]

# In[64]:


final_df['Ticker'] = final_df['Ticker'].apply(
    lambda x: f"[{x}](https://finance.yahoo.com/quote/{x})"
)

# In[65]:


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


# In[66]:


from datawrapper import Datawrapper


# Initialize Datawrapper
dw = Datawrapper(access_token="your_token")

# Create a new table
chart = dw.create_chart(
    title="Weekly Changes in Company Health (%)", 
    chart_type="tables",
    data=final_df
)

# Create the byline text
byline = """Column definitions (all values represent daily percentage changes in composite ratio scores):
Tkr: Ticker Symbol | Comp: Composite Score | CF: Cash Flow | Eff: Efficiency | Liq: Liquidity | Price: Price | Prof: Profitability | 
Solv: Solvency | Value: Valuation | Vol: Volume | Fac: Factor | """

# Update the chart properties with enhanced styling
metadata = {
    "visualize": {
        "dark-mode-invert": True,
        "perPage": 15,
        "columns": {
            "Ticker": {
                "align": "left",
                "title": "Tkr",
                "width": "80px",
                "markdown": True,
                "fixedWidth": True,
                "bold": True
            }
        },
        "header": {
            "style": {
                "bold": True,
                "fontSize": 0.91,
                "color": "#494949"
            },
            "borderBottom": "2px",
            "borderBottomColor": "#333333"
        },
        "heatmap": {
            "mode": "continuous",
            "stops": "equidistant",
            "colors": [
                {"color": "#ff4444", "position": 0},
                {"color": "#ffb6b6", "position": 0.25},
                {"color": "#ffffff", "position": 0.5},
                {"color": "#b6dbb6", "position": 0.75},
                {"color": "#44ff44", "position": 1}
            ],
            "rangeMax": "30",
            "rangeMin": "-30",
            "stopCount": 5,
            "hideValues": False,
            "interpolation": "equidistant"
        },
        "striped": True,
        "pagination": {"enabled": True, "position": "bottom"},
        "markdown": True,
        "showHeader": True,
        "compactMode": True,
        "sortDirection": "desc",
        "sortBy": "Comp"
    },
    "describe": {
           "intro": (
            "Daily percentage changes in composite financial metrics derived from ratio analysis across different dimensions of company performance."
            f" {formatted_week_label}."
            " Derived from <a href='https://docs.sov.ai/realtime-datasets/equity-datasets/financial-ratios'>Sov.ai™ Ratios</a> datasets."
            # "Learn more about our <a href='https://docs.sov.ai/methodology'>methodology</a> and <a href='https://docs.sov.ai/data-sources'>data sources</a>."
        ),
        "byline": byline,
        "source-name": "Sov.ai™ Financial Ratios",
        "source-url": "https://docs.sov.ai/realtime-datasets/equity-datasets/financial-ratios",
        "hide-title": False,
        "markdown": True
    },
    "publish": {
        "embed-width": 595,
        "embed-height": 668,
        "blocks": {
            "logo": {"enabled": False},
            "embed": False,
            "download-pdf": False,
            "download-svg": False,
            "get-the-data": True,
            "download-image": False
        },
        "autoDarkMode": False,
        "chart-height": 570,
        "force-attribution": False
    }
}

# Add properties for numeric columns
numeric_columns = ["Price", "CF", "Eff", "Liq", "Prof", "Solv", "Value", "Vol", "Fac", "Comp"]
for column in numeric_columns:
    metadata["visualize"]["columns"][column] = {
        "title": column,
        "width": "60px",
        "format": "+0.0%",
        "align": "right",
        "heatmap": {
            "enabled": True,
            "colors": {
                "min": "#ff4444",
                "max": "#44ff44"
            }
        },
        "fixedWidth": True
    }

# Update and publish the chart
dw.update_chart(chart['id'], metadata=metadata)
dw.publish_chart(chart['id'])

# Get and print the published URL
published_url = dw.get_chart_display_urls(chart['id'])
print("Published Chart URL:", published_url)

# In[67]:


url_ratios = dw.get_chart_display_urls(chart['id'])[0]["url"]

# In[68]:


normalized_df = new_df.reset_index().copy()

normalized_df["Composite"] = normalized_df.drop(columns=["ticker","date","Factor","Volume"]).mean(axis=1)

# Using pandas rank method with 'fraction' normalization
for column in normalized_df.select_dtypes(include=['float64', 'int64']).columns:
    normalized_df[column] = normalized_df[column].rank(method='average', pct=True)

df_filter = pd.read_parquet("https://storage.googleapis.com/sovai-public/concats/filters/latest.parquet")

normalized_df = normalized_df.merge(df_filter[["sector","market_cap"]].reset_index(), on="ticker",how="left")

normalized_df = normalized_df[["ticker", "Cash Flow","Profitability","sector","market_cap"]].dropna()

normalized_df = normalized_df.sort_values("market_cap").tail(1000)

# Rename columns in the DataFrame
normalized_df = normalized_df.rename(columns={
    'Cash Flow': 'Cash Flow Improved',
    'Profitability': 'Profitability Improved'
})

from datawrapper import Datawrapper
# Initialize Datawrapper
dw = Datawrapper(access_token="your_token")

# Create a new scatter plot
chart = dw.create_chart(
    title="Cash Flow and Profitability Improvements",
    chart_type="d3-scatter-plot"
)

# Configure the visualization properties with updates
metadata = {
    "visualize": {
        "dark-mode-invert": True,
        "size": "dynamic",
        "shape": "fixed",
        "opacity": "0.81",
        "max-size": 62.39,
        "outlines": True,
        "fixed-size": 6.36,
        "color-outline": "#ffffff",
        "show-color-key": False,  # Remove color legend
        "color-by-column": True,
        "hover-highlight": True,
        "plotHeightFixed": 434.32,
        "show-size-legend": False,  # Remove size legend
        "size-legend-type": "stacked",
        "tooltip": {
            "body": "Market Cap: <b>{{ market_cap }}</b>",  # Simplified tooltip
            "title": "{{ ticker }} <small>({{ sector }})</small>",
            "sticky": True,
            "enabled": True
        },
        "color-category": {
            "map": {
                "Healthcare": "#FF5872",
                "Basic Materials": "#00D5E9",
                "Consumer Defensive": "#ffe730",
                "Industrials": "#7FEB00",
                "Technology": "#FF8C00",
                "Financial Services": "#9370DB",
                "Consumer Cyclical": "#20B2AA",
                "Real Estate": "#FFB6C1",
                "Energy": "#4682B4",
                "Communication Services": "#DDA0DD",
                "Utilities": "#90EE90"
            }
        }
    },
    "axes": {
        "x": "Cash Flow Improved",  # Updated column name
        "y": "Profitability Improved",  # Updated column name
        "size": "market_cap",
        "color": "sector",
        "labels": "ticker"
    },
    "describe": {
        "intro": "Visualization showing the relationship between recent improvements in cash-flow and profitability.",
        "source-name": "Market Data",
        "byline": ""
    }
}

# Update the chart with our configuration
dw.update_chart(chart['id'], metadata=metadata)

# Add the data to the chart
dw.add_data(chart['id'], data=normalized_df)

# Publish the chart
dw.publish_chart(chart['id'])

# Display the chart
iframe_code = dw.get_iframe_code(chart['id'])
print(iframe_code)

# In[69]:


url_plot = dw.get_chart_display_urls(chart['id'])[0]["url"]

# In[70]:


url_ratios

# In[77]:


# Define title
page_title = "Predict a Mockingbird"

# Define content sections using the content_sections dictionary
content_sections = {
    "section_1": {
        "heading": f"Financial Ratio Radar - {datetime.now().strftime('%Y-%m-%d')}",
        "content": (
            "This is a financial early-warning system that monitors △ changes in ratios to identify critical positive and negative trends in companies before major players take note. It focuses on 8 key areas:"
        ),
        "url": url_ratios,
        "list": [
            "Cash Flow (CF): Change in how well the company manages its cash",
            "Efficiency (EF): Change in how well they use their resources",
            "Liquidity (LIQ): Change in how easily they can pay short-term bills",
            "Profitability (Prof): Change in how much money they're making",
            "Solvency (Solv): Change in how well they can handle their debts",
            "Valuation (Value): Change in how fair their stock price is",
            "Composite (Comp): Combines all 6 measures into one number",
            "Factor (Fac): A measure of statistical change (can ignore)"
        ]
    },
    "section_2": {
        "heading": None,
        "content": (
            "Watch two key trends: Cash Flow and Profits. Companies improving in both "
            "(top right of the plot) often make great investments - they're getting better at making money AND keeping it. "
            "Bad news: bottom left shows companies struggling with both."
        ),
        "url": url_plot,
        "list": None
    },

    "section_3": {
        "heading": None,
        "content": (
            "Download your own financial data from the Sov.ai ratio package to spot powerful "
            "market custom patterns and ratios yourself."
        ),
        "url": None,
        "list": None
    }
    # Add more sections as needed
}

# Handle page creation or append
handle_page_creation_or_append(page_title, DATABASE_ID, content_sections)

