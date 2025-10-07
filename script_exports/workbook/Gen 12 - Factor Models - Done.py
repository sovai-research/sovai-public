#!/usr/bin/env python
# coding: utf-8

# In[ ]:


!pip install uv && uv pip install sovai['full'] --system > output.log 2>&1

# In[12]:


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

sov.token_auth(token="visit https://sov.ai/profile for your token")

tickers_meta = pd.read_parquet("data/tickers.parq")

df_factors =  sov.data("factors/comprehensive")

df_factors["composite"] = df_factors.drop(columns=["returns"]).mean(axis=1)

# Get indices of top and bottom 15 from each column
high_low_df = pd.concat([
    pd.concat([
        df_factors.nlargest(30, columns=col),
        df_factors.nsmallest(30, columns=col)
    ]) for col in df_factors.columns
]).drop_duplicates()

high_low_df = high_low_df.reset_index()

high_low_df

# In[3]:


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


# In[13]:


from datawrapper import Datawrapper

# Initialize Datawrapper
dw = Datawrapper(access_token="your_token")

# Format the data
df = high_low_df.copy()
df['ticker'] = df['ticker'].apply(
    lambda x: f"[{x}](https://finance.yahoo.com/quote/{x})"
)

# Create the chart
chart = dw.create_chart(
    title="Stock Factor Model Scores",
    chart_type="tables"
)

# Add the data
dw.add_data(chart['id'], data=df)

# Configure visualization properties
properties = {
    "visualize": {
        "dark-mode-invert": True,
        "perPage": 20,
        "columns": {
            "ticker": {
                "align": "left",
                "title": "Stock",
                "width": "100",
                "markdown": True
            },
            "date": {
                "title": "Date",
                "width": "100",
                "format": "YYYY-MM-DD"
            },
            "composite": {
                "title": "Comp",
                "width": "80",
                "format": "0.0"
            },
            "returns": {
                "title": "Ret",
                "width": "80",
                "format": "+0.0%"
            },
            "value": {
                "title": "Value",
                "width": "80",
                "format": "0"
            },
            "profitability": {
                "title": "Prof",
                "width": "80",
                "format": "0"
            },
            "solvency": {
                "title": "Solv",
                "width": "80",
                "format": "0"
            },
            "cash_flow": {
                "title": "Cash",
                "width": "80",
                "format": "0"
            },
            "illiquidity": {
                "title": "Illiq",
                "width": "80",
                "format": "0"
            },
            "momentum_long_term": {
                "title": "Mom LT",
                "width": "80",
                "format": "0"
            },
            "momentum_medium_term": {
                "title": "Mom MT",
                "width": "80",
                "format": "0"
            },
            "short_term_reversal": {
                "title": "ST Rev",
                "width": "80",
                "format": "0"
            },
            "price_volatility": {
                "title": "Vol",
                "width": "80",
                "format": "0"
            },
            "dividend_yield": {
                "title": "Div",
                "width": "80",
                "format": "0"
            },
            "earnings_consistency": {
                "title": "Earn",
                "width": "80",
                "format": "0"
            },
            "small_size": {
                "title": "Size",
                "width": "80",
                "format": "0"
            },
            "low_growth": {
                "title": "Growth",
                "width": "80",
                "format": "0"
            },
            "low_equity_issuance": {
                "title": "Equity",
                "width": "80",
                "format": "0"
            },
            "bounce_dip": {
                "title": "Bounce",
                "width": "80",
                "format": "0"
            },
            "accrual_growth": {
                "title": "Accr",
                "width": "80",
                "format": "0"
            },
            "low_depreciation_growth": {
                "title": "Depr",
                "width": "80",
                "format": "0"
            },
            "current_liquidity": {
                "title": "Liq",
                "width": "80",
                "format": "0"
            },
            "low_rnd": {
                "title": "R&D",
                "width": "80",
                "format": "0"
            },
            "momentum": {
                "title": "Mom",
                "width": "80",
                "format": "0"
            },
            "market_risk": {
                "title": "Mkt",
                "width": "80",
                "format": "0"
            },
            "business_risk": {
                "title": "Bus",
                "width": "80",
                "format": "0"
            },
            "political_risk": {
                "title": "Pol",
                "width": "80",
                "format": "0"
            },
            "inflation_fluctuation": {
                "title": "Inf Fl",
                "width": "80",
                "format": "0"
            },
            "inflation_persistence": {
                "title": "Inf Pr",
                "width": "80",
                "format": "0"
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
        "markdown": True,
        "showHeader": True,
        "compactMode": True,
        "firstRowIsHeader": False,
        "firstColumnIsSticky": True
    },
    "describe": {
        "intro": ("Stock factor model scores showing key fundamental, technical, and risk metrics. All scores range from 0-100 with higher values indicating stronger signals."
                 f" {formatted_week_label}."
                 " Derived from <a href='https://docs.sov.ai/realtime-datasets/equity-datasets/factor-signals'>Sov.ai™ Factor</a> datasets."),
        "byline": "",
        "source-name": "Factor Model Data",
        "hide-title": False
    },
    "publish": {
        "embed-width": 1200,
        "embed-height": 800,
        "blocks": {
            "logo": {"enabled": False},
            "embed": False,
            "download-pdf": False,
            "download-svg": False,
            "get-the-data": True,
            "download-image": False
        },
        "chart-height": 700
    }
}

# Set column order
properties["visualize"]["column-order"] = [
    "ticker",
    "date",
    "composite",
    "returns",
    "profitability",
    "value",
    "solvency",
    "cash_flow",
    "illiquidity",
    "momentum_long_term",
    "momentum_medium_term",
    "short_term_reversal",
    "price_volatility",
    "dividend_yield",
    "earnings_consistency",
    "small_size",
    "low_growth",
    "low_equity_issuance",
    "bounce_dip",
    "accrual_growth",
    "low_depreciation_growth",
    "current_liquidity",
    "low_rnd",
    "momentum",
    "market_risk",
    "business_risk",
    "political_risk",
    "inflation_fluctuation",
    "inflation_persistence"
]

# Update the chart with the properties
dw.update_chart(
    chart['id'],
    metadata=properties
)

# Publish the chart
dw.publish_chart(chart['id'])

# Get the published URL
published_url_factors = dw.get_chart_display_urls(chart['id'])
print("Published Chart URL:", published_url)

# In[5]:


df_coefficients = sov.data("factors/coefficients")

coeff= df_coefficients.mean(axis=1).to_frame(); coeff.columns = ["coeff"]

df_standard_errors = sov.data("factors/standard_errors")

serr= df_standard_errors.mean(axis=1).to_frame(); serr.columns = ["se"]

df_t_statistics = sov.data("factors/t_statistics")

tstat= df_t_statistics.abs().mean(axis=1).to_frame(); tstat.columns = ["tstat"]

df_model_metrics = sov.data("factors/model_metrics")

# In[6]:


df_stats = pd.concat([df_model_metrics[["rsquared","aic"]],tstat, coeff, serr],axis=1)

# In[8]:


df_stats = df_stats.reset_index().drop(columns=["date"])

# In[10]:


df_stats.sort_values("rsquared")

# In[ ]:


## That is interesting look at factor model breakdown of tstats over time to indicarte regime change.

# In[14]:


from datawrapper import Datawrapper

# Initialize Datawrapper
dw = Datawrapper(access_token="your_token")

# Create the chart
chart = dw.create_chart(
    title="Stock Factor Error Analysis",
    chart_type="tables"
)

# Add the data to the chart
dw.add_data(chart['id'], data=df_stats)

# Configure visualization properties
properties = {
    "visualize": {
        "dark-mode-invert": True,
        "perPage": 15,
        "columns": {
            "ticker": {
                "align": "left",
                "title": "Stock",
                "width": "100"
            },
            "rsquared": {
                "title": "R²",
                "align": "right",
                "format": "0.000"
            },
            "aic": {
                "title": "AIC",
                "align": "right",
                "format": "0.000"
            },
            "tstat": {
                "title": "t-Stat",
                "align": "right",
                "format": "0.000"
            },
            "coeff": {
                "title": "Coefficient",
                "align": "right",
                "format": "0.000"
            },
            "se": {
                "title": "Std Error",
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
        "sortBy": "rsquared",
        "sortDirection": "desc",
        "searchable": True
    },
    "describe": {
        "intro": "Statistical analysis of stocks factor model R-squared, AIC, t-statistics, coefficients and standard errors. Sorted by R-squared (highest to lowest).",
        "byline": "",
        "source-name": "Stock Returns Analysis",
        "hide-title": False
    },
    "publish": {
        "embed-width": 700,
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
published_url_errors = dw.get_chart_display_urls(chart['id'])
print("Published Chart URL:", published_url)

# In[16]:


from datetime import datetime
# Define title
page_title = "Predict a Mockingbird"

# Define content sections using the content_sections dictionary
content_sections = {
    "section_1": {
        "heading": "Factor Model Coefficients",
        "content": (
            "Analysis showing percentile rankings of factor model coefficients across stocks. "
            "Higher percentiles indicate stronger factor sensitivity relative to peers. "
            "Coefficients represent each stock's exposure to fundamental, technical, and risk factors, "
            "helping identify both systematic factor plays and unique alpha opportunities."
        ),
        "url": published_url_factors[0]["url"],
        "list": None
    },
    "section_2": {
        "heading": "Factor Model Error Analysis",
        "content": (
            "For long-term systematic investing, stocks with high R², negative AIC, high t-stats and low standard errors offer reliable factor"
            "exposure and predictable returns suitable for core portfolio holdings. In contrast, stocks with low R², positive AIC, lower t-stats"
            "and higher standard errors present opportunities for active traders seeking alpha through uncorrelated returns and market inefficiencies,"
            "though they require more sophisticated risk management and trading strategies."
            
        ),
        "url": published_url_errors[0]["url"],
        "list": None
    }

    # Add more sections as needed
}

# Handle page creation or append
handle_page_creation_or_append(page_title, DATABASE_ID, content_sections)
