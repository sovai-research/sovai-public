#!/usr/bin/env python
# coding: utf-8

# In[ ]:


!pip install uv && uv pip install sovai['full'] --system > output.log 2>&1

# In[4]:


import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
import os

# List of paths to the notebooks you want to run in sequence
notebooks_to_run = [
    "Gen - Financial Ratios.ipynb",
    "Gen - Financial Ratios.ipynb",
    # Add the paths for all 10 notebooks here
]

def run_notebook(notebook_path):
    """Executes a notebook and returns the output as a dictionary."""
    with open(notebook_path, "r", encoding="utf-8") as f:
        notebook = nbformat.read(f, as_version=4)

    ep = ExecutePreprocessor(timeout=600, kernel_name="python3")
    try:
        ep.preprocess(notebook, {"metadata": {"path": os.path.dirname(notebook_path)}})
        print(f"Successfully ran: {notebook_path}")
    except Exception as e:
        print(f"Error running notebook {notebook_path}: {e}")

# Run each notebook in succession
for notebook_path in notebooks_to_run:
    run_notebook(notebook_path)


# In[ ]:


import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
from google.cloud import storage
import os
import tempfile

# Initialize GCS client
client = storage.Client()

# Configuration
BUCKET_NAME = 'your-bucket'  # Replace with your GCS bucket name
NOTEBOOKS_FOLDER = 'notebooks/'  # Folder in the bucket where notebooks are stored
LOCAL_TEMP_DIR = tempfile.gettempdir()

# List of notebook filenames to run in sequence
notebooks_to_run = [
    "Gen - Financial Ratios.ipynb",
    "Notebook2.ipynb",
    "Notebook3.ipynb",
    # Add all 10 notebook filenames here
]

def download_notebook(bucket_name, source_blob_name, destination_file_name):
    """Downloads a notebook from GCS to a local temporary directory."""
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)
    print(f"Downloaded {source_blob_name} to {destination_file_name}")

def run_notebook(notebook_path):
    """Executes a notebook and saves the output."""
    with open(notebook_path, "r", encoding="utf-8") as f:
        notebook = nbformat.read(f, as_version=4)

    ep = ExecutePreprocessor(timeout=600, kernel_name="python3")

    try:
        ep.preprocess(notebook, {"metadata": {"path": os.path.dirname(notebook_path)}})
        output_path = notebook_path.replace('.ipynb', '_executed.ipynb')
        with open(output_path, "w", encoding="utf-8") as f:
            nbformat.write(notebook, f)
        print(f"Successfully ran and saved: {output_path}")
    except Exception as e:
        print(f"Error running notebook {notebook_path}: {e}")

def upload_notebook(bucket_name, source_file_name, destination_blob_name):
    """Uploads the executed notebook back to GCS."""
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_name)
    print(f"Uploaded {source_file_name} to {destination_blob_name}")

# Run each notebook in sequence
for notebook in notebooks_to_run:
    source_blob = os.path.join(NOTEBOOKS_FOLDER, notebook)
    local_notebook = os.path.join(LOCAL_TEMP_DIR, notebook)
    
    # Download the notebook from GCS
    download_notebook(BUCKET_NAME, source_blob, local_notebook)
    
    # Execute the notebook
    run_notebook(local_notebook)
    
    # Upload the executed notebook back to GCS (optional)
    executed_notebook = local_notebook.replace('.ipynb', '_executed.ipynb')
    destination_blob = os.path.join(NOTEBOOKS_FOLDER, executed_notebook.split('/')[-1])
    upload_notebook(BUCKET_NAME, executed_notebook, destination_blob)
    
    # Clean up local files
    os.remove(local_notebook)
    os.remove(executed_notebook)

