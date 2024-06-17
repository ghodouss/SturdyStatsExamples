## Dash Examples

This directory contains a set of prebuilt dash apps built on top of trained indices. 

#### IMPORTANT: 
All scripts require the `api_key` field to be filled in.

## Requirements
Python version 3.10+
To install the required packages:
`pip install -r requirements.txt`


## Scripts

### sunburst.py

Requires: a trained index (supervised or unsupervised).
This script constructs an interactive sunburst. The sunburst presents high level themes (topic_groups and topics). The sunburst is interactive. Clicking on a topic or topic group will return  

To run:
`python sunburst.py`

To view:
Go to URL: `http://localhost:8050/?folder_id={index_id}`
The `index_id` must be passed as a URL parameter. 
