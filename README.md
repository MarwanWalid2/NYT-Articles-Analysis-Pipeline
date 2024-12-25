# NYT Articles Analysis Pipeline

## Overview
A comprehensive data pipeline that performs end-to-end analysis of New York Times articles, from data collection to topic modeling visualization. The project demonstrates advanced NLP techniques, web scraping, and machine learning approaches to extract insights from news articles.

## Features
- Automated article collection using NYT API
- Web scraping of full article content
- Advanced text preprocessing and normalization
- Topic modeling using Latent Dirichlet Allocation (LDA)
- Interactive visualization of topic distributions
- Object-oriented design with modular components


## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/nyt-analysis.git
cd nyt-analysis
```

3. Set up NYT API credentials:
- Obtain an API key from [NYT Developer Portal](https://developer.nytimes.com/)
- Set your API key in the configuration file


## Technical Details

### Data Collection
- Uses NYT Article Search API for metadata retrieval
- Implements rate limiting to comply with API restrictions
- Web scraping with BeautifulSoup4 for full article content

### Text Processing
- Comprehensive text normalization pipeline
- Stopword removal and stemming
- Custom token filtering
- Handling of special characters and formatting

### Topic Modeling
- LDA implementation using Gensim
- Interactive visualization with pyLDAvis
- Configurable number of topics and model parameters

## Requirements
- Python 3.8+
- pandas
- numpy
- nltk
- beautifulsoup4
- gensim
- pyLDAvis
- requests
