import os
import json
import time
import requests
import pandas as pd
import numpy as np
from datetime import datetime
from bs4 import BeautifulSoup
from typing import List, Dict, Optional

# NLP Libraries
import nltk
import string
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer

# Topic Modeling
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

# Visualization
import pyLDAvis
import pyLDAvis.gensim
import matplotlib.pyplot as plt

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

class NYTArticleCollector:
    """
    Handles collection of NYT articles using the NYT API and web scraping.
    """
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_2) AppleWebKit/601.3.9 (KHTML, like Gecko) Version/9.0.2 Safari/601.3.9'
        }

    def create_search_url(self, query: str, begin_date: str, end_date: str) -> str:
        """Creates the NYT API search URL with parameters."""
        return f"http://api.nytimes.com/svc/search/v2/articlesearch.json?q={query}&begin_date={begin_date}&end_date={end_date}&sort=oldest&api-key={self.api_key}"

    def send_request(self, base_url: str, page: int) -> Dict:
        """Sends a request to the NYT API with rate limiting."""
        url = f"{base_url}&page={page}"
        response = requests.get(url).json()
        time.sleep(6)  # Rate limiting
        return response

    def parse_response(self, response: Dict) -> pd.DataFrame:
        """Parses API response into a pandas DataFrame."""
        data = {
            'abstract': [], 'web_url': [], 'source': [], 'headline': [],
            'pub_date': [], 'document_type': [], 'news_desk': [],
            'section_name': [], 'word_count': [], '_id': []
        }
        
        articles = response['response']['docs']
        for article in articles:
            data['abstract'].append(article['abstract'])
            data['web_url'].append(article['web_url'])
            data['source'].append(article['source'])
            data['headline'].append(article['headline']['main'])
            data['pub_date'].append(article['pub_date'])
            data['document_type'].append(article['document_type'])
            data['news_desk'].append(article['news_desk'])
            data['section_name'].append(article.get('section_name'))
            data['word_count'].append(article['word_count'])
            data['_id'].append(article['_id'])
            
        return pd.DataFrame(data)

    def collect_metadata(self, query: str, begin_date: str, end_date: str, max_pages: int = 10) -> pd.DataFrame:
        """Collects article metadata from NYT API."""
        base_url = self.create_search_url(query, begin_date, end_date)
        df = pd.DataFrame()
        
        for page in range(1, max_pages + 1):
            response = self.send_request(base_url, page)
            if not response['response']['docs']:
                break
                
            tmp_df = self.parse_response(response)
            df = pd.concat([df, tmp_df])
            print(f'Collected & processed {page} query page!')
            
        print(f'Collected & processed metadata of {len(df)} articles!')
        return df

    def scrape_article_text(self, url: str) -> str:
        """Scrapes full article text from NYT website."""
        try:
            response = requests.get(url, headers=self.headers)
            soup = BeautifulSoup(response.content, 'lxml')
            paragraphs = soup.select('p.evys1bk0')
            
            text = ' '.join(p.text for p in paragraphs)
            return text.strip()
        except Exception as e:
            print(f"Error scraping article {url}: {str(e)}")
            return ""

class TextPreprocessor:
    """
    Handles text preprocessing and normalization tasks.
    """
    
    def __init__(self):
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words("english"))

    def normalize_text(self, text: str, min_word_length: int = 3) -> List[str]:
        """
        Applies full text normalization pipeline:
        1. Lowercase conversion
        2. Number removal
        3. Punctuation removal
        4. Stopword removal
        5. Short word removal
        6. Stemming
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove numbers
        text = re.sub(r'\d+', '', text)
        
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Remove extra whitespace
        text = " ".join(text.split())
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and short words
        tokens = [word for word in tokens 
                 if word not in self.stop_words and len(word) >= min_word_length]
        
        # Apply stemming
        tokens = [self.stemmer.stem(word) for word in tokens]
        
        return tokens

class TopicModeler:
    """
    Handles LDA topic modeling and visualization.
    """
    
    def __init__(self, num_topics: int = 10):
        self.num_topics = num_topics
        self.model = None
        self.corpus = None
        self.id2word = None

    def prepare_data(self, texts: List[List[str]]):
        """Prepares text data for topic modeling."""
        # Create Dictionary
        self.id2word = corpora.Dictionary(texts)
        
        # Create Corpus
        self.corpus = [self.id2word.doc2bow(text) for text in texts]

    def train_model(self):
        """Trains the LDA model."""
        self.model = gensim.models.ldamodel.LdaModel(
            corpus=self.corpus,
            id2word=self.id2word,
            num_topics=self.num_topics,
            random_state=100,
            update_every=1,
            chunksize=100,
            passes=10,
            alpha='auto',
            per_word_topics=True
        )

    def visualize(self) -> pyLDAvis._prepare.PreparedData:
        """Creates interactive visualization of topic model."""
        return pyLDAvis.gensim.prepare(
            self.model, self.corpus, self.id2word, 
            sort_topics=False
        )

def main():
    # Initialize components
    collector = NYTArticleCollector(api_key="YOUR_API_KEY")
    preprocessor = TextPreprocessor()
    topic_modeler = TopicModeler(num_topics=10)
    
    # Collect article metadata
    df = collector.collect_metadata(
        query="music",
        begin_date="20220101",
        end_date="20220930"
    )
    
    # Scrape full article texts
    print("Scraping full article texts...")
    df['main_text'] = df['web_url'].apply(collector.scrape_article_text)
    
    # Preprocess texts
    print("Preprocessing texts...")
    processed_texts = df['main_text'].apply(preprocessor.normalize_text).tolist()
    
    # Train topic model
    print("Training topic model...")
    topic_modeler.prepare_data(processed_texts)
    topic_modeler.train_model()
    
    # Generate visualization
    print("Generating visualization...")
    vis = topic_modeler.visualize()
    
    # Save results
    df.to_csv('nyt_articles.csv', index=False)
    pyLDAvis.save_html(vis, 'topic_visualization.html')
    
    print("Analysis complete! Results saved to nyt_articles.csv and topic_visualization.html")

if __name__ == "__main__":
    main()
