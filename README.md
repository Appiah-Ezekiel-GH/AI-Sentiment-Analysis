# AI Sentiment Analysis on Reddit, YouTube, and Twitter

## Overview
This repository contains a comprehensive analysis of sentiment towards Artificial Intelligence (AI) as expressed in comments from various subreddits, YouTube videos, and Twitter conversations. The analysis explores public perception, emotional responses, and discourse patterns related to AI technologies across different online communities. The project employs advanced natural language processing techniques and machine learning models to extract insights from large volumes of text data.

## Project Objectives
- Analyze sentiment distribution towards AI across different subreddit communities, YouTube videos, and Twitter conversations
- Identify key topics and themes in AI discussions
- Track temporal patterns and trends in sentiment over time
- Explore emotional responses associated with AI-related discourse
- Visualize relationships between concepts and opinions in the data
- Compare sentiment across different AI-related technologies and applications

## Data Collection
- **Source**: Comments from multiple sources including:
  - **Reddit**:
    - r/ArtificialInteligence (5,634 comments)
    - r/ChatGPT (14,000+ comments)
    - r/technology
    - r/Futurology
    - r/science
    - r/MachineLearning
  - **YouTube**:
    - Comments from AI-related videos
    - Video transcripts from technology channels
    - User engagement metrics
  - **Twitter/X**:
    - Tweets containing AI-related hashtags
    - Public conversations about artificial intelligence
    - Engagement metrics (likes, retweets, replies)
- **Method**: 
  - Reddit: Utilizes the PRAW (Python Reddit API Wrapper) library to fetch comments with exponential backoff for handling rate limits
  - YouTube: YouTube Data API for retrieving comments and video metadata
  - Twitter/X: Twitter API v2 for collecting tweets and conversation threads
- **Time Period**: Data collected spans several months to capture temporal patterns across all platforms
- **Filtering**: Content filtered using keywords such as 'AI', 'Artificial Intelligence' to ensure relevance
- **Data Structure**: Stored in pandas DataFrames with platform-specific columns and unified sentiment analysis fields

## Data Processing Pipeline
1. **Data Cleaning**:
   - Removal of URLs, HTML tags, and special characters
   - Conversion to lowercase for consistency
   - Removal of duplicates and irrelevant content (e.g., AutoModerator posts)
   - Handling of missing values

2. **Text Preprocessing**:
   - **Tokenization**: Breaking text into individual tokens using NLTK's word_tokenize
   - **Stop Word Removal**: Filtering common words that don't contribute to meaning
   - **Lemmatization**: Reducing words to their base form using WordNetLemmatizer
   - **Special Character Handling**: Removing or normalizing punctuation and symbols

3. **Feature Engineering**:
   - Timestamp conversion for temporal analysis
   - Comment length calculation
   - Extraction of mentions of specific AI technologies

## Sentiment Analysis Methodology
1. **Model-Based Approaches**:
   - **RoBERTa Model**: Pre-trained transformer model fine-tuned for sentiment classification
     - Implementation uses the Hugging Face transformers library
     - Classifies text into positive, neutral, and negative categories
     - Provides probability scores for each sentiment class

2. **Lexicon-Based Approaches**:
   - **NRCLex**: Emotion analysis based on the NRC Emotion Lexicon
     - Categorizes text into 8 emotions: anger, fear, anticipation, trust, surprise, sadness, joy, and disgust
     - Provides intensity scores for each emotion category
   - **Opinion Lexicon**: Dictionary-based approach counting positive and negative words
     - Uses NLTK's opinion_lexicon for identifying sentiment-bearing words

3. **Metrics and Evaluation**:
   - Sentiment scores normalized to facilitate comparison across subreddits and platforms
   - Validation of sentiment classification using manual samples
   - Comparison of results between different sentiment analysis methods

## Visualization & Analysis Techniques
1. **Text Visualization**:
   - **Word Clouds**: Generated using WordCloud library to visualize frequent terms in comments
     - Separate visualizations for positive, negative, and neutral comments
     - Size of words proportional to frequency in the corpus
   - **Network Graphs**: Created with NetworkX to show relationships between terms
     - Edges represent co-occurrence or semantic relationships
     - Node size indicates term frequency or importance

2. **Statistical Visualization**:
   - **Time Series Analysis**: Plotting sentiment trends using matplotlib and seaborn
     - Daily/weekly aggregations to identify patterns
     - Moving averages to smooth out noise
   - **Distribution Plots**: Visualizing sentiment and emotion distributions
     - Histograms and kernel density estimates
     - Box plots for comparison across subreddits and platforms

3. **Dimensionality Reduction**:
   - **Topic Modeling**: Latent Dirichlet Allocation (LDA) for discovering underlying topics
     - Optimal topic number determined through coherence scores
     - Topics visualized through word distributions
   - **UMAP & t-SNE**: For visualizing high-dimensional text data in 2D space
     - Comment clustering based on semantic similarity
     - Interactive visualizations using plotly

4. **Cross-Platform Analysis**:
   - **Sentiment Comparison**: Comparing AI sentiment across Reddit, YouTube, and Twitter
   - **Platform-Specific Discourse**: Identifying unique characteristics of AI discussions on each platform
   - **Temporal Correlation**: Analyzing how sentiment shifts across platforms following major AI events

## Key Findings
1. **Sentiment Distribution**:
   - Overall sentiment distribution across the analyzed subreddits, YouTube videos, and Twitter conversations
   - Comparison of sentiment between technical and non-technical communities
   - Identification of outlier communities with extremely positive or negative sentiment

2. **Temporal Patterns**:
   - Evolution of sentiment over time, including reactions to major AI news events
   - Weekly and daily patterns in discussion volume and sentiment
   - Trend analysis showing changing perceptions of AI technologies

3. **Topic Analysis**:
   - Dominant topics in AI discussions across different communities
   - Correlation between topics and sentiment polarity
   - Emerging themes and concerns within the AI discourse

4. **Emotional Analysis**:
   - Primary emotions associated with AI discussions (fear, hope, trust, etc.)
   - Variation in emotional responses across different AI applications
   - Relationship between emotional intensity and comment engagement

5. **Network Analysis**:
   - Key concept relationships and term associations in AI discussions
   - Central themes that bridge different aspect of AI discourse
   - Community-specific terminology and focal points

6. **Platform Comparison**:
   - Differences in sentiment expression between Reddit, YouTube, and Twitter users
   - Platform-specific jargon and terminology when discussing AI technologies
   - Variation in response to AI news events across different social media platforms

## Project Structure
```
├── Main_Analysis_Document.ipynb   # Main analysis notebook
├── README.md                      # Project documentation
├── data/                          # Data directory
│   ├── SR_AI_df.csv               # ArtificialInteligence subreddit data
│   ├── SR_GPT_df.csv              # ChatGPT subreddit data
│   ├── youtube_comments.csv       # YouTube comments data
│   ├── twitter_data.csv           # Twitter/X data
│   └── ...                        # Other data files
├── visualizations/                # Generated plots and visualizations
└── requirements.txt               # Package dependencies
```

## Usage
To replicate this analysis:

1. **Environment Setup**:
   ```
   # Clone the repository
   git clone https://github.com/yourusername/ai-sentiment-analysis.git
   cd ai-sentiment-analysis

   # Create and activate virtual environment (optional)
   python -m venv env
   source env/bin/activate  # On Windows: env\Scripts\activate

   # Install dependencies
   pip install -r requirements.txt
   ```

2. **Data Collection** (optional, as CSV files are provided):
   - Set up a Reddit developer account and obtain API credentials
   - Update the `CLIENT_ID`, `CLIENT_SECRET`, and `USER_AGENT` variables in the notebook
   - Run the data collection cells to fetch new data
   - Set up a YouTube Data API project and obtain API credentials
   - Update the `YOUTUBE_API_KEY` variable in the notebook
   - Run the YouTube data collection cells to fetch new data
   - Set up a Twitter Developer account and obtain API credentials
   - Update the `TWITTER_API_KEY`, `TWITTER_API_SECRET`, `TWITTER_ACCESS_TOKEN`, and `TWITTER_ACCESS_TOKEN_SECRET` variables in the notebook
   - Run the Twitter data collection cells to fetch new data

3. **Run the Analysis**:
   - Open `Main_Analysis_Document.ipynb` in Jupyter Notebook or JupyterLab
   - Run the cells sequentially to reproduce the analysis
   - Modify parameters as needed to explore different aspects of the data

4. **Customization**:
   - Modify the list of subreddits, YouTube channels, or Twitter hashtags to analyze different communities
   - Adjust the sentiment analysis parameters for different thresholds
   - Experiment with alternative visualization techniques

## Dependencies
- **Python 3.x**
- **Data Processing**: pandas, numpy
- **API Interaction**: 
  - praw (Reddit)
  - google-api-python-client (YouTube)
  - tweepy (Twitter/X)
- **NLP & Text Processing**: 
  - nltk
  - transformers
  - wordcloud
  - scikit-learn
- **Visualization**: 
  - matplotlib
  - seaborn
  - plotly
  - networkx
- **Machine Learning**:
  - torch
  - umap-learn
  - scikit-learn
- **Emotion Analysis**: nrclex

## Note
- Please be aware of Reddit's API usage policies and rate limits when running the data collection scripts.
- Some visualizations may require significant computational resources depending on the volume of data.
- The sentiment analysis results should be interpreted with consideration for the limitations of automated sentiment classification methods.
- For privacy reasons, user identifiers have been anonymized in accordance with ethical research practices.

## Acknowledgements
- Reddit API for providing access to comment data
- YouTube Data API for providing access to video metadata
- Twitter API for providing access to tweet data
- Hugging Face for pre-trained transformer models
- NLTK project for lexicons and NLP tools
- The various open-source Python libraries that made this analysis possible

## Future Work
- Expand analysis to include more specialized AI-focused communities
- Implement more advanced sentiment analysis techniques using multi-modal approaches
- Create an interactive dashboard for exploring the results
- Conduct deeper cross-platform analysis to identify how AI discussions vary across social media ecosystems
- Apply more sophisticated topic modeling techniques like BERTopic
