
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import spacy
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import nltk

# Download necessary nltk packages
nltk.download('vader_lexicon')

# Load spaCy
nlp = spacy.load("en_core_web_sm")

# Load VADER sentiment analyzer
sia = SentimentIntensityAnalyzer()

st.title('Movie Review Sentiment Analysis Dashboard')

# Sidebar for navigation
st.sidebar.title('Navigation')
pages = ['Overview', 'Sentiment Analysis', 'Aspect-Based Analysis', 'Opinion Mining']
choice = st.sidebar.radio('Go to', pages)

# Sample data loading - in a real app, use the actual data
@st.cache_data
def load_data():
    # This would be replaced with actual data loading
    df = pd.read_csv('IMDB Dataset.csv')
    # Add sentiment analysis using VADER
    df['sentiment_score'] = df['review'].apply(lambda x: sia.polarity_scores(x)['compound'])
    df['sentiment'] = df['sentiment_score'].apply(lambda s: 'positive' if s >= 0.05 else ('negative' if s <= -0.05 else 'neutral'))
    return df

df = load_data()

# Overview Page
if choice == 'Overview':
    st.header('Dataset Overview')
    st.write(f'Total reviews: {len(df)}')

    # Display sentiment distribution
    sentiment_counts = df['sentiment'].value_counts()
    fig = px.pie(values=sentiment_counts.values, names=sentiment_counts.index, title='Sentiment Distribution')
    st.plotly_chart(fig)

    # Sample reviews
    st.subheader('Sample Reviews')
    sample_df = df.sample(5)
    for i, row in sample_df.iterrows():
        st.write(f"**Sentiment**: {row['sentiment']}")
        st.write(f"**Review**: {row['review'][:300]}...")
        st.write("---")

# Sentiment Analysis Page
elif choice == 'Sentiment Analysis':
    st.header('Sentiment Analysis')

    # Sentiment distribution over time (let's assume we have a date field)
    st.subheader('Word Clouds by Sentiment')

    sentiment_choice = st.selectbox('Choose sentiment to visualize', ['positive', 'negative', 'neutral'])

    # Generate wordcloud
    if sentiment_choice:
        reviews = df[df['sentiment'] == sentiment_choice]['review'].tolist()
        if reviews:
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(reviews[:100]))
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)

    # Interactive sentiment analysis
    st.subheader('Try Your Own Text')
    user_text = st.text_area('Enter text to analyze')
    if user_text:
        score = sia.polarity_scores(user_text)['compound']
        sentiment = 'positive' if score >= 0.05 else ('negative' if score <= -0.05 else 'neutral')
        st.write(f"Sentiment: **{sentiment}** (Score: {score:.2f})")

# Aspect-Based Analysis
elif choice == 'Aspect-Based Analysis':
    st.header('Aspect-Based Sentiment Analysis')

    # Common aspects in movie reviews
    common_aspects = ['acting', 'plot', 'script', 'effects', 'music', 'cinematography', 'direction', 'characters']

    # Function for aspect sentiment
    def get_aspect_sentiment(text, aspect):
        if aspect.lower() in text.lower():
            start = text.lower().find(aspect.lower())
            end = start + len(aspect)
            context_start = max(0, start - 50)
            context_end = min(len(text), end + 50)
            context = text[context_start:context_end]
            score = sia.polarity_scores(context)['compound']
            return score
        return None

    # Analyze aspects in sample reviews
    st.subheader('Aspect Sentiment in Reviews')

    # Create random aspect sentiment data for visualization
    aspect_sentiments = {}
    for aspect in common_aspects:
        pos = np.random.randint(10, 100)
        neg = np.random.randint(10, 100)
        neu = np.random.randint(5, 50)
        aspect_sentiments[aspect] = {'positive': pos, 'negative': neg, 'neutral': neu}

    # Plot
    aspects = list(aspect_sentiments.keys())
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=aspects,
        y=[aspect_sentiments[a]['positive'] for a in aspects],
        name='Positive',
        marker_color='green'
    ))

    fig.add_trace(go.Bar(
        x=aspects,
        y=[aspect_sentiments[a]['negative'] for a in aspects],
        name='Negative',
        marker_color='red'
    ))

    fig.add_trace(go.Bar(
        x=aspects,
        y=[aspect_sentiments[a]['neutral'] for a in aspects],
        name='Neutral',
        marker_color='gray'
    ))

    fig.update_layout(
        title='Aspect-Based Sentiment Distribution',
        xaxis_title='Aspects',
        yaxis_title='Count',
        barmode='group'
    )

    st.plotly_chart(fig)

    # Interactive aspect analysis
    st.subheader('Analyze Aspects in Your Text')
    user_text = st.text_area('Enter text to analyze aspects')
    if user_text:
        results = []
        for aspect in common_aspects:
            score = get_aspect_sentiment(user_text, aspect)
            if score is not None:
                sentiment = 'positive' if score >= 0.05 else ('negative' if score <= -0.05 else 'neutral')
                results.append({'Aspect': aspect, 'Sentiment': sentiment, 'Score': score})

        if results:
            st.table(pd.DataFrame(results))
        else:
            st.write("No common aspects found in the text.")

# Opinion Mining
elif choice == 'Opinion Mining':
    st.header('Opinion Mining')

    # Opinion extraction function
    def extract_opinions(text):
        doc = nlp(text)
        opinions = []

        # Extract opinion phrases (adjective + noun combinations)
        for token in doc:
            if token.pos_ == "ADJ":  # Opinion words are often adjectives
                # Find the noun it's describing
                for child in token.children:
                    if child.dep_ == "nsubj" and child.pos_ == "NOUN":
                        opinions.append({"Opinion": token.text, "Target": child.text})

        return opinions

    # Interactive opinion mining
    st.subheader('Extract Opinions from Text')
    user_text = st.text_area('Enter text to extract opinions')
    if user_text:
        if len(user_text) > 1000:
            st.warning("Text is too long. Analyzing first 1000 characters only.")
            user_text = user_text[:1000]

        opinions = extract_opinions(user_text)

        if opinions:
            st.table(pd.DataFrame(opinions))
        else:
            st.write("No explicit opinions extracted. Try different text.")

    # Sample opinions visualization
    st.subheader('Common Opinion-Target Pairs in Reviews')

    # Mock data for visualization
    common_opinions = [
        {'Opinion': 'great', 'Target': 'movie', 'Count': 120},
        {'Opinion': 'boring', 'Target': 'plot', 'Count': 75},
        {'Opinion': 'excellent', 'Target': 'acting', 'Count': 95},
        {'Opinion': 'terrible', 'Target': 'script', 'Count': 60},
        {'Opinion': 'amazing', 'Target': 'effects', 'Count': 80}
    ]

    fig = px.bar(
        common_opinions, 
        x='Opinion', 
        y='Count', 
        color='Target',
        title='Common Opinion-Target Pairs'
    )
    st.plotly_chart(fig)
