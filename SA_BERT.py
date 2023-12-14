import csv
import re
import pandas as pd
import nltk

nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import plotly.express as px
import plotly.graph_objects as go
from colorama import Fore, Style
from typing import Dict
import streamlit as st

# switch to BERT

import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.nn.functional import softmax
import pandas as pd
import numpy as np
sentiments_list = ['anger', 'sadness', 'neutral', 'joy', 'admiration']

def extract_video_id(youtube_link):
    video_id_regex = r"^(?:https?:\/\/)?(?:www\.)?(?:youtube\.com\/watch\?v=|youtu.be\/)([a-zA-Z0-9_-]{11})"
    match = re.search(video_id_regex, youtube_link)
    if match:
        video_id = match.group(1)
        return video_id
    else:
        return None


def analyze_sentiment(csv_file):
    # youtube comments csv -> dataframe
    raw_df = pd.read_csv(csv_file)
    #set model path?
    model_path = "./best_model_second"
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(model_path)
    sentiments_count = [0, 0, 0, 0, 0]
    for i in range(len(raw_df)):
        encoded = tokenizer(raw_df['Comment'][i], padding="max_length", truncation=True, max_length=200,
                            return_tensors='pt')
        output = model(**encoded)
        logits = output.logits
        pred = torch.argmax(softmax(logits, dim=1)).item()
        sentiments_count[pred] += 1

    #print("pred", end=" ")
    #for i in range(5):
    #    print(sentiments_list[i], end=" ")
    #print()

    # sentiment 합
    print("sentiment result")
    results = dict(zip(sentiments_list, sentiments_count))
    return results


def bar_chart(csv_file: str) -> None:
    # Call analyze_sentiment function to get the results
    results: Dict[str, int] = analyze_sentiment(csv_file)

    # Get the counts for each sentiment category
    num_anger = results['anger']
    num_sadness = results['sadness']
    num_neutral = results['neutral']
    num_joy = results['joy']
    num_admiration = results['admiration']

    # Create a Pandas DataFrame with the results
    df = pd.DataFrame({
        'Sentiment': ['anger', 'sadness', 'neutral', 'joy', 'admiration'],
        'Number of Comments': [num_anger, num_sadness, num_neutral, num_joy, num_admiration]
    })

    # Create the bar chart using Plotly Express
    fig = px.bar(df, x='Sentiment', y='Number of Comments', color='Sentiment',
                 color_discrete_sequence=['#FFA07A', '#87CEFA', '#D3D3D3', '#FFD966', '#B1D599'],
                 title='Sentiment Analysis Results')
    fig.update_layout(title_font=dict(size=20))

    # Show the chart
    st.plotly_chart(fig, use_container_width=True)


def plot_sentiment(csv_file: str) -> None:
    # Call analyze_sentiment function to get the results
    results: Dict[str, int] = analyze_sentiment(csv_file)

    # Get the counts for each sentiment category
    num_anger = results['anger']
    num_sadness = results['sadness']
    num_neutral = results['neutral']
    num_joy = results['joy']
    num_admiration = results['admiration']

    # Plot the pie chart
    labels = ['anger', 'sadness', 'neutral', 'joy', 'admiration'],
    values = [num_anger, num_sadness, num_neutral, num_joy, num_admiration]
    colors = ['red', 'blue', 'grey', 'yellow', 'green']
    fig = go.Figure(data=[go.Pie(labels=labels, values=values, textinfo='label+percent',
                                 marker=dict(colors=colors))])
    fig.update_layout(
        title={'text': 'Sentiment Analysis Results', 'font': {'size': 20, 'family': 'Arial', 'color': 'grey'},
               'x': 0.5, 'y': 0.9},
        font=dict(size=14))
    st.plotly_chart(fig)


def create_scatterplot(csv_file: str, x_column: str, y_column: str) -> None:
    # Load data from CSV
    data = pd.read_csv(csv_file)

    # Create scatter plot using Plotly
    fig = px.scatter(data, x=x_column, y=y_column, color='Category')

    # Customize layout
    fig.update_layout(
        title='Scatter Plot',
        xaxis_title=x_column,
        yaxis_title=y_column,
        font=dict(size=18)
    )

    # Display plot in Streamlit
    st.plotly_chart(fig, use_container_width=True)


def print_sentiment(csv_file: str) -> None:
    # Call analyze_sentiment function to get the results
    results: Dict[str, int] = analyze_sentiment(csv_file)

    # Get the counts for each sentiment category
    num_anger = results['anger']
    num_sadness = results['sadness']
    num_neutral = results['neutral']
    num_joy = results['joy']
    num_admiration = results['admiration']

    # Determine the overall sentiment
    #if num_positive > num_negative:
    overall_sentiment = 'POSITIVE'
    color = Fore.GREEN
    #elif num_negative > num_positive:
        #overall_sentiment = 'NEGATIVE'
        #color = Fore.RED
    #else:
        #overall_sentiment = 'NEUTRAL'
        #color = Fore.YELLOW

    # Print the overall sentiment in color
    print('\n' + Style.BRIGHT + color + overall_sentiment.upper().center(50, ' ') + Style.RESET_ALL)



