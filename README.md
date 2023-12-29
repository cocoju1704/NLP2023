# YouTube Sentiment Analysis & Aggregation Tool

This is a term-project for NLP2023 class(CSE4057) in Hanyang University.
This project provides a web application for sentiment analysis & aggregation result of YouTube comments.
On the input of a Youtube video link, it labels the sentimenets of each comment by using our own pretrained BERT model.
This model was pretrained by the datasets colleced on Youtube comments of news videos, which then was labeled by ChatGPT.(a.k.a ChatGPT labeling)
It then aggregates the analysis of each comment and present it to the user by visualizing the result into a bar plot.📊

## Purpose 
- To check the performance of ChatGPT on labeling unlabeled dataset.
- To test various prompt engineering skills on improving the performance of ChatGPT labeling.
- To experience the overall framework of Sentiment Analysis from dataset creation, model finetuning to validation.


## Features 

- Extracts the comments from a Youtube video URL and save them into an unlabeled CSV file.
- Retrieves comments from the specified YouTube video and saves them to a CSV file.
- Performs sentiment analysis on the comments using a pretrained BERT model.
- Generates bar charts and scatter plots to visualize the sentiment analysis results.
- Provides an interactive web interface using Streamlit.

## Installation 🛠️

1. Clone the repository:

2. Install the required dependencies:

3. Download the pretrained BERT Model from this link and locate it inside the project folder.
https://drive.google.com/file/d/1T8KXj2RLm3sHzaz20Au4JhgICFAGmpYH/view?usp=drive_link

4. Obtain a YouTube Data API key from the [Google Cloud Console](https://console.cloud.google.com/) and replace `YOUR_OWN_API_KEY` in "/.streamlit/secrets.toml" with your actual API key.

5. Run the application via Streamlit


## Usage 🚀

1. Open the application in your web browser.

2. Copy & paste a Youtube video URL into the sidebar.

3. Wait for the application to retrieve the video and channel information, save the comments to a CSV file, perform sentiment analysis, and display the results. (This might take a while if the size of the comments is large)

4. View the aggregated result of comments classified into 5 labels:["anger", "sadness", "neutral", "joy", "surprise"]

## User modifications

### Changing Models
- To use your own finetuned BERT model, change the model_path("./model_A") parameter of bar_chart(csv_file, "./model_A") in app.py to your own path. Further modifications might be required if you wish to use another set of labels due to visualization.
### Changing Aggregation Method
- To use another method on aggregating the results, change the for loop in analyze_sentiment() in SA_BERT.py


## Contribution

The web interface was implemented based on this project.
https://github.com/JatinAgrawal0/youtube-comment-sentimental-analysis




