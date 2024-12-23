# Stock-Predictions-with-Sentiment-Model
My attempt at training a Roberta sentiment model on quarterly reports to predict resulting stock changes.

I used the Roberta-base model from Hugging-Face's Library

I then used domain adaptation on the base model using the attached business_sentiment.csv I found online through Kaggle

To train the model, I first wrote a web-scraping script in python using getURLText.py to efficiently transfer the quarterly reports text into txt files stored on the primary csv file

To obtain stock data, I used the yfinance api through Stock_Changes.py, allowing me to quickly access data and then compare to the S&P 500 over that same time.

I am still making improvements to the model training as the outcome is currently lacking, so look for changes in the future.
