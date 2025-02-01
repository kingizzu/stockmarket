import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import pytz
import os
import csv
import urllib.parse
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import torch.nn as nn

# Streamlit App Title
st.title("Stock Sentiment Analysis & Prediction")

# Create a form for user input
with st.form(key="stock_form"):
    st.subheader("Enter Stock Information")
    
    # User inputs for stock name and code
    news_search = st.text_input("Stock Name (e.g., KLCI, Apple, Tesla)", value="KLCI")
    stock_code = st.text_input("Stock Code (e.g., ^KLSE, AAPL, TSLA)", value="^KLSE")
    
    # Submit button
    submit_button = st.form_submit_button(label="Submit")

# Process user input after submission
if submit_button:
    st.success(f"You have selected: **{news_search}** with stock code **{stock_code}**")
    
    # You can now pass `news_search` and `stock_code` into your scraping and prediction functions.
    st.write("Proceeding with data scraping and sentiment analysis, it may takes more than 5 minutes...")

    #############################################Scraping##################################################
    duration_days = 10
    today = datetime.today().strftime("%Y-%m-%d")
    five_days_ago = (datetime.today() - timedelta(days=duration_days)).strftime("%Y-%m-%d")
    
    def get_search_url():
        stock_name = news_search
        encoded_stock = urllib.parse.quote(stock_name)
        base_url = f"https://theedgemalaysia.com/news-search-results?keywords={encoded_stock}&to={today}&from={five_days_ago}&language=english&offset="
        return base_url, stock_name
    
    base_url, stock_name = get_search_url()
    csv_file = "scraped_articles.csv"
    
    def initialize_csv(file_path):
        if not os.path.exists(file_path):
            with open(file_path, mode="w", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                writer.writerow(["Date", "title", "detail", "combined_text"])
    
    def scrape_page(page_number):
        url = f"{base_url}{page_number}"
        print(f"Scraping page: {url}")
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, "html.parser")
            articles = soup.find_all("div", class_="col-md-8 col-12")
            scraped_data = []
            for article in articles:
                headline = article.find("span", class_="NewsList_newsListItemHead__dg7eK")
                headline_text = headline.get_text(strip=True).replace(":", ".") if headline else "No headline found"
                description = article.find("span", class_="NewsList_newsList__2fXyv")
                description_text = description.get_text(strip=True).replace(":", ".") if description else "No description found"
                combined_text = f"{headline_text} - {description_text}"
                timestamp_container = article.find_next("div", class_="NewsList_infoNewsListSub__Ui2_Z")
                timestamp_text = timestamp_container.get_text(strip=True) if timestamp_container else "No timestamp found"
                if timestamp_text != "No timestamp found":
                    cleaned_timestamp_text = timestamp_text.split("(")[0].strip()
                    try:
                        datetime_obj = datetime.strptime(cleaned_timestamp_text, "%d %b %Y, %I:%M %p")
                        timezone = pytz.timezone("Asia/Kuala_Lumpur")
                        localized_datetime = timezone.localize(datetime_obj)
                        iso_timestamp = localized_datetime.isoformat()
                    except ValueError:
                        iso_timestamp = "Invalid timestamp format"
                else:
                    iso_timestamp = "No timestamp available"
                scraped_data.append([iso_timestamp, headline_text, description_text, combined_text])
            with open(csv_file, mode="a", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                writer.writerows(scraped_data)
            print(f"Saved {len(scraped_data)} articles to CSV.")
            print("-" * 50)
        else:
            print(f"Failed to retrieve page {page_number}. Status code: {response.status_code}")
    
    initialize_csv(csv_file)
    for page in range(0, 50, 10):
        scrape_page(page)

    #############################Sentiment Analysis################################
    file_path = 'scraped_articles.csv'
    data = pd.read_csv(file_path)
    model_path = "StephanAkkerman/FinTwitBERT-sentiment"  
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)

    def analyze_sentiment(text):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=256)
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=1)
        positive = probabilities[0][1].item()
        neutral = probabilities[0][0].item()
        negative = probabilities[0][2].item()
        score = positive - negative
        return positive, neutral, negative, score
        
    data['positive'], data['neutral'], data['negative'], data['score'] = zip(*data['combined_text'].apply(analyze_sentiment))
    data['entry_count'] = 1
    sentiment_data_path = "sentiment_analysis_results.csv"
    data.to_csv(sentiment_data_path, index=False)

    st.title("📰 Latest 5 News Articles with Sentiment Scores")
    csv_file_path = "sentiment_analysis_results.csv"
    try:
        df = pd.read_csv(csv_file_path)
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.drop_duplicates(subset=['title', 'detail'], keep='first')
        latest_news = df.sort_values(by='Date', ascending=False).head(5)
        for i, row in latest_news.iterrows():
            st.markdown(f"### {i+1}. {row['title']}")
            st.write(f"📅 **Date:** {row['Date'].strftime('%Y-%m-%d %H:%M:%S')}")
            st.write(f"📰 **Summary:** {row['detail']}")
            st.write(f"👍 **Positive:** `{row['positive']:.5f}` | 😐 **Neutral:** `{row['neutral']:.5f}` | 👎 **Negative:** `{row['negative']:.5f}`")
            st.write(f"🧮 **Overall Sentiment Score:** {row['score']:.5f}")
            st.write("---")
    except Exception as e:
        st.error(f"❌ Error loading CSV: {e}")

    file_path = "sentiment_analysis_results.csv" 
    df = pd.read_csv(file_path)
    columns_to_remove = ['title', 'detail', 'combined_text']
    df = df.drop(columns=[col for col in columns_to_remove if col in df.columns], errors='ignore')
    df['Date'] = pd.to_datetime(df['Date']).dt.date
    aggregated_df = df.groupby('Date', as_index=False).mean()
    processed_sentiment_analysis = "processed_sentiment_analysis.csv"
    aggregated_df.to_csv(processed_sentiment_analysis, index=False)

####################Get the stock data################################
    try:
        sentiment_data = pd.read_csv(processed_sentiment_analysis)
        print("Sentiment analysis data loaded successfully.")
    except Exception as e:
        print(f"Error loading sentiment analysis data: {e}")
    
    try:
        stock_data = yf.Ticker(stock_code)
        stock_history = stock_data.history(period="1mo")
        if stock_history.empty:
            print(f"No stock data found for stock code: {stock_code}")
        else:
            stock_history.reset_index(inplace=True)
            st.subheader(f"📊 Stock Price Chart for {stock_code}")
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(stock_history["Date"], stock_history["Close"], label="Closing Price", linewidth=2)
            ax.set_xlabel("Date")
            ax.set_ylabel("Stock Price")
            ax.set_title(f"{stock_code} Stock Market Trend")
            ax.legend()
            ax.grid()
            st.pyplot(fig)
            st.subheader("📃 Latest Stock Data")
            st.dataframe(stock_history.tail(5))
            st.write("---")
            stock_history['Date'] = stock_history['Date'].dt.date
            print("Stock market data fetched successfully.")
    except Exception as e:
        print(f"Error fetching stock market data: {e}")
    
    try:
        if 'Date' in sentiment_data.columns:
            sentiment_data['Date'] = pd.to_datetime(sentiment_data['Date']).dt.date
        combined_data = pd.merge(sentiment_data, stock_history, on='Date', how='inner')
        combined_data = combined_data.drop(columns=['entry_count', 'High', 'Low', 'Volume', 'Dividends', 'Stock Splits'])
        print("Data combined successfully.")
        combined_data_path = f"combined_data.csv"
        combined_data.to_csv(combined_data_path, index=False)
        print(f"Combined data saved to {combined_data_path}.")
    except Exception as e:
        print(f"Error combining data: {e}")

######################################Stock Price Prediction#########################################
    class TransformerModel(nn.Module):
        def __init__(self, input_dim, embed_dim, num_heads, num_layers, dropout):
            super(TransformerModel, self).__init__()
            self.embedding = nn.Linear(input_dim, embed_dim)
            self.positional_encoding = nn.Parameter(torch.zeros(1, sequence_length, embed_dim))
            self.transformer = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(embed_dim, num_heads, dropout=dropout),
                num_layers
            )
            self.fc = nn.Linear(embed_dim, 1)
    
        def forward(self, x):
            x = self.embedding(x) + self.positional_encoding
            x = self.transformer(x)
            x = x.mean(dim=1)
            return self.fc(x)
    
    input_dim = 5
    embed_dim = 64
    num_heads = 4
    num_layers = 2
    dropout = 0.1
    sequence_length = 3
    
    model = TransformerModel(input_dim, embed_dim, num_heads, num_layers, dropout)
    model.load_state_dict(torch.load('transformer_model.pth'))
    model.eval()
    
    new_data = pd.read_csv('combined_data.csv')
    features = ['positive', 'neutral', 'negative', 'score', 'Open']
    target = 'Close'
    
    scaler = MinMaxScaler()
    new_data[features + [target]] = scaler.fit_transform(new_data[features + [target]])
    
    def create_input_sequence(data, features, sequence_length):
        sequences = []
        for i in range(len(data) - sequence_length):
            seq = data[features].iloc[i:i + sequence_length].values
            sequences.append(seq)
        return torch.tensor(sequences, dtype=torch.float32)
    
    input_sequences = create_input_sequence(new_data, features, sequence_length)
    with torch.no_grad():
        predictions = model(input_sequences)
    
    predicted_prices = predictions.numpy().flatten()
    predicted_prices_2d = [[0, 0, 0, 0, 0, pred] for pred in predicted_prices]
    predicted_prices_real = scaler.inverse_transform(predicted_prices_2d)[:, -1]
    
    st.subheader(f"💹Predicted Next Closing Price: RM{predicted_prices_real[-1]:.2f}")
    
    plt.plot(new_data['Close'][sequence_length:].values, label="Actual")
    plt.plot(predicted_prices, label="Predicted")
    plt.legend()
    plt.show()
