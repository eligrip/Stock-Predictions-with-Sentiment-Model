import requests
import time
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from requests.exceptions import RequestException

# Load the CSV file
file_path = 'your_csv_path'
financial_reports = pd.read_csv(file_path)
financial_reports['Date'] = pd.to_datetime(financial_reports['Date'])

def get_stock_price_change(ticker, report_date):
    """
    Fetches stock price change adjusted for SPY index from Yahoo Finance.
    """
    report_date = pd.to_datetime(report_date)
    one_month_after = report_date + timedelta(days=33)
    
    try:
        stock_data = yf.download(ticker, start=report_date+ timedelta(days=3), end=one_month_after)
        time.sleep(1)
        spy_data = yf.download('SPY', start=report_date + timedelta(days=3), end=one_month_after)
        time.sleep(0.5)
        if stock_data.empty:
            print(f"No data available for {ticker} between {report_date} and {one_month_after}")
            return None
        else:
            start_price = stock_data['Close'].iloc[0]
            end_price = stock_data['Close'].iloc[-1]
            start_price_spy = spy_data['Close'].iloc[0]
            end_price_spy = spy_data['Close'].iloc[-1]
            percent_change = float(((end_price - start_price) / start_price) * 100)
            percent_change_spy = float(((end_price_spy - start_price_spy) / start_price_spy) * 100)
            print(percent_change)
            print(percent_change_spy)
            return float(percent_change-percent_change_spy)
    except Exception as e:
        print(f"Error fetching stock data for {ticker}: {e}")
        return None
    
def retrieve_just_price(ticker, ticker_info):
    '''
    Isolate's ticker from Company Information
    '''
    split_info = ticker_info.split(ticker)[1]
    return split_info.split("Name")[0]

#Calls primary function
financial_reports["Stock Price Change"] = financial_reports.apply(
    lambda row: get_stock_price_change(row["Ticker"], row["Date"]), axis=1
)
# Save the updated DataFrame
path = "your_path"
financial_reports.to_csv(path, index = False)
