from vnstock import *

import pandas as pd
import matplotlib.pyplot as plt
from datetime import date, datetime, timedelta
import numpy as np

def fm_date(date):
    return date.strftime("%Y-%m-%d")

# Get list of dates for the last 365 days
def get_last_days(num_days):
    current_date = datetime.now()
    days_ago = current_date - timedelta(days=num_days)
    date_list = [days_ago + timedelta(days=i) for i in range(num_days)]
    # Format the dates as "%Y-%m-%d" and store them in a new list
    formatted_dates = [fm_date(date) for date in date_list]

    return formatted_dates

def get_stock_price(symbol, num_days):
    dates_train = get_last_days(num_days)
    df_his = stock_historical_data(symbol, dates_train[0], dates_train[len(dates_train) - 1], '1D', 'stock')
    return df_his



def draw_ma(symbol, num_days):
    df_500 = get_stock_price(symbol, 500)
    df = get_stock_price(symbol, num_days)
    df['20_MA'] = df['close'].rolling(window=20).mean()
    df['50_MA'] = df['close'].rolling(window=50).mean()
    df_500['200_MA'] = df_500['close'].rolling(window=200).mean()

    # Split df_500 and get the last range_days rows
    df_500_tail = df_500.tail(num_days)
    df = df.merge(df_500_tail[['time', '200_MA']], on='time', how='inner')

    
    # Create traces for the closing prices and moving averages
    trace_20_ma = go.Scatter(x=df['time'], y=df['20_MA'], name='20-day MA')
    trace_50_ma = go.Scatter(x=df['time'], y=df['50_MA'], name='50-day MA')
    trace_200_ma = go.Scatter(x=df['time'], y=df['200_MA'], name='200-day MA')

        # Create the figure and add the traces
    fig = make_subplots(rows=2, cols=1)

    fig.add_trace(go.Candlestick(x=df['time'],
                                     open=df['open'],
                                     high=df['high'],
                                     low=df['low'],
                                     close=df['close'],
                                     name=symbol
                                    ),
                      row=1, col=1)
    # Add volume bar chart
    fig.add_trace(go.Bar(x=df['time'], y=df['volume'], marker=dict(color='blue'), name='Volume'),
                row=2, col=1)

    fig.add_traces([trace_20_ma, trace_50_ma, trace_200_ma])

    # Update the layout
    fig.update_layout(title='Stock Price with Moving Averages',
                      xaxis_title='Date',
                      yaxis_title='Price')

    fig.update_layout(xaxis_rangeslider_visible=False)

    # Show the figure
    fig.show()

def calculate_rsi(data, column_name='close', period=14):
    # Calculate price changes
    delta = data[column_name].diff(1)

    # Separate gains and losses
    gains = delta.where(delta > 0, 0)
    losses = -delta.where(delta < 0, 0)

    # Calculate average gains and losses over the specified period
    avg_gain = gains.rolling(window=period, min_periods=1).mean()
    avg_loss = losses.rolling(window=period, min_periods=1).mean()

    # Calculate relative strength (RS)
    rs = avg_gain / avg_loss

    # Calculate RSI
    rsi = 100 - (100 / (1 + rs))

    # Add RSI to the DataFrame
    data['RSI'] = rsi

    return data

# Visualize RSI
def visualize_rsi(data):
    data_rsi = calculate_rsi(data, 'close', 14)
    # Plot the RSI data
    plt.figure(figsize=(14,7))
    plt.plot(data_rsi.index, data_rsi['RSI'], label='RSI', color='blue')
    
    # Add overbought and oversold lines
    plt.axhline(0, linestyle='--', alpha=0.5, color='gray')
    plt.axhline(20, linestyle='--', alpha=0.5, color='red')
    plt.axhline(30, linestyle='--', alpha=0.5, color='orange')
    plt.axhline(70, linestyle='--', alpha=0.5, color='orange')
    plt.axhline(80, linestyle='--', alpha=0.5, color='red')
    plt.axhline(100, linestyle='--', alpha=0.5, color='gray')
    
    # Add title and labels
    plt.title('RSI Plot')
    plt.xlabel('Date')
    plt.ylabel('RSI')
    
    # Show the plot
    plt.show()

def calculate_macd(data, short_window=12, long_window=26, signal_window=9):
    # Calculate Short-term Exponential Moving Average (EMA)
    short_ema = data['close'].ewm(span=short_window, adjust=False).mean()

    # Calculate Long-term Exponential Moving Average (EMA)
    long_ema = data['close'].ewm(span=long_window, adjust=False).mean()

    # Calculate MACD Line
    data['MACD'] = short_ema - long_ema

    # Calculate Signal Line
    data['Signal_Line'] = data['MACD'].ewm(span=signal_window, adjust=False).mean()

    # Calculate MACD Histogram
    data['MACD_Histogram'] = data['MACD'] - data['Signal_Line']

    return data


import plotly.graph_objects as go
from plotly.subplots import make_subplots

def draw_macd(df):
    df_macd = calculate_macd(df)
    # Create a subplot, and add a scatter plot for 'MACD Line' and 'Signal Line'
    fig = make_subplots(rows=2, cols=1)

    fig.add_trace(go.Candlestick(x=df_macd['time'],
                                     open=df_macd['open'],
                                     high=df_macd['high'],
                                     low=df_macd['low'],
                                     close=df_macd['close'],
                                     name="Stock price"
                                    ),
                      row=1, col=1)

    fig.add_trace(go.Scatter(x=df_macd['time'], y=df_macd['MACD'], mode='lines', name='MACD Line'), row=2, col=1)
    fig.add_trace(go.Scatter(x=df_macd['time'], y=df_macd['Signal_Line'], mode='lines', name='Signal Line'), row=2, col=1)

    # Add a bar plot for 'MACD Histogram'
    fig.add_trace(go.Bar(x=df_macd['time'], y=df_macd['MACD_Histogram'], name='MACD Histogram', marker_color='gray'), row=2, col=1)

    # Set plot title
    fig.update_layout(title_text='MACD and Signal Line')

    # Set x and y axis titles
    fig.update_xaxes(title_text='Date')
    fig.update_yaxes(title_text='MACD Value')

    fig.update_layout(xaxis_rangeslider_visible=False)
    # Show the plot
    fig.show()


def get_price_multi_ticker(tickers, num_days):
    df = pd.DataFrame()
    for ticker in tickers:
        df_ticker = get_stock_price(ticker, num_days)
        df_ticker['ticker'] = ticker
        df = df.append(df_ticker)

    return df

def draw_chart_muti_ticker(tickers, num_days):
    df = get_price_multi_ticker(tickers, num_days)

    # Create subplots
    fig = make_subplots(rows=2, cols=1)
    
    # Loop over tickers and add candlestick chart to subplot
    for i, ticker in enumerate(tickers, start=1):
        df_ticker = df[df['ticker'] == ticker]
        if i == 1:
            fig.add_trace(go.Candlestick(x=df_ticker['time'],
                                     open=df_ticker['open'],
                                     high=df_ticker['high'],
                                     low=df_ticker['low'],
                                     close=df_ticker['close'],
                                     name=ticker
                                    ),
                      row=1, col=1)
            # Add volume bar chart
            fig.add_trace(go.Bar(x=df_ticker['time'], y=df_ticker['volume'], marker=dict(color='blue'), name='Volume'),
                        row=2, col=1)
        else:
            fig.add_trace(go.Scatter(x=df_ticker['time'], y=df_ticker['close'], mode='lines',
                name=ticker),
              row=1, col=1)
    

    fig.update_layout(xaxis_rangeslider_visible=False)
    fig.show()

def bollinger_bands(df, window=20, num_std_dev=2):
    rolling_mean = df['close'].rolling(window=window).mean()
    rolling_std = df['close'].rolling(window=window).std()
    df['Bollinger High'] = rolling_mean + (rolling_std*num_std_dev)
    df['Bollinger Low'] = rolling_mean - (rolling_std*num_std_dev)
    return df



def bollinger_bands_plot(df, window=20, num_std_dev=2):
    bollinger_df = bollinger_bands(df, window=20, num_std_dev=2)
    # Create a candlestick chart
    fig = go.Figure(data=[go.Candlestick(x=bollinger_df.index,
                    open=bollinger_df['open'],
                    high=bollinger_df['high'],
                    low=bollinger_df['low'],
                    close=bollinger_df['close'])])

    # Add Bollinger Bands
    fig.add_trace(go.Scatter(x=df.index, y=bollinger_df['Bollinger High'], line=dict(color='gray'), name='Bollinger High'))
    fig.add_trace(go.Scatter(x=df.index, y=bollinger_df['Bollinger Low'], line=dict(color='gray'), name='Bollinger Low'))
    fig.show()


def calculate_stochastic(data, high_col='high', low_col='low', close_col='close', period=14, k_smooth=3, d_smooth=3):
    # Calculate %K
    data['%K'] = ((data[close_col] - data[low_col].rolling(window=period).min()) / 
                   (data[high_col].rolling(window=period).max() - data[low_col].rolling(window=period).min())) * 100

    # Smooth %K to get %D
    data['%D'] = data['%K'].rolling(window=k_smooth).mean()

    # Smooth %D to get a signal line
    data['%D_Signal'] = data['%D'].rolling(window=d_smooth).mean()

    return data

def draw_stochastic(df):
    df_sto = calculate_stochastic(df)
    # Plotting Stochastic Oscillator using Plotly
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=df_sto['time'], y=df_sto['%K'], mode='lines', name='%K'))
    fig.add_trace(go.Scatter(x=df_sto['time'], y=df_sto['%D'], mode='lines', name='%D'))
    fig.add_trace(go.Scatter(x=df_sto['time'], y=df_sto['%D_Signal'], mode='lines', name='%D Signal Line'))

    fig.update_layout(title='Stochastic Oscillator', xaxis_title='Date', yaxis_title='Percentage', legend=dict(x=0, y=1, traceorder='normal'))

    fig.show()

import plotly.express as px

def draw_obv(df):

    # Calculate daily price changes
    df['Price Change'] = df['close'].diff()

    # Initialize OBV with the first day's volume
    df['OBV'] = 0
    df.loc[df['Price Change'] > 0, 'OBV'] = df['volume']
    df.loc[df['Price Change'] < 0, 'OBV'] = -df['volume']

    # Cumulative sum of OBV
    df['OBV'] = df['OBV'].cumsum()

    # Create subplots
    fig = make_subplots(rows=2, cols=1)
    fig.add_trace(go.Candlestick(x=df['time'],
                                     open=df['open'],
                                     high=df['high'],
                                     low=df['low'],
                                     close=df['close'],
                                     name='price'
                                    ),
                      row=1, col=1)
    # Add volume bar chart
    # fig.add_trace(go.Bar(x=df['time'], y=df['volume'], marker=dict(color='blue'), name='Volume'),
    #             row=2, col=1)

    # Create an interactive plot using Plotly Express
    fig.add_trace(px.line(df, x=df.index, y='OBV', title='On-Balance Volume (OBV)',
                labels={'OBV': 'On-Balance Volume (OBV)', 'Date': 'Date'},
                line_shape='linear').data[0],
                row=2, col=1)

    fig.update_layout(
        xaxis_title='Date',
        yaxis_title='On-Balance Volume (OBV)',
        template='plotly_dark',  # You can choose different templates, e.g., 'plotly', 'ggplot2', 'seaborn', etc.
        xaxis_rangeslider_visible=False
    )

    # Show the plot
    fig.show()

