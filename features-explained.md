---

##  Feature Dictionary and Business Meaning

This project combines technical indicators and sentiment analytics to model short term stock movement.  
Below is a complete explanation of each feature used in the XGBoost classifier.

---

###  Price and Target Variables

| Feature | Description | Business Meaning |
|----------|--------------|------------------|
| close | Adjusted closing price for the trading day. | Core market value; baseline for returns and moving averages. |
| ret | Daily return = (today’s close ÷ yesterday’s close) − 1. | Captures short-term momentum and volatility. |
| fwd_ret_H | Forward 5-day return (price 5 days ahead ÷ today’s price − 1). | Target horizon used to label positive/negative outcomes. |
| label_up | 1 if 5-day forward return is positive, 0 otherwise. | Binary outcome the model predicts. |

---

###  Trend Indicators (Moving Averages)

| Feature | Description | Business Meaning |
|----------|--------------|------------------|
| sma5 | 5-day Simple Moving Average of close price. | Short term trend direction; reacts quickly to new movements. |
| sma20 | 20-day Simple Moving Average. | Medium-term trend; helps identify reversals and trend strength. |

---

###  Momentum and Oscillator Indicators

| Feature | Description | Business Meaning |
|----------|--------------|------------------|
| rsi14 | 14-day Relative Strength Index calculated from average gains/losses. | Measures market momentum and overbought or oversold conditions. |

---

###  Volatility and Market Context

| Feature | Description | Business Meaning |
|----------|--------------|------------------|
| vol10 | Rolling 10-day standard deviation of daily returns. | Captures short-term uncertainty and risk. |
| vol20 | Rolling 20-day volatility. | Reflects medium-term stability or market stress. |
| ret_spy | Daily return of S&P 500 ETF (SPY). | Benchmarks the stock’s behavior against the overall market. |
| vol_spy10 | 10-day rolling volatility of SPY returns. | Indicates broader market turbulence influencing all equities. |

---

###  Reddit Sentiment and Community Activity

| Feature | Description | Business Meaning |
|----------|--------------|------------------|
| n_docs | Number of Reddit posts mentioning the ticker each day. | Proxy for attention and community engagement. |
| sent_mean | Average VADER sentiment score for that day. | Captures overall bullish or bearish tone. |
| pos_share | Share of posts with positive sentiment (> 0.05). | Measures dominance of optimistic discussions. |
| n_docs_3d | 3-day moving average of post counts. | Highlights sustained community attention. |
| sent_mean_3d | 3-day moving average sentiment. | Tracks persistent optimism or pessimism. |
| sent_diff | Daily change in average sentiment. | Detects sudden mood shifts after news or rumors. |
| sent_mean_l1 | Sentiment lagged by 1 day. | Tests delayed trader response to yesterday’s tone. |
| sent_mean_l3 | Sentiment lagged by 3 days. | Captures slower emotional carryover. |
| n_docs_l1 | Post count lagged by 1 day. | Measures yesterday’s discussion impact. |
| n_docs_l3 | Post count lagged by 3 days. | Captures multi-day persistence of attention. |

---

###  Interaction and Cross-Market Features

| Feature | Description | Business Meaning |
|----------|--------------|------------------|
| sent_x_vol | Interaction = sent_mean_3d × vol10. | Tests if sentiment has greater effect during volatile periods. |

---

###  Model Interpretation

Feature importance showed that:
- Technical indicators (close, sma20, vol20, vol_spy10, vol10, rsi14) dominated predictive power.  
- Sentiment variables (sent_mean, n_docs_3d, etc.) had low importance for Tesla’s 2023–2025 window, which aligns with expectations for highly efficient large-cap stocks.  
- Cross-market context (ret_spy, vol_spy10) contributed moderately, showing macro volatility’s role in short-term price changes.

While sentiment did not meaningfully improve accuracy here, this framework can be repurposed for smaller or hype-driven equities where social mood has a stronger influence.

---

