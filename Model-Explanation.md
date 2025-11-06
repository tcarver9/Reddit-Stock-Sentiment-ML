---

## Understanding the XGBoost Classifier

This model uses an XGBoost classifier to estimate the probability that a stock’s price will rise over the next five trading days.  
It does not forecast exact prices; rather, it predicts whether short-term movement is likely to be positive or negative.

---

### How the Model Works

- The model is trained using a target variable called `label_up`.  
  - `label_up = 1` means the stock price increased over the next five days.  
  - `label_up = 0` means the stock price stayed the same or declined.  
- XGBoost learns from technical indicators (moving averages, RSI, volatility) and sentiment data (Reddit metrics) to estimate this probability.  
- The output is a value between 0 and 1:
  - Values closer to 1 indicate a higher probability of an upward move.
  - Values closer to 0 indicate a higher probability of a downward move.

---

### Reading Predictions

When the model runs, it produces a probability column and a classification signal:

| Predicted Probability | Model Interpretation |
|------------------------|----------------------|
| 0.80 | 80% chance the stock will rise in the next 5 days |
| 0.50 | No clear directional signal (essentially random) |
| 0.30 | 30% chance of an upward move, more likely to fall |

By default, the model classifies days where the probability is greater than 0.5 as “up” and those below 0.5 as “down.”  
This threshold can be adjusted depending on how conservative or aggressive you want the predictions to be.

---

### Visualizing the Predictions

A useful way to interpret the classifier is to overlay predicted signals on the stock’s price chart.  
For example:

- Green markers indicate dates where the model predicted an upward move.  
- Red markers indicate dates where the model predicted a downward move.  

This visualization helps show how model signals align with actual market movement.

---

### Model Confidence and AUC

The performance metric used is AUC (Area Under the ROC Curve).  
- AUC = 0.5 represents random guessing.  
- AUC closer to 1.0 indicates strong predictive power.

In this project, the AUC value is around 0.43, which suggests the model performs only slightly better than random chance.  
This is expected for large, efficient stocks such as Tesla, where information is rapidly priced in.

---

### Summary

| Concept | Explanation |
|----------|--------------|
| Model type | XGBoost binary classifier |
| Prediction goal | Direction of price movement over the next 5 days |
| Output | Probability between 0 and 1 |
| Decision threshold | 0.5 for “up” vs. “down” |
| Interpretation | p > 0.5 → Uptrend; p < 0.5 → Downtrend |
| Overall AUC | About 0.43 (weak predictive power for TSLA) |

The classifier framework demonstrates how market data and sentiment features can be combined to evaluate short term directional tendencies 

---

