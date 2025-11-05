---

## Visualization Explanations

The pipeline automatically generates a series of charts under the `/artifacts` directory.  
Each visualization highlights a different aspect of model performance, data relationships, or sentiment dynamics.

---

### 1. results.png

**Description**  
Compares model AUC performance and feature importance.

**Interpretation**  
- The AUC (Area Under the ROC Curve) measures how well the model distinguishes upward versus downward movements.  
  - AUC = 0.5 represents random guessing.  
  - Higher AUC values indicate stronger predictive ability.  
- The feature importance bars show which variables contribute most to the XGBoost model.  
  Technical indicators dominate, while sentiment features add limited predictive power.  

**Business insight**  
The model confirms that large, efficient stocks like Tesla incorporate most public sentiment quickly, but the workflow itself provides a framework for testing behavioral signals in other markets.

---

### 2. sent_vs_price.png

**Description**  
A time series chart comparing normalized closing price with the three day average Reddit sentiment (sent_mean_3d).

**Interpretation**  
- Reveals how sentiment trends move alongside price.  
- Spikes in sentiment often align with brief upward movements in price.  
- Divergences between the two may signal fading enthusiasm or overbought conditions.

**Business insight**  
This type of visualization can help analysts understand how investor mood correlates with price action and identify hype driven volatility in real time.

---

### 3. correlation_heatmap.png

**Description**  
A correlation matrix showing relationships among technical and sentiment features.

**Interpretation**  
- Bright colors represent strong positive correlations, darker colors represent negative ones.  
- SMA and volatility metrics are highly correlated, while sentiment features remain largely independent.  

**Business insight**  
Low correlation between sentiment and technical variables indicates that social data introduces a distinct behavioral dimension, potentially valuable for diversification or alternative signal generation.

---

### 4. roc_calibration.png

**Description**  
Displays both the ROC curve and the calibration curve for the trained model.

**Interpretation**  
- The ROC curve compares true versus false positive rates; curves closer to the upper left corner indicate better classifiers.  
- The calibration plot checks how well predicted probabilities align with real outcomes.  

**Business insight**  
Even if directional accuracy is limited, the calibration curve shows whether probability estimates are trustworthy for decision making and risk management.

---

### 5. strategy_lift.png

**Description**  
Compares cumulative returns from a model driven strategy versus a buy and hold baseline.

**Interpretation**  
- If the model’s curve rises above buy and hold, it captures a meaningful trading signal.  
- Overlapping curves indicate no significant predictive edge.

**Business insight**  
This chart links machine-learning outputs to financial outcomes, showing how predictive modeling can be evaluated through an investment or business impact lens.

---

### Summary

Together, these visualizations provide a complete picture of the project:
- results.png quantifies model performance.  
- sent_vs_price.png connects sentiment to price behavior.  
- correlation_heatmap.png reveals relationships among engineered features.  
- roc_calibration.png validates model reliability.  
- strategy_lift.png evaluates potential strategic lift from predictions.  

---

