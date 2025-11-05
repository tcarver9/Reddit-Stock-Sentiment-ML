# Reddit-API-Stock-Prediction

## Business Summary

- **Objective:** Estimate short-horizon directional risk using investor sentiment and market context.
- **Why it matters:** Helps prioritize marketing/ops decisions (timing campaigns, inventory hedging) when demand correlates with risk-on markets and sentiment.
- **Key findings:** 
  - Sentiment tracks market regimes; during high volatility, sentiment-weighted signals can improve discrimination.
  - Model quality on holdout shown via ROC/Calibration; strategy curve illustrates potential lift 
- **Artifacts:**
  - ![AUC & Importances](artifacts/results.png)
  - ![Price vs Sentiment](artifacts/sent_vs_price.png)
  - ![Activity vs Volume](artifacts/activity_vs_volume.png)
  - ![ROC & Calibration](artifacts/roc_calibration.png)
  - ![Strategy Lift](artifacts/strategy_lift.png)
  - ![Return by Sentiment Quintile](artifacts/return_by_sent_quintile.png)
  - ![Correlation Heatmap](artifacts/correlation_heatmap.png)
