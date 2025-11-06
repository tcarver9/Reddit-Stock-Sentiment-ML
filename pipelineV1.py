import os
import math
import time
import json
import datetime as dt
from dataclasses import dataclass
from typing import List, Optional, Dict

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from tqdm import tqdm
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, accuracy_score
from xgboost import XGBClassifier
from dotenv import load_dotenv

load_dotenv() # Load env file

@dataclass
class Config: # Read variables from env
    ticker: str = os.getenv("TICKER", "AAPL")
    start_date: str = os.getenv("START_DATE", "2023-01-01")
    end_date: Optional[str] = os.getenv("END_DATE") if os.getenv("END_DATE") else None
    subreddits: List[str] = tuple([s.strip() for s in os.getenv("SUBREDDITS", "stocks,wallstreetbets,investing").split(",")]) #Add more subreddits with a ,
    pushshift_url: str = "https://api.pushshift.io/reddit/search/submission" # Public endpoint to fetch Reddit posts
    min_posts_per_day: int = 1
    max_days_per_call: int = 7 # How many calendar days queried per request

CFG = Config()

def fetch_prices_yf(ticker: str, start: str, end: Optional[str] = None) -> pd.DataFrame:
    px = yf.download(ticker, start=start, end=end, auto_adjust=True)
    if px is None or px.empty:
        raise ValueError("No price data returned. Check ticker and dates.")

    # Ensure the datetime index becomes a column named 'date'
    px = px.copy()
    px = px.rename_axis("date").reset_index()

    # If columns are MultiIndex, flatten them 
    if isinstance(px.columns, pd.MultiIndex):
        px.columns = [
            "_".join([str(part) for part in col if part and str(part) != "nan"])
            for col in px.columns
        ]

    # Normalize column names to lowercase
    px = px.rename(columns=str.lower)
    # Remove _aapl from column names
    t_suffix = f"_{ticker.lower()}"
    px.columns = [c[:-len(t_suffix)] if c.endswith(t_suffix) else c for c in px.columns]

    # Expect columns like 'close' and 'adj close' (with auto_adjust=True, 'close' is adjusted)
    if "close" not in px.columns:
        raise ValueError(f"'close' column not found after yfinance download. Got columns: {list(px.columns)}")

    # Daily return
    px["ret"] = px["close"].pct_change()

    # Next day direction label
    px["label_up"] = (px["close"].shift(-1) > px["close"]).astype(int)

    # Simple moving averages
    px["sma5"] = px["close"].rolling(5).mean()
    px["sma20"] = px["close"].rolling(20).mean()

    # RSI (quick version from returns)
    up = px["ret"].clip(lower=0).rolling(14).mean()
    down = (-px["ret"].clip(upper=0)).rolling(14).mean()
    rs = up / (down.replace(0, np.nan))
    px["rsi14"] = 100 - (100 / (1 + rs))

    # Make sure 'date' is timezone-naive and datetime
    px["date"] = pd.to_datetime(px["date"]).dt.tz_localize(None)

    return px


def daterange(start_date: dt.date, end_date: dt.date, step_days: int):
    cur = start_date
    while cur <= end_date:
        yield cur, min(cur + dt.timedelta(days=step_days - 1), end_date)
        cur = cur + dt.timedelta(days=step_days)


# Pull reddit posts and turn them into daily sentiment
def fetch_pushshift_day_bucketed(ticker: str, subs: List[str], start: str, end: Optional[str]) -> pd.DataFrame: 
    start_dt = pd.to_datetime(start).date()
    end_dt = pd.to_datetime(end).date() if end else dt.date.today()

    from nltk import download as nltk_download
    nltk_download("vader_lexicon", quiet=True)
    sia = SentimentIntensityAnalyzer()

    daily_rows = []
    session = requests.Session()
    session.headers.update({"User-Agent": "portfolio-stock-sentiment/1.0"})

    for chunk_start, chunk_end in tqdm(list(daterange(start_dt, end_dt, CFG.max_days_per_call)), desc="Pulling Reddit"):
        params = {
            "q": ticker,
            "subreddit": ",".join(subs),
            "after": int(dt.datetime.combine(chunk_start, dt.time.min).timestamp()),
            "before": int(dt.datetime.combine(chunk_end + dt.timedelta(days=1), dt.time.min).timestamp()),
            "size": 250,
            "sort": "desc",
            "sort_type": "created_utc"
        }

        all_items = []
        next_after = params["after"]
        while True:
            params["after"] = next_after
            r = session.get(CFG.pushshift_url, params=params, timeout=20)
            if r.status_code != 200:
                time.sleep(1.0)
                r = session.get(CFG.pushshift_url, params=params, timeout=20)
                if r.status_code != 200:
                    break
            payload = r.json()
            data = payload.get("data", [])
            if not data:
                break
            all_items.extend(data)
            oldest = min([d["created_utc"] for d in data])
            next_after = oldest
            if len(data) < params["size"]:
                break

        if not all_items:
            for d in pd.date_range(chunk_start, chunk_end, freq="D"):
                daily_rows.append({"date": d.date(), "n_docs": 0, "sent_mean": 0.0, "pos_share": 0.0})
            continue

        df = pd.DataFrame(all_items)
        df["created_utc"] = pd.to_datetime(df["created_utc"], unit="s")
        df["date"] = df["created_utc"].dt.date

        texts = []
        for _, row in df.iterrows():
            title = row.get("title") or ""
            selftext = row.get("selftext") or ""
            if selftext in ("[removed]", "[deleted]"):
                selftext = ""
            texts.append({"date": row["date"], "text": f"{title}. {selftext}"})

        tx = pd.DataFrame(texts)
        if tx.empty:
            for d in pd.date_range(chunk_start, chunk_end, freq="D"):
                daily_rows.append({"date": d.date(), "n_docs": 0, "sent_mean": 0.0, "pos_share": 0.0})
            continue

        tx["sent"] = tx["text"].astype(str).apply(lambda t: sia.polarity_scores(t)["compound"])
        day_agg = tx.groupby("date").agg(
            n_docs=("text","count"), # Number of matching posts per day
            sent_mean=("sent","mean"), # Average compound sentiment
            pos_share=("sent", lambda s: float(np.mean(s > 0))) #Fraction of posts with positive sentiment
        ).reset_index()

        cal = pd.DataFrame({"date": pd.date_range(chunk_start, chunk_end, freq="D").date})
        day_agg = cal.merge(day_agg, on="date", how="left").fillna({"n_docs":0, "sent_mean":0.0, "pos_share":0.0})
        daily_rows.extend(day_agg.to_dict("records"))

        time.sleep(0.4)

    daily = pd.DataFrame(daily_rows)
    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily.sort_values("date")
    daily["sent_mean_3d"] = daily["sent_mean"].rolling(3, min_periods=1).mean() # 3 day mean
    daily["n_docs_3d"] = daily["n_docs"].rolling(3, min_periods=1).mean() # 3 day mean
    daily["sent_diff"] = daily["sent_mean"].diff() # Day to day change 
    return daily



def build_feature_table(px: pd.DataFrame, sent: pd.DataFrame) -> pd.DataFrame:
    f = px[["date","close","ret","sma5","sma20","rsi14","label_up"]].copy()
    g = pd.merge(f, sent, on="date", how="left").sort_values("date")
    for col in ["n_docs","sent_mean","pos_share","sent_mean_3d","n_docs_3d","sent_diff"]:
        if col in g.columns:
            g[col] = g[col].ffill(limit=2)
            g[col] = g[col].fillna(0.0)
    g = g.dropna(subset=["label_up"])
    g = g.iloc[:-1, :]
    return g.reset_index(drop=True)

# Train with cross validation
def walkforward_xgb(df: pd.DataFrame) -> Dict[str, float]:
    features = ["close","ret","sma5","sma20","rsi14","n_docs","sent_mean","pos_share","sent_mean_3d","n_docs_3d","sent_diff"]
    df_model = df.dropna(subset=features + ["label_up"]).copy()

    X = df_model[features].values
    y = df_model["label_up"].values

    tscv = TimeSeriesSplit(n_splits=5)
    aucs, accs = [], []

    for tr, te in tscv.split(X):
        model = XGBClassifier(
            n_estimators=400,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=4
        )
        model.fit(X[tr], y[tr])
        p = model.predict_proba(X[te])[:,1]
        aucs.append(roc_auc_score(y[te], p))
        accs.append(accuracy_score(y[te], (p > 0.5).astype(int)))

    return {
        "mean_auc": float(np.mean(aucs)),
        "std_auc": float(np.std(aucs)),
        "mean_acc": float(np.mean(accs)),
        "std_acc": float(np.std(accs))
    }

def main():
    print(f"Ticker: {CFG.ticker}, period: {CFG.start_date} to {CFG.end_date or 'today'}")
    prices = fetch_prices_yf(CFG.ticker, CFG.start_date, CFG.end_date)
    print(f"Prices shape: {prices.shape}") # How many rows of trading days were found

    sent = fetch_pushshift_day_bucketed(CFG.ticker, CFG.subreddits, CFG.start_date, CFG.end_date)
    print(f"Reddit daily rows: {sent.shape}") # How many calendar days were processed

    feat = build_feature_table(prices, sent)
    print(f"Feature table: {feat.shape}") # Intersection after joining

    metrics = walkforward_xgb(feat)
    print("Walk forward results:") # prints a JSON with mean AUC and mean accuracy.AUC near 0.5 means no better than chance. Anything higher suggests the features carry some signal
    print(json.dumps(metrics, indent=2))

    features = ["close","ret","sma5","sma20","rsi14","n_docs","sent_mean","pos_share","sent_mean_3d","n_docs_3d","sent_diff"]
    X = feat[features].values
    y = feat["label_up"].values
    cutoff = int(len(X) * 0.8)
    clf = XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9, random_state=42)
    clf.fit(X[:cutoff], y[:cutoff])
    importances = clf.feature_importances_
    top = sorted(zip(features, importances), key=lambda z: z[1], reverse=True)
    print("Feature importance preview:") # List features by importance in the final
    for name, val in top:
        print(f"{name}: {val:.3f}")

if __name__ == "__main__":
    main()
