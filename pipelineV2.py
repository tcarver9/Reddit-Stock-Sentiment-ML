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

    # rolling volatility and interactions
    px["vol10"] = px["ret"].rolling(10).std()
    px["vol20"] = px["ret"].rolling(20).std()

    # Make sure 'date' is timezone-naive and datetime
    px["date"] = pd.to_datetime(px["date"]).dt.tz_localize(None)
    
    print("Prices columns:", px.columns.tolist()[:12], "...")

    return px

def daterange(start_date: dt.date, end_date: dt.date, step_days: int):
    cur = start_date
    while cur <= end_date:
        yield cur, min(cur + dt.timedelta(days=step_days - 1), end_date)
        cur = cur + dt.timedelta(days=step_days)

def _token_variants(ticker: str, company_hint: Optional[str] = None):
    t = ticker.upper()
    variants = {t, f"${t}"}
    if company_hint:
        variants.add(company_hint.lower())
        variants.add(company_hint.title())
    return list(variants)

def _pushshift_get(endpoint: str, params: dict, session: requests.Session):
    url = f"https://api.pushshift.io/reddit/search/{endpoint}"
    r = session.get(url, params=params, timeout=30)
    if r.status_code != 200:
        time.sleep(1.0)
        r = session.get(url, params=params, timeout=30)
    if r.status_code != 200:
        return []
    payload = r.json()
    return payload.get("data", [])

# Pull reddit posts and turn them into daily sentiment
def fetch_pushshift_day_bucketed(ticker: str, subs: List[str], start: str, end: Optional[str], company_hint: Optional[str] = "Apple") -> pd.DataFrame:
    start_dt = pd.to_datetime(start).date()
    end_dt = pd.to_datetime(end).date() if end else dt.date.today()

    from nltk import download as nltk_download
    nltk_download("vader_lexicon", quiet=True)
    sia = SentimentIntensityAnalyzer()

    daily_rows = []
    session = requests.Session()
    session.headers.update({"User-Agent": "portfolio-stock-sentiment/1.1"})
    tokens = _token_variants(ticker, company_hint)

    for chunk_start, chunk_end in tqdm(list(daterange(start_dt, end_dt, CFG.max_days_per_call)), desc="Pulling Reddit"):
        after_ts = int(dt.datetime.combine(chunk_start, dt.time.min).timestamp())
        before_ts = int(dt.datetime.combine(chunk_end + dt.timedelta(days=1), dt.time.min).timestamp())

        # Pull submissions and comments
        common = {
            "subreddit": ",".join(subs),
            "after": after_ts,
            "before": before_ts,
            "size": 250,
            "sort": "desc",
            "sort_type": "created_utc"
        }

        items = []
        for endpoint in ("submission", "comment"):
            # paginate by “before” 
            next_before = before_ts
            while True:
                params = dict(common)
                params["before"] = next_before
                batch = _pushshift_get(endpoint, params, session)
                if not batch:
                    break
                items.extend(batch)
                oldest = min(b.get("created_utc", next_before) for b in batch)
                if len(batch) < params["size"] or oldest == next_before:
                    break
                next_before = oldest

        if not items:
            for d in pd.date_range(chunk_start, chunk_end, freq="D"):
                daily_rows.append({"date": d.date(), "n_docs": 0, "sent_mean": 0.0, "pos_share": 0.0})
            continue

        # Build text + score
        rows = []
        for it in items:
            created = it.get("created_utc")
            if created is None: 
                continue
            title = it.get("title") or ""
            body = it.get("selftext") or it.get("body") or ""
            if body in ("[removed]", "[deleted]"):
                body = ""
            text = f"{title}. {body}".strip()
            # simple keyword filter 
            low = text.lower()
            if not any(tok.lower() in low for tok in tokens):
                continue
            score = it.get("score") or 0  # upvotes proxy
            rows.append({"created_utc": pd.to_datetime(created, unit="s"), "text": text, "score": int(score)})

        if not rows:
            for d in pd.date_range(chunk_start, chunk_end, freq="D"):
                daily_rows.append({"date": d.date(), "n_docs": 0, "sent_mean": 0.0, "pos_share": 0.0})
            continue

        tx = pd.DataFrame(rows)
        tx["date"] = tx["created_utc"].dt.date
        tx["sent"] = tx["text"].astype(str).apply(lambda t: sia.polarity_scores(t)["compound"])

        # Weighted by upvotes 
        tx["w"] = tx["score"].clip(lower=0) + 1
        day = tx.groupby("date").agg(
            n_docs=("text","count"),
            sent_mean=("sent", lambda s: float(np.average(s, weights=tx.loc[s.index, "w"]))),
            pos_share=("sent", lambda s: float(np.average((s > 0).astype(float), weights=tx.loc[s.index, "w"])))
        ).reset_index()

        # ensure all days exist
        cal = pd.DataFrame({"date": pd.date_range(chunk_start, chunk_end, freq="D").date})
        day = cal.merge(day, on="date", how="left").fillna({"n_docs":0, "sent_mean":0.0, "pos_share":0.0})

        daily_rows.extend(day.to_dict("records"))
        time.sleep(0.35)  # be polite

    daily = pd.DataFrame(daily_rows).sort_values("date")
    daily["date"] = pd.to_datetime(daily["date"])

    # Smooth + deltas + lags
    daily["sent_mean_3d"] = daily["sent_mean"].rolling(3, min_periods=1).mean()
    daily["n_docs_3d"] = daily["n_docs"].rolling(3, min_periods=1).mean()
    daily["sent_diff"] = daily["sent_mean"].diff()
    daily["sent_mean_l1"] = daily["sent_mean"].shift(1)
    daily["sent_mean_l3"] = daily["sent_mean"].rolling(3).mean().shift(1)
    daily["n_docs_l1"] = daily["n_docs"].shift(1)
    return daily


def build_feature_table(px: pd.DataFrame, sent: pd.DataFrame) -> pd.DataFrame:
    # Pull volatility columns from prices too
    price_cols = ["date","close","ret","sma5","sma20","rsi14","vol10","vol20","label_up"]
    missing = [c for c in price_cols if c not in px.columns]
    if missing:
        print("Warning: missing in prices, will recompute where possible:", missing)
        # if vol10/vol20 missing, recompute from ret
        if "vol10" in missing and "ret" in px.columns:
            px["vol10"] = px["ret"].rolling(10).std()
        if "vol20" in missing and "ret" in px.columns:
            px["vol20"] = px["ret"].rolling(20).std()

    f = px[[c for c in price_cols if c in px.columns]].copy()

    g = pd.merge(f, sent, on="date", how="left").sort_values("date")
    
    print("Feature cols after merge:", g.columns.tolist()[:20], "...")


    # fill small gaps in sentiment
    for col in ["n_docs","sent_mean","pos_share","sent_mean_3d","n_docs_3d","sent_diff","sent_mean_l1","sent_mean_l3","n_docs_l1"]:
        if col in g.columns:
            g[col] = g[col].ffill(limit=2)
            g[col] = g[col].fillna(0.0)

    # if vol10/vol20 still missing, recompute from joined returns
    if "vol10" not in g.columns and "ret" in g.columns:
        g["vol10"] = g["ret"].rolling(10).std()
    if "vol20" not in g.columns and "ret" in g.columns:
        g["vol20"] = g["ret"].rolling(20).std()

    # Interaction term
    if "sent_mean_3d" in g.columns and "vol10" in g.columns:
        g["sent_x_vol"] = g["sent_mean_3d"] * g["vol10"]

    # drop last row because label uses shift(-1)
    g = g.dropna(subset=["label_up"])
    g = g.iloc[:-1, :]
    return g.reset_index(drop=True)

def walkforward_xgb(df: pd.DataFrame) -> dict:
    # price/technical features
    tech = ["close","ret","sma5","sma20","rsi14","vol10","vol20"]

    # sentiment features 
    sent = [
        "n_docs","sent_mean","pos_share",
        "sent_mean_3d","n_docs_3d","sent_diff",
        "sent_mean_l1","sent_mean_l3","n_docs_l1",
        "sent_x_vol"
    ]

    # tech only
    df_tech = df.dropna(subset=tech + ["label_up"]).copy()
    Xt = df_tech[tech].values
    yt = df_tech["label_up"].values

    tscv = TimeSeriesSplit(n_splits=5)
    auc_tech = []
    for tr, te in tscv.split(Xt):
        m = XGBClassifier(
            n_estimators=400, max_depth=4, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
            random_state=42, n_jobs=4
        )
        m.fit(Xt[tr], yt[tr])
        p = m.predict_proba(Xt[te])[:, 1]
        auc_tech.append(roc_auc_score(yt[te], p))

    # tech + sentiment 
    needed = tech + sent
    df_ts = df.dropna(subset=needed + ["label_up"]).copy()
    Xs = df_ts[needed].values
    ys = df_ts["label_up"].values

    auc_techsent = []
    for tr, te in tscv.split(Xs):
        m = XGBClassifier(
            n_estimators=400, max_depth=4, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
            random_state=42, n_jobs=4
        )
        m.fit(Xs[tr], ys[tr])
        p = m.predict_proba(Xs[te])[:, 1]
        auc_techsent.append(roc_auc_score(ys[te], p))

    print(f"Tech-only AUC  : {np.mean(auc_tech):.3f} ± {np.std(auc_tech):.3f}")
    print(f"Tech+Sent AUC  : {np.mean(auc_techsent):.3f} ± {np.std(auc_techsent):.3f}")

    return {
        "auc_tech": float(np.mean(auc_tech)),
        "auc_tech_std": float(np.std(auc_tech)),
        "auc_tech_sent": float(np.mean(auc_techsent)),
        "auc_tech_sent_std": float(np.std(auc_techsent)),
        "n_rows_tech": int(len(df_tech)),
        "n_rows_tech_sent": int(len(df_ts))
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

    with open("results.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("Saved metrics to results.json")
    feat.to_csv("features.csv", index=False)
    print("Saved merged features to features.csv")

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
