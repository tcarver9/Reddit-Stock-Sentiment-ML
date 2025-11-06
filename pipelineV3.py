import os
import math
import time
import json
import datetime as dt
from dataclasses import dataclass
from typing import List, Optional, Dict
from requests.exceptions import RequestException

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import yfinance as yf
from tqdm import tqdm
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.model_selection import TimeSeriesSplit
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve, auc, brier_score_loss
from xgboost import XGBClassifier
from dotenv import load_dotenv

# .venv\Scripts\activate

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

    # Ensure the datetime index becomes a column named date
    px = px.copy()
    px = px.rename_axis("date").reset_index()

    # If columns are MultiIndex, flatten them 
    if isinstance(px.columns, pd.MultiIndex):
        px.columns = [
            "_".join([str(part) for part in col if part and str(part) != "nan"])
            for col in px.columns
        ]

    px = px.rename(columns=str.lower)
    # Remove _aapl from column names
    t_suffix = f"_{ticker.lower()}"
    px.columns = [c[:-len(t_suffix)] if c.endswith(t_suffix) else c for c in px.columns]

    # Expect columns like 'close' and 'adj close' (with auto_adjust=True, 'close' is adjusted)
    if "close" not in px.columns:
        raise ValueError(f"'close' column not found after yfinance download. Got columns: {list(px.columns)}")

    # Daily return
    px["ret"] = px["close"].pct_change()

    # predict 5 day ahead return
    H = 5  # number of trading days ahead
    px["fwd_ret_H"] = px["close"].shift(-H) / px["close"] - 1
    px["label_up"] = (px["fwd_ret_H"] > 0).astype(int)

    # Simple moving averages
    px["sma5"] = px["close"].rolling(5).mean()
    px["sma20"] = px["close"].rolling(20).mean()

    # RSI quick version from returns
    up = px["ret"].clip(lower=0).rolling(14).mean()
    down = (-px["ret"].clip(upper=0)).rolling(14).mean()
    rs = up / (down.replace(0, np.nan))
    px["rsi14"] = 100 - (100 / (1 + rs))

    # rolling volatility and interactions
    px["vol10"] = px["ret"].rolling(10).std()
    px["vol20"] = px["ret"].rolling(20).std()

    # Fetch SPY benchmark and flatten columns
    spy = yf.download("SPY", start=start, end=end, auto_adjust=True)

    # Flatten MultiIndex 
    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = [col[0].lower() if isinstance(col, tuple) else col.lower() for col in spy.columns]
    else:
        spy.columns = [c.lower() for c in spy.columns]

    # Reset index to make date a column
    spy = spy.reset_index()
    spy.rename(columns={"Date": "date"}, inplace=True)

    # Compute SPY returns and volatility
    spy["ret_spy"] = spy["close"].pct_change()
    spy["vol_spy10"] = spy["ret_spy"].rolling(10).std()

    # Merge SPY into main prices DataFrame
    px = px.merge(spy[["date", "ret_spy", "vol_spy10"]], on="date", how="left")

    # Make sure date is timezone correct and datetime
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

def _pushshift_get(endpoint: str, params: dict, session: requests.Session, retries: int = 3, backoff: int = 5):

    url = f"https://api.pushshift.io/reddit/search/{endpoint}"

    for attempt in range(1, retries + 1):
        try:
            r = session.get(url, params=params, timeout=30)
            if r.status_code == 200:
                payload = r.json()
                return payload.get("data", [])
            else:
                print(f"Pushshift non-200 ({r.status_code}) on attempt {attempt}/{retries}")
        except RequestException as e:
            print(f"Pushshift connection error ({e}) on attempt {attempt}/{retries}")
        # Wait and retry
        time.sleep(backoff)

    print(f"Pushshift failed after {retries} retries; skipping this batch.")
    return []

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
            # paginate by before 
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
    price_cols = ["date","close","ret","sma5","sma20","rsi14","vol10","vol20","ret_spy","vol_spy10","label_up"]
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
    
    print("Feature cols after merge:", g.columns.tolist())



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
    tech = ["close","ret","sma5","sma20","rsi14","vol10","vol20","ret_spy","vol_spy10"]

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
            n_estimators=600, max_depth=5, learning_rate=0.03,
            subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,reg_alpha=0.3,
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
            n_estimators=600, max_depth=5, learning_rate=0.03,
            subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,reg_alpha=0.3,
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
    print(f"Prices shape: {prices.shape}")  # How many rows of trading days

    sent = fetch_pushshift_day_bucketed(CFG.ticker, CFG.subreddits, CFG.start_date, CFG.end_date)
    print(f"Reddit daily rows: {sent.shape}")  # How many calendar days processed

    feat = build_feature_table(prices, sent)
    print(f"Feature table: {feat.shape}")  # Intersection after joining

    # tech vs tech+sent
    metrics = walkforward_xgb(feat)
    print("Walk forward results:")
    print(json.dumps(metrics, indent=2))

    # Save metrics + dataset 
    with open("results.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("Saved metrics to results.json")
    feat.to_csv("features.csv", index=False)
    print("Saved merged features to features.csv")

    # Final model on 80% 
    # Build a feature list 
    base_tech = ["close","ret","sma5","sma20","rsi14","vol10","vol20"]
    optional_tech = ["ret_spy","vol_spy10"]       # only if SPY context
    sent_cols = [
        "n_docs","sent_mean","pos_share","sent_mean_3d","n_docs_3d",
        "sent_diff","sent_mean_l1","sent_mean_l3","n_docs_l1","sent_x_vol"
    ]

    all_candidates = base_tech + optional_tech + sent_cols
    features = [c for c in all_candidates if c in feat.columns]

    # Drop rows with missing feature
    df_model = feat.dropna(subset=features + ["label_up"]).copy()
    X = df_model[features].values
    y = df_model["label_up"].values
    dates = pd.to_datetime(df_model["date"]).values

    cutoff = int(len(X) * 0.8)
    clf = XGBClassifier(
        n_estimators=600, max_depth=5, learning_rate=0.03,
        subsample=0.9, colsample_bytree=0.9,
        reg_lambda=1.0, reg_alpha=0.3,
        random_state=42, n_jobs=4
    )
    clf.fit(X[:cutoff], y[:cutoff])
    importances = clf.feature_importances_

    # Importance preview
    top = sorted(zip(features, importances), key=lambda z: z[1], reverse=True)
    print("Feature importance preview:")
    for name, val in top:
        print(f"{name}: {val:.3f}")

    # Holdout preds for quality/strategy charts
    y_prob = clf.predict_proba(X[cutoff:])[:, 1]
    y_true = y[cutoff:]
    dates_holdout = dates[cutoff:]

    os.makedirs("artifacts", exist_ok=True)

    # AUC bars + feature importances
    auc_tech = metrics.get("auc_tech", 0.0)
    auc_tech_sent = metrics.get("auc_tech_sent", 0.0)
    imp_series = pd.Series(importances, index=features).sort_values(ascending=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    axes[0].bar(["Tech only", "Tech + Sent"], [auc_tech, auc_tech_sent])
    axes[0].set_title("Walk-forward AUC"); axes[0].set_ylabel("AUC")
    ymin = min(0.45, auc_tech, auc_tech_sent) - 0.01
    ymax = max(0.70, auc_tech, auc_tech_sent) + 0.01
    axes[0].set_ylim(ymin, ymax)
    for i, v in enumerate([auc_tech, auc_tech_sent]):
        axes[0].text(i, v + 0.005, f"{v:.3f}", ha="center", va="bottom", fontsize=9)

    topk = min(12, len(imp_series))
    imp_series.tail(topk).plot(kind="barh", ax=axes[1])
    axes[1].set_title("XGBoost Feature Importance"); axes[1].set_xlabel("importance")
    plt.savefig("artifacts/results.png", dpi=200); plt.close(fig)
    print("Saved chart to artifacts/results.png")

    # Price vs Sentiment
    fig = plt.figure(figsize=(12,5)); ax1 = plt.gca()
    price_norm = feat["close"] / feat["close"].iloc[0]
    ax1.plot(feat["date"], price_norm, label="Price (normalized)")
    ax1.set_xlabel("Date"); ax1.set_ylabel("Price (normalized)")
    ax2 = ax1.twinx()
    if "sent_mean_3d" in feat.columns:
        sent = feat["sent_mean_3d"].fillna(0.0)
        smin, smax = sent.min(), sent.max()
        sent_scaled = (sent - smin) / (smax - smin + 1e-9)
        ax2.plot(feat["date"], sent_scaled, linestyle="--", label="Sentiment (scaled)")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1+lines2, labels1+labels2, loc="upper left")
    plt.title("Price vs. Sentiment (3-day avg)")
    plt.tight_layout(); plt.savefig("artifacts/sent_vs_price.png", dpi=200); plt.close(fig)
    print("Saved chart to artifacts/sent_vs_price.png")

    # Activity vs Volume
    if "n_docs_3d" in feat.columns and "volume" in feat.columns:
        fig = plt.figure(figsize=(12,5)); ax1 = plt.gca()
        ax1.plot(feat["date"], feat["volume"], label="Trading Volume")
        ax1.set_xlabel("Date"); ax1.set_ylabel("Volume")
        ax2 = ax1.twinx()
        ax2.plot(feat["date"], feat["n_docs_3d"].fillna(0.0), linestyle="--", label="Reddit docs (3d avg)")
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1+lines2, labels1+labels2, loc="upper left")
        plt.title("Market Activity vs. Community Activity")
        plt.tight_layout(); plt.savefig("artifacts/activity_vs_volume.png", dpi=200); plt.close(fig)
        print("Saved chart to artifacts/activity_vs_volume.png")

    # Correlation heatmap
    cols_for_corr = [c for c in features if c in feat.columns]
    if cols_for_corr:
        corr_df = feat[cols_for_corr].copy()
        if "fwd_ret_H" in feat.columns:
            corr_df["fwd_ret_H"] = feat["fwd_ret_H"]
        corr = corr_df.corr().values; labels = corr_df.columns.tolist()
        fig = plt.figure(figsize=(10,8)); ax = plt.gca()
        im = ax.imshow(corr, interpolation="nearest")
        ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=90)
        ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
        plt.title("Feature Correlation Heatmap")
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.tight_layout(); plt.savefig("artifacts/correlation_heatmap.png", dpi=200); plt.close(fig)
        print("Saved chart to artifacts/correlation_heatmap.png")

    # ROC + Calibration (holdout)
    if len(y_true) > 10:
        fpr, tpr, _ = roc_curve(y_true, y_prob); roc_auc = auc(fpr, tpr)
        fig = plt.figure(figsize=(12,5))
        ax1 = plt.subplot(1,2,1)
        ax1.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
        ax1.plot([0,1],[0,1], linestyle="--")
        ax1.set_xlabel("False Positive Rate"); ax1.set_ylabel("True Positive Rate")
        ax1.set_title("ROC (Holdout)"); ax1.legend(loc="lower right")
        ax2 = plt.subplot(1,2,2)
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy="quantile")
        ax2.plot(prob_pred, prob_true, marker="o", label="Model")
        ax2.plot([0,1],[0,1], linestyle="--")
        ax2.set_xlabel("Predicted probability"); ax2.set_ylabel("Observed frequency")
        ax2.set_title("Calibration (Holdout)"); ax2.legend()
        plt.tight_layout(); plt.savefig("artifacts/roc_calibration.png", dpi=200); plt.close(fig)
        print("Saved chart to artifacts/roc_calibration.png")

    # Illustrative strategy lift 
    if "ret" in feat.columns and len(y_true) > 10:
        hold = pd.DataFrame({"date": pd.to_datetime(dates_holdout), "prob": y_prob}).sort_values("date")
        tmp = feat[["date","ret"]].copy()
        df_hold = hold.merge(tmp, on="date", how="left").sort_values("date")
        df_hold["prob_shift"] = df_hold["prob"].shift(1)
        df_hold["position"] = (df_hold["prob_shift"] >= 0.55).astype(float)
        df_hold["ret_next"] = df_hold["ret"].shift(-1)
        df_hold["strat_ret"] = df_hold["position"] * df_hold["ret_next"]
        df_hold["cum_strat"] = (1 + df_hold["strat_ret"].fillna(0)).cumprod()
        df_hold["cum_buyhold"] = (1 + df_hold["ret_next"].fillna(0)).cumprod()
        fig = plt.figure(figsize=(12,5))
        plt.plot(df_hold["date"], df_hold["cum_buyhold"], label="Buy & Hold (holdout)")
        plt.plot(df_hold["date"], df_hold["cum_strat"], linestyle="--", label="Model (p>=0.55)")
        plt.legend(); plt.title("Illustrative Cumulative Return (Holdout)\n(For demonstration only)")
        plt.xlabel("Date"); plt.ylabel("Cumulative index")
        plt.tight_layout(); plt.savefig("artifacts/strategy_lift.png", dpi=200); plt.close(fig)
        print("Saved chart to artifacts/strategy_lift.png")

    # Forward return by sentiment quintile if label horizon present
    if "fwd_ret_H" in feat.columns and "sent_mean_3d" in feat.columns:
        dfq = feat[["date","sent_mean_3d","fwd_ret_H"]].dropna().copy()
        if len(dfq) >= 20:
            q = pd.qcut(dfq["sent_mean_3d"], q=5, labels=[1,2,3,4,5])
            grp = dfq.groupby(q)["fwd_ret_H"].mean()
            fig = plt.figure(figsize=(6,4))
            plt.plot(grp.index.astype(int), grp.values, marker="o")
            plt.xticks([1,2,3,4,5]); plt.xlabel("Sentiment quintile (low → high)")
            plt.ylabel("Mean forward H-day return")
            plt.title("Return by Sentiment Quintile")
            plt.tight_layout(); plt.savefig("artifacts/return_by_sent_quintile.png", dpi=200); plt.close(fig)
            print("Saved chart to artifacts/return_by_sent_quintile.png")

if __name__ == "__main__":
    main()

