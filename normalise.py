"""
normalise.py
============
Reads all 5 platform datasets, maps each to a common schema,
binarises engagement_class per-platform at the platform median,
and outputs data/combined_dataset.csv

Location : d:/dg-social/phase2/normalise.py
Run      : python normalise.py
Output   : d:/dg-social/phase2/data/combined_dataset.csv
"""

import os
import json
import warnings
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
BASE       = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE, "data")
RAW_DIR    = os.path.join(DATA_DIR, "raw_datasets")
OUTPUT_CSV = os.path.join(DATA_DIR, "combined_dataset.csv")

# ─────────────────────────────────────────────
# COMMON SCHEMA
# ─────────────────────────────────────────────
# Every platform maps to these columns only:
#
#   platform         str   facebook/tiktok/twitter/youtube
#   post_type        str   Photo/Video/Text/Link/Reel/Status/Short
#   post_hour        int   0-23  (hour of day post was published)
#   post_weekday     int   1-7   (1=Monday … 7=Sunday)
#   is_weekend       int   0/1
#   is_peak_hour     int   0/1   (peak = 8-10, 12-14, 18-22)
#   has_media        int   0/1
#   is_video         int   0/1
#   is_paid          int   0/1
#   caption_length   int   word count proxy (or video duration for TikTok)
#   hashtag_count    int   number of hashtags (0 if unavailable)
#   follower_count   int   page/channel/author follower count (0 if unavailable)
#   total_likes      int   raw likes/reactions
#   total_comments   int   raw comments
#   total_shares     int   raw shares/retweets/repins
#   total_views      int   raw views/impressions/reach (0 if unavailable)
#   engagement_score float (likes + comments + shares)  — used for binarisation
#   engagement_class int   0=LOW, 1=HIGH  — target variable (per-platform median split)

PEAK_HOURS = set(range(8, 11)) | set(range(12, 15)) | set(range(18, 23))


def binarise_per_platform(df: pd.DataFrame) -> pd.DataFrame:
    """Split engagement_class at the per-platform median of engagement_score."""
    df["engagement_class"] = 0
    for platform in df["platform"].unique():
        mask   = df["platform"] == platform
        median = df.loc[mask, "engagement_score"].median()
        df.loc[mask & (df["engagement_score"] >= median), "engagement_class"] = 1
    return df


def is_peak(hour):
    try:
        return int(int(hour) in PEAK_HOURS)
    except Exception:
        return 0


def is_wknd(weekday):
    """weekday: 1=Mon…7=Sun or 0=Mon…6=Sun — handle both."""
    try:
        wd = int(weekday)
        return int(wd in [6, 7, 0])  # covers both conventions
    except Exception:
        return 0


# ═══════════════════════════════════════════════════════════════
# 1. FACEBOOK  (UCI)  — 500 rows
# ═══════════════════════════════════════════════════════════════
def load_facebook() -> pd.DataFrame:
    path = os.path.join(DATA_DIR, "dataset.csv")
    df   = pd.read_csv(path, sep=";")

    # Type: Photo / Status / Link / Video
    type_map = {
        "Photo":  ("Photo", 1, 0),
        "Video":  ("Video", 1, 1),
        "Status": ("Text",  0, 0),
        "Link":   ("Link",  0, 0),
    }
    df["_type"]    = df["Type"].map(lambda x: type_map.get(str(x), ("Text", 0, 0)))
    df["post_type"]    = df["_type"].map(lambda x: x[0])
    df["has_media"]    = df["_type"].map(lambda x: x[1])
    df["is_video"]     = df["_type"].map(lambda x: x[2])

    out = pd.DataFrame({
        "platform":       "facebook",
        "post_type":      df["post_type"],
        "post_hour":      pd.to_numeric(df["Post Hour"],    errors="coerce").fillna(12).astype(int),
        "post_weekday":   pd.to_numeric(df["Post Weekday"], errors="coerce").fillna(1).astype(int),
        "is_weekend":     df["Post Weekday"].map(is_wknd),
        "is_peak_hour":   df["Post Hour"].map(is_peak),
        "has_media":      df["has_media"],
        "is_video":       df["is_video"],
        "is_paid":        pd.to_numeric(df["Paid"],         errors="coerce").fillna(0).astype(int),
        "caption_length": 0,                                # not available in UCI
        "hashtag_count":  0,                                # not available in UCI
        "follower_count": pd.to_numeric(df["Page total likes"], errors="coerce").fillna(0).astype(int),
        "total_likes":    pd.to_numeric(df["like"],         errors="coerce").fillna(0).astype(int),
        "total_comments": pd.to_numeric(df["comment"],      errors="coerce").fillna(0).astype(int),
        "total_shares":   pd.to_numeric(df["share"],        errors="coerce").fillna(0).astype(int),
        "total_views":    pd.to_numeric(df["Lifetime Post Total Reach"], errors="coerce").fillna(0).astype(int),
    })
    out["engagement_score"] = out["total_likes"] + out["total_comments"] + out["total_shares"]
    print(f"  Facebook loaded   : {len(out):>7,} rows")
    return out


# ═══════════════════════════════════════════════════════════════
# 2. TIKTOK  — 19 382 rows
# ═══════════════════════════════════════════════════════════════
def load_tiktok() -> pd.DataFrame:
    path = os.path.join(RAW_DIR, "tiktok", "tiktok_dataset.csv")
    df   = pd.read_csv(path)

    # No timestamp → default peak/weekday unknown → fill 0
    out = pd.DataFrame({
        "platform":       "tiktok",
        "post_type":      "Video",
        "post_hour":      0,
        "post_weekday":   0,
        "is_weekend":     0,
        "is_peak_hour":   0,
        "has_media":      1,
        "is_video":       1,
        "is_paid":        0,
        # video_duration_sec as proxy for caption_length
        "caption_length": pd.to_numeric(df["video_duration_sec"], errors="coerce").fillna(0).astype(int),
        "hashtag_count":  0,
        # verified_status as follower proxy (1 if verified)
        "follower_count": df["verified_status"].map(lambda x: 1 if str(x).lower() == "verified" else 0),
        "total_likes":    pd.to_numeric(df["video_like_count"],    errors="coerce").fillna(0).astype(int),
        "total_comments": pd.to_numeric(df["video_comment_count"], errors="coerce").fillna(0).astype(int),
        "total_shares":   pd.to_numeric(df["video_share_count"],   errors="coerce").fillna(0).astype(int),
        "total_views":    pd.to_numeric(df["video_view_count"],    errors="coerce").fillna(0).astype(int),
    })
    out["engagement_score"] = out["total_likes"] + out["total_comments"] + out["total_shares"]
    print(f"  TikTok loaded     : {len(out):>7,} rows")
    return out


# ═══════════════════════════════════════════════════════════════
# 3. TWITTER / DMO  — 23 006 rows
# ═══════════════════════════════════════════════════════════════
def load_twitter() -> pd.DataFrame:
    path = os.path.join(
        RAW_DIR, "twitter",
        "DMO social media engagement dataset",
        "Data LIWC 01 02 23.csv"
    )
    df = pd.read_csv(path, encoding="latin1")

    # Time column: "Business hours" / "Non-business hours"
    def parse_hour(t):
        if isinstance(t, str) and "non" not in t.lower():
            return 10   # business hours → proxy 10am
        return 20       # non-business hours → proxy 8pm

    # Day column: "weekday" / "weekend"
    def parse_weekday(d):
        if isinstance(d, str) and "weekend" in d.lower():
            return 6    # Saturday proxy
        return 2        # Tuesday proxy

    # ContentType → post_type
    def map_ctype(c):
        c = str(c).lower()
        if "video" in c:
            return ("Video", 1, 1)
        if "photo" in c or "image" in c:
            return ("Photo", 1, 0)
        return ("Text", 0, 0)

    df["_ct"] = df["ContentType"].map(map_ctype)

    out = pd.DataFrame({
        "platform":       "twitter",
        "post_type":      df["_ct"].map(lambda x: x[0]),
        "post_hour":      df["Time"].map(parse_hour),
        "post_weekday":   df["Day"].map(parse_weekday),
        "is_weekend":     df["Day"].map(lambda d: 1 if isinstance(d, str) and "weekend" in d.lower() else 0),
        "is_peak_hour":   df["Time"].map(lambda t: 1 if isinstance(t, str) and "non" not in t.lower() else 0),
        "has_media":      df["_ct"].map(lambda x: x[1]),
        "is_video":       df["_ct"].map(lambda x: x[2]),
        "is_paid":        0,
        # WC = word count of tweet text
        "caption_length": pd.to_numeric(df["WC"],           errors="coerce").fillna(0).astype(int),
        "hashtag_count":  0,
        "follower_count": pd.to_numeric(df["Followers"],     errors="coerce").fillna(0).astype(int),
        "total_likes":    pd.to_numeric(df["like_count"],    errors="coerce").fillna(0).astype(int),
        "total_comments": pd.to_numeric(df["reply_count"],   errors="coerce").fillna(0).astype(int),
        "total_shares":   pd.to_numeric(df["retweet_count"], errors="coerce").fillna(0).astype(int),
        "total_views":    0,
    })
    out["engagement_score"] = out["total_likes"] + out["total_comments"] + out["total_shares"]
    print(f"  Twitter loaded    : {len(out):>7,} rows")
    return out


# ═══════════════════════════════════════════════════════════════
# 4. YOUTUBE NON-TRENDS  — 390 043 rows
# ═══════════════════════════════════════════════════════════════
def load_youtube_nontrends() -> pd.DataFrame:
    path = os.path.join(RAW_DIR, "youtube_nontrends", "Youtube_Videos.csv")
    df   = pd.read_csv(path, low_memory=False)

    df["_pub"] = pd.to_datetime(df["publishedAt"], errors="coerce", utc=True)

    # tags column — count hashtags/tags as proxy
    def count_tags(t):
        if pd.isna(t) or t == "[none]":
            return 0
        return len(str(t).split("|"))

    out = pd.DataFrame({
        "platform":       "youtube",
        "post_type":      "Video",
        "post_hour":      df["_pub"].dt.hour.fillna(0).astype(int),
        "post_weekday":   df["_pub"].dt.weekday.fillna(0).astype(int) + 1,  # 1-7
        "is_weekend":     df["_pub"].dt.weekday.map(lambda w: int(w >= 5) if pd.notna(w) else 0),
        "is_peak_hour":   df["_pub"].dt.hour.map(lambda h: is_peak(h) if pd.notna(h) else 0),
        "has_media":      1,
        "is_video":       1,
        "is_paid":        0,
        "caption_length": 0,
        "hashtag_count":  df.get("tags", pd.Series([""] * len(df))).map(count_tags),
        "follower_count": 0,
        "total_likes":    pd.to_numeric(df["likes"],         errors="coerce").fillna(0).astype(int),
        "total_comments": pd.to_numeric(df["comment_count"], errors="coerce").fillna(0).astype(int),
        "total_shares":   0,
        "total_views":    pd.to_numeric(df["view_count"],    errors="coerce").fillna(0).astype(int),
    })
    out["engagement_score"] = out["total_likes"] + out["total_comments"] + out["total_shares"]
    print(f"  YouTube NT loaded : {len(out):>7,} rows")
    return out


# ═══════════════════════════════════════════════════════════════
# 5. YOUTUBE TRENDING  (US + IN + GB + CA + others)
# ═══════════════════════════════════════════════════════════════
def load_youtube_trending() -> pd.DataFrame:
    folder  = os.path.join(RAW_DIR, "youtube_trending")
    csvs    = [f for f in os.listdir(folder) if f.endswith("videos.csv")]
    frames  = []

    for fname in csvs:
        fpath = os.path.join(folder, fname)
        try:
            df = pd.read_csv(fpath, encoding="latin1", low_memory=False)
            df["_pub"] = pd.to_datetime(df["publish_time"], errors="coerce", utc=True)

            def count_tags_yt(t):
                if pd.isna(t) or t == "[none]":
                    return 0
                return len(str(t).split("|"))

            chunk = pd.DataFrame({
                "platform":       "youtube",
                "post_type":      "Video",
                "post_hour":      df["_pub"].dt.hour.fillna(0).astype(int),
                "post_weekday":   df["_pub"].dt.weekday.fillna(0).astype(int) + 1,
                "is_weekend":     df["_pub"].dt.weekday.map(lambda w: int(w >= 5) if pd.notna(w) else 0),
                "is_peak_hour":   df["_pub"].dt.hour.map(lambda h: is_peak(h) if pd.notna(h) else 0),
                "has_media":      1,
                "is_video":       1,
                "is_paid":        0,
                "caption_length": 0,
                "hashtag_count":  df["tags"].map(count_tags_yt),
                "follower_count": 0,
                "total_likes":    pd.to_numeric(df["likes"],         errors="coerce").fillna(0).astype(int),
                "total_comments": pd.to_numeric(df["comment_count"], errors="coerce").fillna(0).astype(int),
                "total_shares":   0,
                "total_views":    pd.to_numeric(df["views"],         errors="coerce").fillna(0).astype(int),
            })
            chunk["engagement_score"] = chunk["total_likes"] + chunk["total_comments"]
            frames.append(chunk)
        except Exception as e:
            print(f"  Skipping {fname}: {e}")

    out = pd.concat(frames, ignore_index=True)
    print(f"  YouTube TR loaded : {len(out):>7,} rows  ({len(csvs)} country files)")
    return out


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════
def main():
    print("\n📦  Loading datasets...")
    fb  = load_facebook()
    tt  = load_tiktok()
    tw  = load_twitter()
    ynt = load_youtube_nontrends()
    ytr = load_youtube_trending()

    print("\n🔗  Concatenating...")
    combined = pd.concat([fb, tt, tw, ynt, ytr], ignore_index=True)
    print(f"  Total rows (raw)  : {len(combined):>7,}")

    # ── Drop rows with no engagement signal ──────────────────
    combined = combined[combined["engagement_score"] > 0].copy()
    print(f"  After zero-drop   : {len(combined):>7,}")

    # ── Drop extreme outliers (top 0.1% per platform) ────────
    def remove_outliers(df):
        frames = []
        for p in df["platform"].unique():
            sub = df[df["platform"] == p].copy()
            cap = sub["engagement_score"].quantile(0.999)
            frames.append(sub[sub["engagement_score"] <= cap])
        return pd.concat(frames, ignore_index=True)

    combined = remove_outliers(combined)
    print(f"  After outlier-cap : {len(combined):>7,}")

    # ── Per-platform binarisation ────────────────────────────
    print("\n🎯  Binarising target per platform at median...")
    combined = binarise_per_platform(combined)

    for p in sorted(combined["platform"].unique()):
        sub    = combined[combined["platform"] == p]
        median = sub["engagement_score"].median()
        low    = (sub["engagement_class"] == 0).sum()
        high   = (sub["engagement_class"] == 1).sum()
        print(f"  {p:<10}  median={median:>10.1f}  LOW={low:>6,}  HIGH={high:>6,}")

    # ── Remove helper column ─────────────────────────────────
    combined.drop(columns=["engagement_score"], inplace=True)

    # ── Final dtypes ─────────────────────────────────────────
    int_cols = [
        "post_hour","post_weekday","is_weekend","is_peak_hour",
        "has_media","is_video","is_paid","caption_length",
        "hashtag_count","follower_count","total_likes",
        "total_comments","total_shares","total_views","engagement_class"
    ]
    for c in int_cols:
        combined[c] = pd.to_numeric(combined[c], errors="coerce").fillna(0).astype(int)

    # ── Save ─────────────────────────────────────────────────
    combined.to_csv(OUTPUT_CSV, index=False)
    print(f"\n✅  Saved → {OUTPUT_CSV}")
    print(f"    Shape  : {combined.shape}")
    print(f"    Columns: {combined.columns.tolist()}")
    print(f"\n    Platform distribution:")
    print(combined["platform"].value_counts().to_string())
    print(f"\n    Class balance (overall):")
    print(combined["engagement_class"].value_counts().to_string())


if __name__ == "__main__":
    main()
