from datetime import datetime
import time
import requests
from pathlib import Path
import re
import pandas as pd
import html
import random
import math

# ==========================================
# Checkpoint Utilities
# ==========================================
CHECKPOINT_FILE = Path("processed_appids.txt")

def load_processed_ids():
    if not CHECKPOINT_FILE.exists():
        return set()
    with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
        return set(line.strip() for line in f if line.strip())

def save_processed_id(appid):
    with open(CHECKPOINT_FILE, "a", encoding="utf-8") as f:
        f.write(f"{appid}\n")

# ==========================================
# Utilities
# ==========================================
def print_log(*args):
    print(f"[{str(datetime.now())[:-3]}] ", end="")
    print(*args)

def clean_html(raw_html):
    if not raw_html: return ""
    cleanr = re.compile('<.*?>')
    return html.unescape(re.sub(cleanr, '', raw_html)).strip()

def get_search_results(params):
    req_sr = requests.get("https://store.steampowered.com/search/results/", params=params)
    if req_sr.status_code != 200:
        return {"items": [], "total_count": 0}
    
    try:
        data = req_sr.json()
        return data
    except:
        return {"items": [], "total_count": 0}

def get_app_details(appid):
    while True:
        if appid is None: return {}
        req = requests.get("https://store.steampowered.com/api/appdetails/", 
                           params={"appids": appid, "cc": "hk", "l": "english"})
        if req.status_code == 200:
            return req.json().get(str(appid), {})
        elif req.status_code == 429:
            time.sleep(10)
        elif req.status_code == 403:
            time.sleep(300)
        else:
            return {}

# ==========================================
# Review Fetching & Dynamic Balancing Logic
# ==========================================

def fetch_raw_reviews(appid, review_type, max_fetch_limit):
    """
    Fetches a LARGE batch of reviews to allow for statistical balancing later.
    """
    url = f"https://store.steampowered.com/appreviews/{appid}"
    params = {
        'json': 1,
        'filter': 'recent',       
        'language': 'english',    
        'num_per_page': 100,
        'purchase_type': 'all',
        'review_type': review_type, 
        'cursor': '*'
    }
    
    raw_reviews = []
    query_summary = {}
    
    while len(raw_reviews) < max_fetch_limit:
        try:
            response = requests.get(url, params=params)
            if response.status_code != 200: break
            
            data = response.json()
            if not query_summary: query_summary = data.get('query_summary', {})
            
            reviews = data.get('reviews', [])
            if not reviews: break 
            
            raw_reviews.extend(reviews)
            
            cursor = data.get('cursor')
            if not cursor or cursor == params['cursor']: break
            params['cursor'] = cursor
            
            time.sleep(0.2) 
        except: break
            
    return raw_reviews, query_summary

def balance_reviews_by_dynamic_playtime(reviews, target_count):
    """
    Dynamically calculates Short/Medium/Long thresholds based on the 
    actual playtime distribution of the collected reviews for THIS game.
    """
    if not reviews:
        return []

    # 1. Extract Playtimes
    # API returns playtime in minutes.
    playtimes = [r.get("author", {}).get("playtime_at_review", 0) for r in reviews]
    
    if not playtimes:
        return reviews[:target_count]

    # 2. Calculate Dynamic Thresholds (Quantiles)
    # Low Threshold: 33rd Percentile
    # High Threshold: 66th Percentile
    try:
        s = pd.Series(playtimes)
        q_low = s.quantile(0.33)
        q_high = s.quantile(0.66)
    except:
        # Fallback if pandas fails or list is empty
        q_low = 120  # 2 hours
        q_high = 600 # 10 hours

    # 3. Bucket the reviews using dynamic thresholds
    buckets = {
        "short": [],  # Bottom 33%
        "medium": [], # Middle 33%
        "long": []    # Top 33%
    }

    for r in reviews:
        minutes = r.get("author", {}).get("playtime_at_review", 0)
        
        if minutes <= q_low:
            buckets["short"].append(r)
        elif minutes <= q_high:
            buckets["medium"].append(r)
        else:
            buckets["long"].append(r)
            
    # 4. Stratified Sampling
    per_bucket_target = math.ceil(target_count / 3)
    final_selection = []
    
    # Round 1: Take up to target from each bucket
    for key in ["short", "medium", "long"]:
        random.shuffle(buckets[key])
        taken = buckets[key][:per_bucket_target]
        final_selection.extend(taken)
        buckets[key] = buckets[key][per_bucket_target:] # Remove taken
    
    # Round 2: Fill deficits from remaining pool (if any bucket was empty/small)
    current_count = len(final_selection)
    still_needed = target_count - current_count
    
    if still_needed > 0:
        leftover_pool = buckets["short"] + buckets["medium"] + buckets["long"]
        random.shuffle(leftover_pool)
        final_selection.extend(leftover_pool[:still_needed])
        
    print_log(f"      [Dynamic Stats] Short < {q_low/60:.1f}h | {q_low/60:.1f}h < Med < {q_high/60:.1f}h | Long > {q_high/60:.1f}h")
    
    return final_selection[:target_count]

def get_app_reviews_balanced(appid):
    """
    Main controller for fetching and balancing reviews.
    Target: 500 Positive, 500 Negative (Balanced by Dynamic Playtime).
    """
    TARGET_FINAL = 1000
    OVERSAMPLE_LIMIT = 1500 
    
    print_log(f"  -> Fetching & Balancing reviews for AppID: {appid}...")

    # 1. Fetch Oversampled Batches
    raw_pos, pos_summary = fetch_raw_reviews(appid, 'positive', OVERSAMPLE_LIMIT)
    raw_neg, neg_summary = fetch_raw_reviews(appid, 'negative', OVERSAMPLE_LIMIT)
    
    # 2. Apply Dynamic Playtime Balancing
    balanced_pos = balance_reviews_by_dynamic_playtime(raw_pos, TARGET_FINAL)
    balanced_neg = balance_reviews_by_dynamic_playtime(raw_neg, TARGET_FINAL)
    
    combined = balanced_pos + balanced_neg
    final_summary = pos_summary if pos_summary else neg_summary
    
    print_log(f"     Raw Fetched: {len(raw_pos)} Pos / {len(raw_neg)} Neg")
    print_log(f"     Balanced Final: {len(balanced_pos)} Pos / {len(balanced_neg)} Neg")
    
    return {'query_summary': final_summary, 'reviews': combined}

# ==========================================
# Flattening Data
# ==========================================

def flatten_game_data(item, genre_name):
    appid = item.get("appid")
    details = item.get("appdetail", {}).get("data", {})
    query_summary = item.get("reviews_data", {}).get("query_summary", {})

    genres_list = details.get("genres", [])
    genres_str = ", ".join([g["description"] for g in genres_list]) if genres_list else None

    return {
        "appid": appid,
        "primary_genre_query": genre_name,
        "name": item.get("name"),
        "type": details.get("type"),
        "release_date": details.get("release_date", {}).get("date"),
        "is_free": details.get("is_free"),
        "developers": ", ".join(details.get("developers", [])) if details.get("developers") else None,
        "publishers": ", ".join(details.get("publishers", [])) if details.get("publishers") else None,
        "genres": genres_str,
        "short_description": details.get("short_description"),
        "num_reviews_in_query": query_summary.get("num_reviews"),
        "review_score": query_summary.get("review_score"),
        "review_score_desc": query_summary.get("review_score_desc"),
        "total_positive": query_summary.get("total_positive"),
        "total_negative": query_summary.get("total_negative"),
        "total_reviews": query_summary.get("total_reviews"),
    }

def flatten_reviews_data(item):
    reviews = item.get("reviews_data", {}).get("reviews", [])
    processed = []
    
    for r in reviews:
        if r.get("language") != "english": continue
        author = r.get("author", {})
        processed.append({
            "appid": item.get("appid"),
            "recommendationid": r.get("recommendationid"),
            "review_text": clean_html(r.get("review")),
            "voted_up": r.get("voted_up"),
            "timestamp_created": r.get("timestamp_created"),
            "votes_up": r.get("votes_up"),
            "votes_funny": r.get("votes_funny"),
            "weighted_vote_score": r.get("weighted_vote_score"),
            "comment_count": r.get("comment_count"),
            "playtime_at_review_minutes": author.get("playtime_at_review"),
            "playtime_forever_minutes": author.get("playtime_forever"),
            "steamid": author.get("steamid")
        })
    return processed

# ==========================================
# Main Execution
# ==========================================

execute_datetime = datetime.now()
base_path = Path(f"steam_data_{execute_datetime.strftime('%Y%m%d')}")
base_path.mkdir(exist_ok=True)
reviews_path = base_path / "reviews"
reviews_path.mkdir(exist_ok=True)

processed_ids = load_processed_ids()
print_log(f"Loaded {len(processed_ids)} processed AppIDs.")

GENRE_TAG_MAP = {
    "sports": 701, "horror": 1667, "science_fiction": 3942,
    "exploration_open_world": 1695, "anime": 4085, "survival": 1662,
    "action_fps": 1663, "hidden_object": 1738, "rpg_action": 4231,
    "casual": 597, "puzzle_matching": 1664, "visual_novel": 3799
}

games_list_accumulated = []

for genre_name, tag_id in GENRE_TAG_MAP.items():
    print_log(f"--- Processing Genre: {genre_name} ---")
    
    genre_dir = reviews_path / genre_name
    genre_dir.mkdir(exist_ok=True)
    
    # 1. Get TOTAL count of games in this genre
    dummy_params = {"tags": tag_id, "category1": 998, "json": 1, "count": 1}
    dummy_res = get_search_results(dummy_params)
    total_games = dummy_res.get("total_count", 1000)
    
    print_log(f"    Total games in catalog for {genre_name}: {total_games}")
    
    games_collected = 0
    TARGET = 10
    attempts = 0
    MAX_ATTEMPTS = 50 
    
    while games_collected < TARGET and attempts < MAX_ATTEMPTS:
        attempts += 1
        
        # 2. GENERATE RANDOM OFFSET
        random_start = random.randint(0, max(0, total_games - 50))
        
        search_params = {
            "tags": tag_id,
            "category1": 998, 
            "start": random_start,
            "count": 1, 
            "json": 1
        }
        
        res = get_search_results(search_params)
        items = res.get("items", [])
        
        if not items: continue
        
        item = items[0]
        
        try:
            logo = item.get("logo", "")
            if "steam/" in logo:
                appid = re.search(r"steam/\w+/(\d+)", logo).group(1)
            else: appid = None
        except: appid = None

        if not appid or str(appid) in processed_ids:
            continue
            
        print_log(f"  [{games_collected+1}/{TARGET}] Found Random Game: {item.get('name')} (ID: {appid}) at offset {random_start}")
        
        # 3. Process Game
        item["appid"] = appid
        item["appdetail"] = get_app_details(appid)
        
        # Fetch Balanced Reviews (Dynamic)
        item["reviews_data"] = get_app_reviews_balanced(appid)
        
        # Flatten Metadata
        game_record = flatten_game_data(item, genre_name)
        games_list_accumulated.append(game_record)
        
        # Save Reviews
        review_records = flatten_reviews_data(item)
        if review_records:
            df_reviews = pd.DataFrame(review_records)
            df_reviews.to_csv(genre_dir / f"reviews_{appid}.csv", index=False, encoding="utf-8-sig")
            print_log(f"    -> Saved {len(df_reviews)} balanced reviews.")
        else:
            print_log("    -> No reviews available.")
            
        save_processed_id(appid)
        processed_ids.add(str(appid))
        games_collected += 1

if games_list_accumulated:
    df_games = pd.DataFrame(games_list_accumulated)
    df_games.to_csv(base_path / f"games_metadata_{execute_datetime.strftime('%Y%m%d')}.csv", index=False, encoding="utf-8-sig")

print_log("Scraping completed.")