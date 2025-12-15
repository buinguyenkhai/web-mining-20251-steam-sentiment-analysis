import time
import requests
import csv
from datetime import datetime, timedelta
from pathlib import Path
import math

# ==========================================
# Configuration
# ==========================================
INPUT_FILE = Path("processed_appids.txt")
OUTPUT_FILE = Path("player_count_history.csv")
INTERVAL_MINUTES = 10

API_URL = "https://api.steampowered.com/ISteamUserStats/GetNumberOfCurrentPlayers/v1/"

def load_app_ids():
    if not INPUT_FILE.exists():
        print(f"Error: {INPUT_FILE} not found!")
        return []
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

def get_current_players(appid):
    params = {'appid': appid}
    MAX_RETRIES = 3
    
    for attempt in range(MAX_RETRIES):
        try:
            # Try to connect
            response = requests.get(API_URL, params=params, timeout=5)
            
            # If successful, return data immediately
            if response.status_code == 200:
                data = response.json()
                return data.get('response', {}).get('player_count', 0)
            
            # If server returns 500/503 (Server Error), we should retry
            # If server returns 404 (Not Found), retrying won't help, so break
            if response.status_code in [404, 403]:
                break
                
        except requests.exceptions.RequestException:
            # Catch network errors (Timeout, ConnectionError, etc.)
            pass
        
        # If we are here, it failed. Wait briefly before retrying.
        # Backoff: Wait 1s after 1st fail, 2s after 2nd fail...
        time.sleep(1 + attempt)

    # If all 3 attempts failed, return -1 (Error Code)
    print(f"    [!] Failed to fetch AppID {appid} after {MAX_RETRIES} attempts.")
    return -1

def get_next_rounded_time(interval_minutes):
    """
    Calculates the next future timestamp that is a perfect multiple of 'interval_minutes'.
    Example: If now is 18:03, and interval is 10, returns 18:10.
    """
    now = datetime.now()
    # Calculate how many minutes past the hour we are
    minute = now.minute
    # Calculate next interval mark
    next_minute = (math.ceil(minute / interval_minutes) * interval_minutes)
    
    # Handle hour rollover (e.g. if next_minute is 60)
    if next_minute == 60:
        next_time = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
    elif next_minute == minute: 
        # If we are exactly on the mark (e.g. 18:10:00), schedule for 18:20
        next_time = now.replace(minute=minute, second=0, microsecond=0) + timedelta(minutes=interval_minutes)
    else:
        next_time = now.replace(minute=next_minute, second=0, microsecond=0)
        
    return next_time

def run_monitoring_cycle(target_timestamp_str):
    """
    Runs the scrape, applying 'target_timestamp_str' to ALL records.
    """
    app_ids = load_app_ids()
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Starting cycle for Timestamp: {target_timestamp_str}")
    
    results = []
    
    for i, appid in enumerate(app_ids):
        count = get_current_players(appid)
        if count is None:
            count = -1
        if count is not None:
            results.append({
                "timestamp": target_timestamp_str, # Uniform Timestamp
                "appid": appid,
                "player_count": count
            })
            
        # Small delay to respect API (20 requests/sec)
        time.sleep(0.05) 

    # Save to CSV
    file_exists = OUTPUT_FILE.exists()
    if results:
        with open(OUTPUT_FILE, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["timestamp", "appid", "player_count"])
            if not file_exists:
                writer.writeheader()
            writer.writerows(results)
            
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Finished cycle. Saved {len(results)} records.")

# ==========================================
# Main Loop
# ==========================================
if __name__ == "__main__":
    print(f"Starting Aligned Monitor (Every {INTERVAL_MINUTES} mins)")
    
    while True:
        # 1. Calculate when the next :10, :20, :30 mark is
        next_run_time = get_next_rounded_time(INTERVAL_MINUTES)
        wait_seconds = (next_run_time - datetime.now()).total_seconds()
        
        # Format the timestamp that will be written to the CSV (e.g. "2023-10-27 18:10:00")
        cycle_timestamp_str = next_run_time.strftime("%Y-%m-%d %H:%M:%S")
        
        print(f"Next cycle scheduled for: {cycle_timestamp_str} (Sleeping {wait_seconds:.1f}s)...")
        
        # 2. Wait until that exact moment
        if wait_seconds > 0:
            time.sleep(wait_seconds)
            
        # 3. Run the job
        run_monitoring_cycle(cycle_timestamp_str)