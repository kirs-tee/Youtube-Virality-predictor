import os
import time
import json
import random
import re
from datetime import datetime, timezone
from dateutil import parser as dparser
from datetime import datetime, timezone, timedelta

from tqdm import tqdm
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from langdetect import detect, DetectorFactory
DetectorFactory.seed = 0



# CONFIG
#API_KEY = os.getenv("YOUTUBE_API_KEY", "AIzaSyCP7q6V1sABovZhUcrZRLEEErSY7PdRR1E")

#ACTUAL API KEYS ARE HIDDEN WHEN UPLOADING TO GITHUB (to follow YouTube API guidelines)
API_KEYS = [
    "YOUTUBE_API_KEY_1",
    "YOUTUBE_API_KEY_2",
    "YOUTUBE_API_KEY_3",
]
#decided to use mulitple keys to extract more data

current_key_index = 0
youtube = None

OUTPUT_JSON = "youtube_gaming_dataset.json"
THUMBNAIL_DIR = "thumbnails"

VIDEOS_PER_SEARCH_PAGE = 10
CATEGORY_ID = "20"
REGIONCODE = "US"
RELEVANCE_LANGUAGE = "en"

MIN_SUBS = 50_000
MAX_SUBS = 500_000


# ODAY ONLY (UTC)
today = datetime.now(timezone.utc).date() 
PUBLISHED_AFTER = f"{today}T00:00:00Z" 
PUBLISHED_BEFORE = f"{today}T23:59:59Z"

QUERY_POOL = [
    "Minecraft", "Terraria", "Stardew Valley", "Subnautica", "No Man's Sky",
    "Skyrim", "Fallout 4", "Fallout New Vegas", "The Witcher 3 Wild Hunt",
    "Elden Ring", "Baldur's Gate 3", "Cyberpunk 2077", "Dragon Age Dreadwolf",
    "Mass Effect Legendary Edition", "Starfield",
    "Dark Souls", "Dark Souls III", "Bloodborne", "Sekiro Shadows Die Twice",
    "Hollow Knight", "Hades", "Returnal",
    "Call of Duty Warzone", "Call of Duty Modern Warfare", "Valorant",
    "Counter Strike 2", "Apex Legends", "Overwatch 2",
    "Rainbow Six Siege", "Battlefield 1", "Battlefield V", "Battlefield 2042",
    "Halo Infinite",
    "Fortnite", "PUBG", "Dead by Daylight", "World of Warcraft",
    "God of War", "God of War Ragnarok", "The Last of Us",
    "The Last of Us Part II", "Horizon Forbidden West", "Ghost of Tsushima",
    "Death Stranding", "Marvel’s Spider-Man 2",
    "Super Mario Odyssey", "Mario Kart 8 Deluxe", "Super Smash Bros Ultimate",
    "The Legend of Zelda Breath of the Wild",
    "The Legend of Zelda Tears of the Kingdom",
    "Animal Crossing New Horizons", "Splatoon 3",
    "Pikmin 4", "Metroid Prime Remastered",
    "Resident Evil 4", "Resident Evil Village", "Alan Wake 2",
    "Five Nights at Freddy's", "Darkest Dungeon II",
    "Undertale", "Celeste", "Cuphead", "Disco Elysium",
    "FIFA 24", "NBA 2K24", "Madden NFL 24", "Forza Horizon 5",
    "Final Fantasy VII Remake", "Final Fantasy XIV",
    "Final Fantasy X", "Final Fantasy X-2",
    "Persona 5", "Persona 3 Reload",
    "Kingdom Hearts III",
    "Street Fighter 6", "Tekken 8"

    "MindsEye", "Tamagotchi Plaza", "Ambulance Life",
    "Scar-Lead Salvation", "Captain Blood",
    "Neptunia Riders VS Dogoos",
    "Hunter x Hunter Nen x Impact",
    "Bubsy In The Purrfect Collection",
    "Nintendo Switch 2 Welcome Tour",
    "Fast & Furious Arcade Edition",
    "Hotel Barcelona",
    "Game of Thrones Kingsroad",
    "Star Wars Beyond Victory",
    "Tales of the Shire A The Lord of the Rings Game"
]


#youtube = build("youtube", "v3", developerKey=API_KEY)
query_cycle = []



# HELPERS
def safe_api_request(request):
    global current_key_index

    try:
        return request.execute()

    except HttpError as e:
        error_reason = ""
        try:
            error_reason = e.error_details[0]["reason"]
        except:
            pass

        if e.resp.status == 403 and "quota" in str(e).lower():
            print("Quota exhausted, switching API key...")
            current_key_index += 1
            build_youtube_client()
            return safe_api_request(request)

        print(f"HttpError {e.resp.status}: {e}")
        raise

    except Exception as e:
        print(f"Error: {e}")
        raise


def build_youtube_client():
    global current_key_index, youtube

    if current_key_index >= len(API_KEYS):
        raise RuntimeError("All API keys exhausted")

    key = API_KEYS[current_key_index]
    print(f"Using API key #{current_key_index + 1}")
    youtube = build("youtube", "v3", developerKey=key)



def get_next_query():
    global query_cycle
    if not query_cycle:
        query_cycle = QUERY_POOL.copy()
        random.shuffle(query_cycle)
    return query_cycle.pop()


def is_english(text):
    if not text:
        return False
    try:
        return detect(text) == "en"
    except:
        return False


def iso8601_to_seconds(duration):
    match = re.match(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", duration or "")
    if not match:
        return 0
    h = int(match.group(1) or 0)
    m = int(match.group(2) or 0)
    s = int(match.group(3) or 0)
    return h * 3600 + m * 60 + s


def clean_description(text):
    if not text:
        return ""
    text = re.sub(r"\S+\.(com|co\.uk|org|net|gg|io|edu|gov)\S*", "", text, flags=re.I)
    text = re.sub(r"(discord|instagram|twitter|tiktok)\s*[:\-]?\s*\S+", "", text, flags=re.I)
    text = re.sub(r"@\S+", "", text)
    text = re.sub(r"#\w+", "", text)
    return re.sub(r"\s+", " ", text).strip()


def extract_hashtags(text):
    return re.findall(r"#\w+", text or "")


def load_json():
    if not os.path.exists(OUTPUT_JSON):
        return []
    with open(OUTPUT_JSON, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data):
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# COLLECT IDS
def collect_video_ids_by_category(max_videos):
    collected = []
    seen = set()
    pbar = tqdm(total=max_videos, desc="Collecting videos")

    while len(collected) < max_videos:
        q_term = get_next_query()

        # ---- PAGE 1 (always) ----
        request = youtube.search().list(
            part="snippet,id",
            type="video",
            videoCategoryId=CATEGORY_ID,
            q=q_term,
            maxResults=VIDEOS_PER_SEARCH_PAGE,
            regionCode=REGIONCODE,
            relevanceLanguage=RELEVANCE_LANGUAGE,
            publishedAfter=PUBLISHED_AFTER,
            publishedBefore=PUBLISHED_BEFORE,
            videoDuration="medium",
            order="relevance"
        )

        resp = safe_api_request(request)
        pages = [resp]

        # ---- MAYBE PAGE 2 (randomised) ----
        next_token = resp.get("nextPageToken")
        if next_token and random.random() < 0.4:
            request_page2 = youtube.search().list(
                part="snippet,id",
                type="video",
                videoCategoryId=CATEGORY_ID,
                q=q_term,
                maxResults=VIDEOS_PER_SEARCH_PAGE,
                regionCode=REGIONCODE,
                relevanceLanguage=RELEVANCE_LANGUAGE,
                publishedAfter=PUBLISHED_AFTER,
                publishedBefore=PUBLISHED_BEFORE,
                videoDuration="medium",
                order="relevance",
                pageToken=next_token
            )
            pages.append(safe_api_request(request_page2))

        # PROCESS RESULTS
        for page in pages:
            for it in page.get("items", []):
                vid = it["id"].get("videoId")
                sn = it["snippet"]
                title = sn.get("title", "").lower()

                if not vid or vid in seen:
                    continue
                if "live" in title or "stream" in title:
                    continue
                if sn.get("liveBroadcastContent") != "none":
                    continue
                if not is_english(sn.get("title")):
                    continue

                collected.append({
                    "videoId": vid,
                    "searchQuery": q_term
                })
                seen.add(vid)
                pbar.update(1)

                if len(collected) >= max_videos:
                    break

            if len(collected) >= max_videos:
                break

        time.sleep(0.15)

    pbar.close()
    return collected



#  FETCH DETAILS

def fetch_video_details(entries):
    videos = []

    for e in entries:
        req = youtube.videos().list(
            part="snippet,contentDetails,statistics",
            id=e["videoId"]
        )
        resp = safe_api_request(req)

        for it in resp.get("items", []):
            it["durationSeconds"] = iso8601_to_seconds(
                it["contentDetails"].get("duration")
            )
            it["searchQuery"] = e["searchQuery"]
            videos.append(it)

    return videos


# BUILD ROWS
def build_dataset_rows(videos, sub_map):
    rows = []

    for v in videos:
        sn = v["snippet"]
        stats = v.get("statistics", {})
        cid = sn["channelId"]
        subs = sub_map.get(cid, 0)

        if not (MIN_SUBS <= subs <= MAX_SUBS):
            continue

        thumb = next(iter(sn.get("thumbnails", {}).values()), {}).get("url")

        rows.append({
            "videoId": v["id"],
            "searchQuery": v["searchQuery"],
            "title": sn.get("title"),
            "description": clean_description(sn.get("description")),
            "hashtags": ";".join(sorted(set(
                extract_hashtags(sn.get("title")) +
                extract_hashtags(sn.get("description"))
            ))),
            "tags": "|".join(sn.get("tags", [])),
            "channelId": cid,
            "channelTitle": sn.get("channelTitle"),
            "channelSubscriberCount": subs,
            "publishedAt": dparser.parse(sn.get("publishedAt")).isoformat(),
            "categoryId": CATEGORY_ID,
            "durationSeconds": v.get("durationSeconds", 0),

            "viewCountDay1": int(stats.get("viewCount", 0)),
            "likeCountDay1": int(stats.get("likeCount", 0)),
            "commentCountDay1": int(stats.get("commentCount", 0)),

            "viewCountDay7": None,
            "likeCountDay7": None,
            "commentCountDay7": None,

            "viewCountDay30": None,
            "likeCountDay30": None,
            "commentCountDay30": None,

            "thumbnailUrl": thumb,
            "thumbnailFilename": None
        })

    return rows



# DAYS UPDATE
def update_day7_metrics():
    data = load_json()
    today_utc = datetime.now(timezone.utc)
    cutoff = today_utc.timestamp() - 7 * 86400

    eligible = [
        v for v in data
        if v["viewCountDay7"] is None
        and dparser.parse(v["publishedAt"]).timestamp() <= cutoff
    ]

    if not eligible:
        print("No videos eligible for Day 7 update.")
        return

    print(f"Updating Day 7 metrics for {len(eligible)} videos")

    for i in tqdm(range(0, len(eligible), 50)):
        batch = eligible[i:i + 50]
        ids = [v["videoId"] for v in batch]

        req = youtube.videos().list(
            part="statistics",
            id=",".join(ids)
        )
        resp = safe_api_request(req)

        stats_map = {v["id"]: v["statistics"] for v in resp.get("items", [])}

        for v in data:
            if v["videoId"] in stats_map:
                s = stats_map[v["videoId"]]
                v["viewCountDay7"] = int(s.get("viewCount", 0))
                v["likeCountDay7"] = int(s.get("likeCount", 0))
                v["commentCountDay7"] = int(s.get("commentCount", 0))

        time.sleep(0.1)

    save_json(data)
    print("Day 7 metrics updated ")
    

def update_day30_metrics():
    data = load_json()
    today_utc = datetime.now(timezone.utc)
    cutoff = today_utc.timestamp() - 30 * 86400

    eligible = [
        v for v in data
        if v.get("viewCountDay30") is None
        and dparser.parse(v["publishedAt"]).timestamp() <= cutoff
    ]

    if not eligible:
        print("No videos eligible for Day 30 update.")
        return

    print(f"Updating Day 30 metrics for {len(eligible)} videos")

    for i in tqdm(range(0, len(eligible), 50)):
        batch = eligible[i:i + 50]
        ids = [v["videoId"] for v in batch]

        req = youtube.videos().list(
            part="statistics",
            id=",".join(ids)
        )
        resp = safe_api_request(req)

        stats_map = {v["id"]: v["statistics"] for v in resp.get("items", [])}

        for v in data:
            if v["videoId"] in stats_map:
                s = stats_map[v["videoId"]]
                v["viewCountDay30"] = int(s.get("viewCount", 0))
                v["likeCountDay30"] = int(s.get("likeCount", 0))
                v["commentCountDay30"] = int(s.get("commentCount", 0))

        time.sleep(0.1)

    save_json(data)
    print("Day 30 metrics updated ")


def count_existing_entries(path):
    if not os.path.exists(path):
        return 0
    with open(path, "r", encoding="utf-8") as f:
        return len(json.load(f))


# MAIN

def main():
    os.makedirs(THUMBNAIL_DIR, exist_ok=True)

    while True:
        try:
            existing = load_json()
            existing_ids = {v["videoId"] for v in existing}

            ids = collect_video_ids_by_category(50)
            videos = fetch_video_details(ids)

            channel_ids = list({v["snippet"]["channelId"] for v in videos})
            resp = safe_api_request(
                youtube.channels().list(
                    part="statistics",
                    id=",".join(channel_ids)
                )
            )

            sub_map = {
                c["id"]: int(c["statistics"].get("subscriberCount", 0))
                for c in resp.get("items", [])
            }

            new_rows = build_dataset_rows(videos, sub_map)
            new_rows = [r for r in new_rows if r["videoId"] not in existing_ids]

            if not new_rows:
                print("No new videos added this batch.")
            else:
                save_json(existing + new_rows)
                print(f"Added {len(new_rows)} videos ")

        except RuntimeError:
            print("All API keys exhausted. Stopping.")
            break



if __name__ == "__main__":
    build_youtube_client()
    update_day7_metrics()
    update_day30_metrics()
    main()

    print("Existing entries:", count_existing_entries(OUTPUT_JSON))