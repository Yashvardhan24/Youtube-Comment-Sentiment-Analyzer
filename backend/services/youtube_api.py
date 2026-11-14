import os
import re

from pathlib import Path
from urllib.parse import urlparse, parse_qs

from dotenv import load_dotenv
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError


ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")


def extract_video_id(url):
    parsed = urlparse(url)
    video_id = None

    if parsed.hostname in ["youtu.be", "www.youtu.be"]:
        video_id = parsed.path.lstrip("/")

    elif parsed.hostname in [
        "youtube.com", "www.youtube.com", "m.youtube.com"
    ]:
        if parsed.path == "/watch":
            video_id = parse_qs(parsed.query).get("v", [None])[0]

        else:
            match = re.match(
                r"/(shorts|embed|live)/([^/?]+)",
                parsed.path
            )
            if match:
                video_id = match.group(2)

    if not video_id or not re.fullmatch(r"[A-Za-z0-9_-]{11}", video_id):
        raise ValueError("Invalid YouTube video URL.")

    return video_id


def fetch_video_comments(video_url, max_comments=1000):

    api_key = os.getenv("YOUTUBE_API_KEY")

    if not api_key:
        raise ValueError("YouTube API key not configured.")

    video_id = extract_video_id(video_url)

    youtube = build(
        "youtube",
        "v3",
        developerKey=api_key
    )

    try:
        # Fetch video information
        video_response = youtube.videos().list(
            part="snippet",
            id=video_id
        ).execute()

        if not video_response.get("items"):
            raise ValueError("Video not found or unavailable.")

        video = video_response["items"][0]["snippet"]

        channel_id = video["channelId"]

        comments = []
        next_page_token = None

        # Fetch comments using pagination
        while len(comments) < max_comments:

            remaining = max_comments - len(comments)

            response = youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                maxResults=min(100, remaining),
                pageToken=next_page_token,
                textFormat="plainText"
            ).execute()

            for item in response.get("items", []):

                snippet = item["snippet"]["topLevelComment"]["snippet"]

                # Exclude video uploader's comments
                author_channel = snippet.get(
                    "authorChannelId", {}
                ).get("value")

                if author_channel == channel_id:
                    continue

                comments.append({
                    "text": snippet["textDisplay"],
                    "author": snippet["authorDisplayName"],
                    "published_at": snippet["publishedAt"],
                    "likes": snippet["likeCount"]
                })

                if len(comments) >= max_comments:
                    break

            next_page_token = response.get("nextPageToken")

            if not next_page_token:
                break

        return {
            "video_id": video_id,
            "title": video["title"],
            "channel": video["channelTitle"],
            "comments": comments,
            "total_fetched": len(comments)
        }

    except HttpError as e:
        raise RuntimeError(f"YouTube API error: {e}")