"""
================================================================================
YouTube Comment Sentiment Analysis
================================================================================

This standalone script searches YouTube videos by keyword, collects exactly 100
public top-level comments, runs sentiment analysis on each comment (using NLTK
VADER), saves the results to a CSV file, and generates a bar chart and a pie
chart showing the sentiment distribution.

It uses the official YouTube Data API v3 (NOT browser scraping).

--------------------------------------------------------------------------------
SETUP INSTRUCTIONS
--------------------------------------------------------------------------------

1) Install the required dependencies:

       pip install requests pandas nltk matplotlib python-dotenv

2) Create a YouTube Data API v3 key:
   - Go to the Google Cloud Console:   https://console.cloud.google.com/
   - Create a new project (or select an existing one).
   - In the left menu, open "APIs & Services" > "Library".
   - Search for "YouTube Data API v3" and click "Enable".
   - Go to "APIs & Services" > "Credentials".
   - Click "Create Credentials" > "API key".
   - Copy the generated API key.

3) Place the API key in a .env file:
   - In the same folder as this script, create a file named exactly:  .env
   - Add the following line to it (no quotes, no spaces around the = sign):

         YOUTUBE_API_KEY=YOUR_API_KEY_HERE

   - Alternatively, you can set it as a real environment variable named
     YOUTUBE_API_KEY instead of using a .env file.

4) Run the script (pass the topic with --keyword):

       python youtube_sentiment_analysis.py --keyword "iPhone 15 review"

   More examples:

       python youtube_sentiment_analysis.py --keyword "Tesla Model 3 review"
       python youtube_sentiment_analysis.py --keyword "ChatGPT review"
       python youtube_sentiment_analysis.py --keyword "Oppenheimer review"

--------------------------------------------------------------------------------
OUTPUT FILES (created in the current folder)
--------------------------------------------------------------------------------
   - youtube_sentiment_results.csv   (all collected comments + sentiment)
   - sentiment_bar_chart.jpg         (bar chart of sentiment distribution)
   - sentiment_pie_chart.jpg         (pie chart of sentiment distribution)
================================================================================
"""

# --- Standard library imports -------------------------------------------------
import argparse  # to read the --keyword command-line argument
import os        # to read the API key from environment variables
import re        # to clean up comment text (whitespace / line breaks)
import sys       # to exit cleanly with an error code on failures
import time      # to add short, polite delays between API calls

# --- Third-party imports ------------------------------------------------------
import requests                       # to call the YouTube Data API over HTTP
import pandas as pd                   # to build the results table and save CSV
import matplotlib                     # plotting library (used for the charts)
matplotlib.use("Agg")                 # use a non-interactive backend (no GUI needed)
import matplotlib.pyplot as plt       # the actual plotting interface
import nltk                           # provides the VADER sentiment analyzer
from dotenv import load_dotenv        # loads variables from a .env file
from nltk.sentiment.vader import SentimentIntensityAnalyzer


# ==============================================================================
# CONFIGURATION CONSTANTS
# ==============================================================================

# Base URL for every YouTube Data API v3 endpoint.
YOUTUBE_API_BASE_URL = "https://www.googleapis.com/youtube/v3"

# Exactly how many top-level comments we want to collect in total.
TARGET_COMMENT_COUNT = 100

# How many videos to fetch from the (expensive) search.list endpoint.
# Keeping this small protects our API quota.
MAX_VIDEOS_TO_SEARCH = 10

# How many comments to request per commentThreads.list page (100 is the max).
COMMENTS_PER_PAGE = 100

# Seconds to wait for each HTTP request before giving up.
REQUEST_TIMEOUT_SECONDS = 15

# Short, polite delay between API calls so we don't hammer the API.
DELAY_BETWEEN_CALLS_SECONDS = 0.5

# Output file names.
CSV_OUTPUT_FILE = "youtube_sentiment_results.csv"
BAR_CHART_FILE = "sentiment_bar_chart.jpg"
PIE_CHART_FILE = "sentiment_pie_chart.jpg"

# Sentiment thresholds for the VADER "compound" score.
POSITIVE_THRESHOLD = 0.05
NEGATIVE_THRESHOLD = -0.05


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def get_api_key():
    """
    Read the YouTube API key from the YOUTUBE_API_KEY environment variable
    (loading a .env file first if present). Exit with a clear message if it is
    missing.
    """
    # load_dotenv() reads a local .env file and puts its values into os.environ.
    load_dotenv()

    api_key = os.environ.get("YOUTUBE_API_KEY")

    # Error handling: missing API key.
    if not api_key:
        print(
            "ERROR: No YouTube API key found.\n"
            "Please set the YOUTUBE_API_KEY environment variable, or create a\n"
            ".env file containing the line:\n\n"
            "    YOUTUBE_API_KEY=YOUR_API_KEY_HERE\n"
        )
        sys.exit(1)

    return api_key


def parse_arguments():
    """
    Parse command-line arguments. The keyword/topic is required and is passed
    with --keyword.
    """
    parser = argparse.ArgumentParser(
        description="Search YouTube by keyword, collect 100 comments, and "
                    "analyze their sentiment."
    )
    parser.add_argument(
        "--keyword",
        type=str,
        required=True,
        help='The topic to search for, e.g. --keyword "iPhone 15 review"',
    )
    args = parser.parse_args()

    # Error handling: missing / empty keyword.
    if not args.keyword or not args.keyword.strip():
        print('ERROR: A non-empty keyword is required, e.g. --keyword "ChatGPT review"')
        sys.exit(1)

    return args.keyword.strip()


def ensure_vader_lexicon():
    """
    Make sure the NLTK VADER lexicon is available. Download it automatically
    if it is missing.
    """
    try:
        # If this succeeds, the lexicon is already installed.
        nltk.data.find("sentiment/vader_lexicon.zip")
    except LookupError:
        # Otherwise, download it quietly.
        print("Downloading the NLTK VADER lexicon (first-time setup)...")
        nltk.download("vader_lexicon")


def clean_comment_text(text):
    """
    Clean a comment by collapsing excessive whitespace and removing line breaks,
    so the CSV stays tidy.
    """
    if text is None:
        return ""
    # Replace any run of whitespace (spaces, tabs, newlines) with a single space.
    cleaned = re.sub(r"\s+", " ", text)
    return cleaned.strip()


def youtube_api_get(endpoint, params):
    """
    Make a GET request to a YouTube Data API v3 endpoint and return the parsed
    JSON response. Centralized error handling lives here so every API call is
    protected (timeouts, quota errors, generic API errors, etc.).

    Returns:
        (data, error) tuple. On success, `data` is the JSON dict and `error`
        is None. On failure, `data` is None and `error` is a short string.
    """
    url = f"{YOUTUBE_API_BASE_URL}/{endpoint}"

    try:
        response = requests.get(url, params=params, timeout=REQUEST_TIMEOUT_SECONDS)
    except requests.exceptions.Timeout:
        return None, "Request timed out."
    except requests.exceptions.RequestException as exc:
        return None, f"Network error: {exc}"

    # If the API returned an HTTP error status, try to extract a useful message.
    if response.status_code != 200:
        # Default message.
        message = f"HTTP {response.status_code} error from YouTube API."

        # Try to read the structured error YouTube sends back.
        try:
            error_json = response.json()
            api_error = error_json.get("error", {})
            message = api_error.get("message", message)

            # Quota errors usually appear as reason "quotaExceeded".
            reasons = [
                item.get("reason", "")
                for item in api_error.get("errors", [])
            ]
            if "quotaExceeded" in reasons or "dailyLimitExceeded" in reasons:
                message = ("YouTube API quota exceeded. Try again later or use a "
                           "different API key. Details: " + message)
        except ValueError:
            # Response body was not JSON; keep the default message.
            pass

        return None, message

    # Parse the successful JSON body.
    try:
        return response.json(), None
    except ValueError:
        return None, "Could not parse the API response as JSON."


# ==============================================================================
# YOUTUBE DATA COLLECTION
# ==============================================================================

def search_videos(keyword, api_key):
    """
    Use the (quota-expensive) search.list endpoint to find a small number of
    relevant videos for the given keyword.

    Returns a list of dicts: [{"video_id": ..., "video_title": ...}, ...]
    """
    params = {
        "part": "snippet",
        "q": keyword,
        "type": "video",
        "maxResults": MAX_VIDEOS_TO_SEARCH,
        "order": "relevance",
        "key": api_key,
    }

    data, error = youtube_api_get("search", params)
    if error:
        print(f"ERROR while searching for videos: {error}")
        sys.exit(1)

    videos = []
    for item in data.get("items", []):
        # A search result for a video stores its id under id.videoId.
        video_id = item.get("id", {}).get("videoId")
        title = item.get("snippet", {}).get("title", "")
        if video_id:
            videos.append({"video_id": video_id, "video_title": title})

    # Error handling: no videos found for this keyword.
    if not videos:
        print(f"ERROR: No videos found for keyword '{keyword}'. Try another keyword.")
        sys.exit(1)

    return videos


def collect_comments_for_video(video, api_key, seen_comment_ids, remaining_needed):
    """
    Collect top-level comments for a single video using commentThreads.list.
    Paginates with nextPageToken only as long as more comments are still needed.

    - Skips replies (we only read the top-level comment of each thread).
    - Deduplicates by comment_id using the shared `seen_comment_ids` set.
    - Stops as soon as `remaining_needed` comments have been collected.

    Returns a list of comment dictionaries collected from this video.
    """
    collected = []
    page_token = None

    while len(collected) < remaining_needed:
        params = {
            "part": "snippet",
            "videoId": video["video_id"],
            "maxResults": COMMENTS_PER_PAGE,
            "textFormat": "plainText",
            "order": "relevance",
            "key": api_key,
        }
        # Add the pagination token only when continuing to a later page.
        if page_token:
            params["pageToken"] = page_token

        data, error = youtube_api_get("commentThreads", params)

        # Gracefully handle per-video issues without crashing the whole run.
        if error:
            # Comments disabled / not found errors should just skip this video.
            lowered = error.lower()
            if "disabled" in lowered or "not found" in lowered or "403" in lowered:
                print(f"  Skipping video '{video['video_title']}' ({error})")
            else:
                print(f"  Warning for video '{video['video_title']}': {error}")
            break

        items = data.get("items", [])

        # Error handling: video has no comments at all.
        if not items and not collected:
            print(f"  Video '{video['video_title']}' has no comments. Skipping.")
            break

        for item in items:
            # The top-level comment lives here. We intentionally ignore replies.
            top_comment = (
                item.get("snippet", {})
                    .get("topLevelComment", {})
            )
            comment_id = top_comment.get("id")
            snippet = top_comment.get("snippet", {})

            # Skip if we somehow lack an id, or if it's a duplicate.
            if not comment_id or comment_id in seen_comment_ids:
                continue

            seen_comment_ids.add(comment_id)

            comment_record = {
                "video_id": video["video_id"],
                "video_title": video["video_title"],
                "comment_id": comment_id,
                "comment_text": clean_comment_text(snippet.get("textDisplay", "")),
                "author_display_name": snippet.get("authorDisplayName", ""),
                "like_count": snippet.get("likeCount", 0),
                "published_at": snippet.get("publishedAt", ""),
                "updated_at": snippet.get("updatedAt", ""),
            }
            collected.append(comment_record)

            # Stop immediately once we have enough from this video.
            if len(collected) >= remaining_needed:
                break

        # Decide whether to fetch another page.
        page_token = data.get("nextPageToken")
        if not page_token:
            # No more pages for this video.
            break

        # Polite, rate-conscious short delay before the next page request.
        time.sleep(DELAY_BETWEEN_CALLS_SECONDS)

    return collected


def collect_comments(videos, api_key):
    """
    Loop through the searched videos and collect comments until we reach exactly
    TARGET_COMMENT_COUNT (100), or until we run out of videos.

    Returns a list of up to 100 comment dictionaries.
    """
    all_comments = []
    seen_comment_ids = set()  # used for deduplication across all videos

    for video in videos:
        # Stop the whole process the moment we have 100 comments.
        if len(all_comments) >= TARGET_COMMENT_COUNT:
            break

        remaining_needed = TARGET_COMMENT_COUNT - len(all_comments)
        print(f"Collecting comments from: {video['video_title']}")

        video_comments = collect_comments_for_video(
            video, api_key, seen_comment_ids, remaining_needed
        )
        all_comments.extend(video_comments)

        print(f"  Collected so far: {len(all_comments)}/{TARGET_COMMENT_COUNT}")

        # Polite, short delay before moving on to the next video.
        time.sleep(DELAY_BETWEEN_CALLS_SECONDS)

    # Make absolutely sure we never return more than 100.
    return all_comments[:TARGET_COMMENT_COUNT]


# ==============================================================================
# SENTIMENT ANALYSIS
# ==============================================================================

def analyze_sentiment(comments):
    """
    Run VADER sentiment analysis on each comment and attach:
      - sentiment_score  (the VADER 'compound' score, from -1 to +1)
      - sentiment_label  (positive / neutral / negative)
    """
    analyzer = SentimentIntensityAnalyzer()

    for comment in comments:
        scores = analyzer.polarity_scores(comment["comment_text"])
        compound = scores["compound"]

        # Apply the required threshold rules.
        if compound >= POSITIVE_THRESHOLD:
            label = "positive"
        elif compound <= NEGATIVE_THRESHOLD:
            label = "negative"
        else:
            label = "neutral"

        comment["sentiment_score"] = compound
        comment["sentiment_label"] = label

    return comments


# ==============================================================================
# OUTPUT: CSV AND CHARTS
# ==============================================================================

def save_to_csv(comments):
    """
    Save the collected, analyzed comments to a CSV file with a fixed column
    order. Returns the path of the saved file.
    """
    # Fixed column order so the CSV is predictable and readable.
    columns = [
        "video_id",
        "video_title",
        "comment_id",
        "comment_text",
        "author_display_name",
        "like_count",
        "published_at",
        "updated_at",
        "sentiment_score",
        "sentiment_label",
    ]
    df = pd.DataFrame(comments, columns=columns)
    df.to_csv(CSV_OUTPUT_FILE, index=False, encoding="utf-8-sig")
    return CSV_OUTPUT_FILE


def make_charts(comments):
    """
    Generate a bar chart and a pie chart of the sentiment distribution and save
    them as JPG files.
    """
    # Count how many comments fell into each sentiment label.
    labels = ["positive", "neutral", "negative"]
    counts = {label: 0 for label in labels}
    for comment in comments:
        counts[comment["sentiment_label"]] += 1

    values = [counts[label] for label in labels]
    colors = ["#4CAF50", "#9E9E9E", "#F44336"]  # green, gray, red

    # ---------------- Bar chart ----------------
    plt.figure(figsize=(7, 5))
    plt.bar(labels, values, color=colors)
    plt.title("YouTube Comment Sentiment Distribution")
    plt.xlabel("Sentiment")
    plt.ylabel("Number of Comments")
    # Add the count above each bar for clarity.
    for index, value in enumerate(values):
        plt.text(index, value, str(value), ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(BAR_CHART_FILE, format="jpg", dpi=150)
    plt.close()

    # ---------------- Pie chart ----------------
    # Only include slices that have at least one comment, to avoid empty slices.
    pie_labels = [label for label in labels if counts[label] > 0]
    pie_values = [counts[label] for label in pie_labels]
    pie_colors = [colors[labels.index(label)] for label in pie_labels]

    plt.figure(figsize=(7, 5))
    plt.pie(
        pie_values,
        labels=pie_labels,
        colors=pie_colors,
        autopct="%1.1f%%",
        startangle=90,
    )
    plt.title("YouTube Comment Sentiment Share")
    plt.axis("equal")  # keep the pie a circle
    plt.tight_layout()
    plt.savefig(PIE_CHART_FILE, format="jpg", dpi=150)
    plt.close()

    return counts


# ==============================================================================
# MAIN PROGRAM
# ==============================================================================

def main():
    # 1) Read inputs and credentials.
    keyword = parse_arguments()
    api_key = get_api_key()

    # 2) Make sure the sentiment lexicon is available.
    ensure_vader_lexicon()

    # 3) Search a small number of relevant videos (quota-aware).
    print(f"\nSearching YouTube for: '{keyword}'")
    videos = search_videos(keyword, api_key)
    print(f"Found {len(videos)} videos to scan for comments.\n")

    # 4) Collect up to 100 unique top-level comments.
    comments = collect_comments(videos, api_key)

    # Error handling: fewer than 100 comments available across all videos.
    if not comments:
        print("ERROR: No comments could be collected for this keyword.")
        sys.exit(1)
    if len(comments) < TARGET_COMMENT_COUNT:
        print(
            f"\nNote: Only {len(comments)} comments were available "
            f"(fewer than {TARGET_COMMENT_COUNT}). Continuing with what we have."
        )

    # 5) Run sentiment analysis.
    comments = analyze_sentiment(comments)

    # 6) Save the CSV.
    csv_path = save_to_csv(comments)

    # 7) Build the charts.
    counts = make_charts(comments)

    # 8) Print a short, friendly summary.
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Keyword used:            {keyword}")
    print(f"Total comments collected: {len(comments)}")
    print(f"Positive comments:        {counts['positive']}")
    print(f"Neutral comments:         {counts['neutral']}")
    print(f"Negative comments:        {counts['negative']}")
    print(f"CSV file:                 {os.path.abspath(csv_path)}")
    print(f"Bar chart:                {os.path.abspath(BAR_CHART_FILE)}")
    print(f"Pie chart:                {os.path.abspath(PIE_CHART_FILE)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
