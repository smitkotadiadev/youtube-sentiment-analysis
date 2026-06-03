# YouTube Comment Sentiment Analysis

A standalone Python script that searches YouTube videos by keyword, collects
exactly **100 public top-level comments**, runs **sentiment analysis** on each
comment (using NLTK VADER), saves the results to a CSV file, and generates a
**bar chart** and a **pie chart** of the sentiment distribution.

It uses the **official YouTube Data API v3** (not browser scraping).

---

## Features

- Searches videos by keyword via the YouTube Data API v3 `search.list` endpoint.
- Collects exactly 100 unique top-level comments via `commentThreads.list`.
- Skips replies and deduplicates comments by `comment_id`.
- Sentiment analysis with NLTK VADER (great for short social-media text).
- Quota-aware and rate-conscious: small video search, paginate comments only as
  needed, request timeouts, short delays, and a hard stop at 100 comments.
- Outputs a CSV plus two `.jpg` charts (matplotlib only — no seaborn).

---

## Requirements

- Python 3.8 or newer
- The following Python packages:
  - `requests`
  - `pandas`
  - `nltk`
  - `matplotlib`
  - `python-dotenv`

---

## 1. Install dependencies

From the project folder, run:

```bash
pip install requests pandas nltk matplotlib python-dotenv
```

Or use the included requirements file:

```bash
pip install -r requirements.txt
```

> The NLTK VADER lexicon downloads automatically the first time you run the
> script, so no manual NLTK setup is needed.

---

## 2. Get a YouTube Data API v3 key

1. Go to the [Google Cloud Console](https://console.cloud.google.com/).
2. Create a new project (or select an existing one).
3. Open **APIs & Services → Library**.
4. Search for **"YouTube Data API v3"** and click **Enable**.
5. Open **APIs & Services → Credentials**.
6. Click **Create Credentials → API key**.
7. Copy the generated API key.

---

## 3. Add your API key to a `.env` file

In the project folder, create a file named exactly `.env`:

```
YOUTUBE_API_KEY=YOUR_API_KEY_HERE
```

There is an `.env.example` you can copy:

```bash
# macOS / Linux
cp .env.example .env

# Windows (Command Prompt)
copy .env.example .env
```

Then edit `.env` and replace `YOUR_API_KEY_HERE` with your real key.

> Alternatively, you can set a real environment variable named `YOUTUBE_API_KEY`
> instead of using a `.env` file.

---

## 4. Run the script

Pass the topic with `--keyword`:

```bash
python youtube_sentiment_analysis.py --keyword "iPhone 15 review"
```

More examples:

```bash
python youtube_sentiment_analysis.py --keyword "Tesla Model 3 review"
python youtube_sentiment_analysis.py --keyword "ChatGPT review"
python youtube_sentiment_analysis.py --keyword "Oppenheimer review"
```

---

## Output files

All files are created in the current folder:

| File | Description |
| --- | --- |
| `youtube_sentiment_results.csv` | All collected comments with sentiment scores and labels |
| `sentiment_bar_chart.jpg` | Bar chart of sentiment distribution |
| `sentiment_pie_chart.jpg` | Pie chart of sentiment share |

### CSV columns

`video_id`, `video_title`, `comment_id`, `comment_text`,
`author_display_name`, `like_count`, `published_at`, `updated_at`,
`sentiment_score`, `sentiment_label`

### Sentiment rules

The VADER `compound` score (range `-1` to `+1`) is labeled as:

- `compound >= 0.05` → **positive**
- `compound <= -0.05` → **negative**
- otherwise → **neutral**

---

## Console summary

After a run, the script prints a short summary:

- Keyword used
- Total comments collected
- Positive / neutral / negative counts
- Path to the CSV file
- Paths to the bar and pie chart files

---

## Troubleshooting

| Problem | Likely cause / fix |
| --- | --- |
| `No YouTube API key found` | Create a `.env` file with `YOUTUBE_API_KEY=...`, or set the environment variable. |
| `A non-empty keyword is required` | Pass `--keyword "your topic"`. |
| `YouTube API quota exceeded` | Daily quota used up; wait until it resets or use a different key. |
| Fewer than 100 comments collected | Not enough comments exist for the keyword; the script continues with what it found. |
| Comments are skipped per video | Comments may be disabled on that video; the script moves to the next one. |

---

## Notes on API quota

- `search.list` is **expensive** (about 100 quota units per call), so the script
  fetches only a small number of relevant videos (default: 10).
- `commentThreads.list` is **cheap** (about 1 unit per call), so comments are
  paginated only as needed until 100 are collected.
