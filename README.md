***What this script does (story_collector.py)***

**Purpose**

Collects posts (and optionally comments) from Reddit subreddits like r/shortscarystories, then stores them into a SQLite database with metadata + raw archives + simple NLP + optional embeddings.

***Two input modes***

Reddit URL (http/https + reddit.com)

**Uses the Reddit JSON endpoints:**

Subreddit listing: https://www.reddit.com/r/<sub>/.json?raw_json=1&limit=50
Comments per post: https://www.reddit.com/<permalink>.json?raw_json=1
Local HTML file (file://...)
Loads HTML from disk
Parses posts using BeautifulSoup, specifically looking for shreddit-post elements
Stores the raw HTML gzipped in the DB

***What it stores (tables)***

articles: the main stories/posts (title, content, author/meta when from API, URLs, timestamps, flags, wordcount)
comments: flattened comment tree from Reddit JSON (comment id, parent id, depth, votes, etc.)
raw_html: gzipped HTML snapshots (if you scraped from an HTML listing)
raw_json: gzipped JSON snapshots (function exists; currently only used if you call it somewhere)
runs: each run’s start/end + totals + error count + script version
nlp_features: basic text stats (wordcount, sentence count, avg sentence length)
embeddings: optional vector (SentenceTransformers MiniLM) stored as JSON
Processing flow (the “harvest ritual”)

**For each run:**
Initialize DB schema (safe if already exists)
Fetch posts (API or HTML)

**For each post:**
Upsert into articles

***Compute & store NLP features***

Optionally compute embeddings if sentence_transformers is installed
If it has a permalink, fetch comments JSON and insert them
Sleep between posts (fixed 7s default, or randomized/custom with --wait)

***Usage patterns***

One-time run:
python story_collector.py https://www.reddit.com/r/shortscarystories/ stories.db

Repeat every N minutes:
python story_collector.py https://www.reddit.com/r/shortscarystories/ stories.db 15

Limit number of posts:
python story_collector.py https://www.reddit.com/r/shortscarystories/ stories.db 0 200

Wait control:
--wait 7 (fixed)
--wait 3 10 (random)
