#!/usr/bin/env python3
# FULLY UPGRADED STORY COLLECTOR (I, II, III, IV, VI, VII, IX, XIII + local NLP)

"""
rscraper.py

Reddit r/shortscarystories (or similar) collector.

- Fetches a listing page HTML (like your uploaded file) or uses the Reddit JSON API.
- Extracts all post entries (i.e., stories below Community highlights / main feed).
- Stores:
    articles(id, reddit_id, title, author, author_flair, link_flair, content,
             url, created_utc, edited_utc, score, upvote_ratio, num_comments,
             over_18, spoiler, locked, archived, distinguished,
             collected_at, collected_date, wordcount)
    comments(id, article_id, comment_id, parent_id, username, is_submitter,
             votes, controversiality, depth, distinguished, created_utc,
             comment, collected_at)
    raw_html(id, source_url, html_gz, collected_at)
    raw_json(id, source_url, json_gz, collected_at)
    runs(id, started_at, finished_at, total_articles, total_comments, errors)
    nlp_features(article_id, wordcount, sentence_count, avg_sentence_length)
    embeddings(article_id, model, vector_json)

  Usage examples:

    # Run once, collect all posts, default wait
    python sssscraper.py https://www.reddit.com/r/shortscarystories/ stories.db

    # Run once, collect 5000 posts, wait 7 seconds between each
    python sssscraper.py https://www.reddit.com/r/shortscarystories/ stories.db 0 5000 --wait 7

    # Run every 15 minutes, collect first 200 posts, random wait 3–10 seconds
    python sssscraper.py https://www.reddit.com/r/shortscarystories/ stories.db 15 200 --wait 3 10

    # Load from local HTML file
    
    # Full extended example with all parameters:
    # Run once, collect 5000 posts, wait randomly 3–15 seconds between each
    python3 scraper.py https://www.reddit.com/r/shortscarystories/ stories.db 0 5000 --wait 3 15
    python sssscraper.py file:///path/to/reddit-r-shortscarystories stories.db


    python story_collector.py https://www.reddit.com/r/shortscarystories/ stories.db
    python story_collector.py file:///path/to/reddit-r-shortscarystories stories.db
    python story_collector.py https://www.reddit.com/r/shortscarystories/ stories.db 15 --wait 3 7
"""

import sys
import os
import sqlite3
import time
import gzip
import json
import random
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import urlparse
from datetime import datetime, timezone

import requests
from bs4 import BeautifulSoup


USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:120.0) Gecko/20100101 Firefox/120.0"
DEFAULT_DB_PATH = "stories.db"
REQUEST_TIMEOUT = 20
COLLECTOR_VERSION = "1.0-upgraded"
ARTICLE_PAUSE_SECONDS = 7
WAIT_MIN: Optional[float] = None
WAIT_MAX: Optional[float] = None

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
    _EMBED_MODEL: Optional[SentenceTransformer] = None
except ImportError:  # pragma: no cover - optional dependency
    SentenceTransformer = None  # type: ignore
    _EMBED_MODEL = None


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def init_db(conn: sqlite3.Connection) -> None:
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute("PRAGMA synchronous = NORMAL;")
    conn.execute("PRAGMA temp_store = MEMORY;")
    conn.execute("PRAGMA cache_size = -200000;")

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS articles (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            reddit_id      TEXT NOT NULL,
            title          TEXT NOT NULL,
            author         TEXT,
            author_flair   TEXT,
            link_flair     TEXT,
            content        TEXT,
            url            TEXT,
            created_utc    REAL,
            edited_utc     REAL,
            score          INTEGER,
            upvote_ratio   REAL,
            num_comments   INTEGER,
            over_18        INTEGER,
            spoiler        INTEGER,
            locked         INTEGER,
            archived       INTEGER,
            distinguished  TEXT,
            collected_at   TEXT NOT NULL,
            collected_date TEXT NOT NULL,
            wordcount      INTEGER,
            UNIQUE (reddit_id)
        );
        """
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS comments (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            article_id      INTEGER NOT NULL,
            comment_id      TEXT,
            parent_id       TEXT,
            username        TEXT,
            is_submitter    INTEGER,
            votes           INTEGER,
            controversiality INTEGER,
            depth           INTEGER,
            distinguished   TEXT,
            created_utc     REAL,
            comment         TEXT,
            collected_at    TEXT,
            UNIQUE(article_id, comment_id),
            FOREIGN KEY(article_id) REFERENCES articles(id) ON DELETE CASCADE
        );
        """
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS raw_html (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            source_url   TEXT,
            html_gz      BLOB,
            collected_at TEXT
        );
        """
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS raw_json (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            source_url   TEXT,
            json_gz      BLOB,
            collected_at TEXT
        );
        """
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS runs (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            started_at      TEXT,
            finished_at     TEXT,
            total_articles  INTEGER,
            total_comments  INTEGER,
            errors          INTEGER,
            version         TEXT
        );
        """
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS nlp_features (
            article_id        INTEGER PRIMARY KEY,
            wordcount         INTEGER,
            sentence_count    INTEGER,
            avg_sentence_len  REAL,
            FOREIGN KEY(article_id) REFERENCES articles(id) ON DELETE CASCADE
        );
        """
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS embeddings (
            article_id  INTEGER PRIMARY KEY,
            model       TEXT,
            vector_json TEXT,
            FOREIGN KEY(article_id) REFERENCES articles(id) ON DELETE CASCADE
        );
        """
    )

    conn.commit()


def compute_wordcount(text: str) -> int:
    return len(text.split()) if text else 0


def basic_nlp_features(text: str) -> Tuple[int, int, float]:
    if not text:
        return 0, 0, 0.0
    words = text.split()
    wordcount = len(words)
    sentences = [s.strip() for s in text.replace("\n", " ").split(".") if s.strip()]
    sentence_count = len(sentences) if sentences else 1
    avg_len = float(wordcount) / float(sentence_count) if sentence_count else float(wordcount)
    return wordcount, sentence_count, avg_len


def upsert_nlp_features(conn: sqlite3.Connection, article_id: int, text: str) -> None:
    wordcount, sentence_count, avg_len = basic_nlp_features(text)
    conn.execute(
        """
        INSERT INTO nlp_features (article_id, wordcount, sentence_count, avg_sentence_len)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(article_id) DO UPDATE SET
            wordcount = excluded.wordcount,
            sentence_count = excluded.sentence_count,
            avg_sentence_len = excluded.avg_sentence_len;
        """,
        (article_id, wordcount, sentence_count, avg_len),
    )
    conn.execute("UPDATE articles SET wordcount = ? WHERE id = ?", (wordcount, article_id))
    conn.commit()


def maybe_get_embed_model() -> Optional["SentenceTransformer"]:
    global _EMBED_MODEL
    if SentenceTransformer is None:
        return None
    if _EMBED_MODEL is None:
        _EMBED_MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _EMBED_MODEL


def upsert_embedding(conn: sqlite3.Connection, article_id: int, text: str) -> None:
    model = maybe_get_embed_model()
    if model is None or not text:
        return
    vec = model.encode(text)
    vector_json = json.dumps(vec.tolist())
    conn.execute(
        """
        INSERT INTO embeddings (article_id, model, vector_json)
        VALUES (?, ?, ?)
        ON CONFLICT(article_id) DO UPDATE SET
            model = excluded.model,
            vector_json = excluded.vector_json;
        """,
        (article_id, "sentence-transformers/all-MiniLM-L6-v2", vector_json),
    )
    conn.commit()


def gzip_bytes(data: bytes) -> bytes:
    return gzip.compress(data)


def store_raw_html(conn: sqlite3.Connection, source_url: str, html: str) -> None:
    conn.execute(
        """
        INSERT INTO raw_html (source_url, html_gz, collected_at)
        VALUES (?, ?, ?);
        """,
        (source_url, gzip_bytes(html.encode("utf-8")), now_utc_iso()),
    )
    conn.commit()


def store_raw_json(conn: Optional[sqlite3.Connection], source_url: str, obj: Any) -> None:
    if conn is None:
        return
    payload = json.dumps(obj).encode("utf-8")
    conn.execute(
        """
        INSERT INTO raw_json (source_url, json_gz, collected_at)
        VALUES (?, ?, ?);
        """,
        (source_url, gzip_bytes(payload), now_utc_iso()),
    )
    conn.commit()


def request_with_retries(url: str, headers: Dict[str, str], expect_json: bool = False,
                         max_retries: int = 3, backoff: float = 2.0) -> Optional[requests.Response]:
    attempt = 0
    while attempt < max_retries:
        try:
            resp = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
            if resp.status_code == 429:
                print(f"[WARN] 429 for {url}, sleeping {backoff}s...")
                time.sleep(backoff)
                attempt += 1
                backoff *= 2
                continue
            resp.raise_for_status()
            return resp
        except Exception as e:
            print(f"[WARN] Request failed for {url}: {e}")
            attempt += 1
            time.sleep(backoff)
    print(f"[ERROR] Giving up on {url} after {max_retries} attempts")
    return None


def upsert_article(
    conn: sqlite3.Connection,
    reddit_id: str,
    title: str,
    content: str,
    url: str,
    meta: Optional[Dict[str, Any]] = None,
) -> int:
    cur = conn.cursor()
    collected_at = now_utc_iso()
    collected_date = collected_at[:10]

    cur.execute("SELECT id FROM articles WHERE reddit_id = ?", (reddit_id,))
    row = cur.fetchone()
    if row is not None:
        article_id = int(row[0])
    else:
        cur.execute(
            """
            INSERT INTO articles (reddit_id, title, content, url, collected_at, collected_date)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(reddit_id) DO UPDATE SET
                title = excluded.title,
                content = excluded.content,
                url = excluded.url,
                collected_at = excluded.collected_at,
                collected_date = excluded.collected_date;
            """,
            (reddit_id, title, content, url, collected_at, collected_date),
        )
        conn.commit()
        cur.execute("SELECT id FROM articles WHERE reddit_id = ?", (reddit_id,))
        row = cur.fetchone()
        if row is None:
            raise RuntimeError("Failed to retrieve article id after upsert")
        article_id = int(row[0])

    if meta:
        cur.execute(
            """
            UPDATE articles SET
                author = COALESCE(?, author),
                author_flair = COALESCE(?, author_flair),
                link_flair = COALESCE(?, link_flair),
                created_utc = COALESCE(?, created_utc),
                edited_utc = COALESCE(?, edited_utc),
                score = COALESCE(?, score),
                upvote_ratio = COALESCE(?, upvote_ratio),
                num_comments = COALESCE(?, num_comments),
                over_18 = COALESCE(?, over_18),
                spoiler = COALESCE(?, spoiler),
                locked = COALESCE(?, locked),
                archived = COALESCE(?, archived),
                distinguished = COALESCE(?, distinguished)
            WHERE id = ?;
            """,
            (
                meta.get("author"),
                meta.get("author_flair"),
                meta.get("link_flair"),
                meta.get("created_utc"),
                meta.get("edited_utc"),
                meta.get("score"),
                meta.get("upvote_ratio"),
                meta.get("num_comments"),
                int(meta["over_18"]) if "over_18" in meta and meta["over_18"] is not None else None,
                int(meta["spoiler"]) if "spoiler" in meta and meta["spoiler"] is not None else None,
                int(meta["locked"]) if "locked" in meta and meta["locked"] is not None else None,
                int(meta["archived"]) if "archived" in meta and meta["archived"] is not None else None,
                meta.get("distinguished"),
                article_id,
            ),
        )
        conn.commit()

    upsert_nlp_features(conn, article_id, content)
    upsert_embedding(conn, article_id, content)
    return article_id


def insert_comment(
    conn: sqlite3.Connection,
    article_id: int,
    comment_data: Dict[str, Any],
) -> None:
    collected_at = now_utc_iso()
    conn.execute(
        """
        INSERT INTO comments (
            article_id, comment_id, parent_id, username, is_submitter, votes,
            controversiality, depth, distinguished, created_utc, comment, collected_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(article_id, comment_id) DO UPDATE SET
            username = excluded.username,
            votes = excluded.votes,
            controversiality = excluded.controversiality,
            distinguished = excluded.distinguished,
            created_utc = excluded.created_utc,
            comment = excluded.comment,
            collected_at = excluded.collected_at;
        """,
        (
            article_id,
            comment_data.get("comment_id"),
            comment_data.get("parent_id"),
            comment_data.get("username"),
            int(comment_data["is_submitter"]) if comment_data.get("is_submitter") is not None else None,
            comment_data.get("votes"),
            comment_data.get("controversiality"),
            comment_data.get("depth"),
            comment_data.get("distinguished"),
            comment_data.get("created_utc"),
            comment_data.get("comment"),
            collected_at,
        ),
    )
    conn.commit()


def fetch_html_from_url(url: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme == "file":
        path = parsed.path
        if not os.path.exists(path):
            raise FileNotFoundError(f"Local file not found: {path}")
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()

    headers = {"User-Agent": USER_AGENT}
    resp = request_with_retries(url, headers, expect_json=False)
    if resp is None:
        raise RuntimeError(f"Failed to fetch HTML from {url}")
    return resp.text


def extract_posts_from_listing(html: str) -> Iterable[Dict[str, Any]]:
    soup = BeautifulSoup(html, "html.parser")
    posts = soup.find_all("shreddit-post")

    for post in posts:
        reddit_id = post.get("id")
        permalink = post.get("permalink")

        title_tag = post.find("faceplate-screen-reader-content")
        title = title_tag.get_text(strip=True) if title_tag else None

        body_div = post.find(attrs={"property": "schema:articleBody"})
        content = body_div.get_text("\n", strip=True) if body_div else ""

        if not reddit_id or not title:
            continue

        yield {
            "reddit_id": reddit_id,
            "title": title,
            "content": content,
            "permalink": permalink,
            "meta": {},
        }


def fetch_posts_from_subreddit_api(listing_url: str) -> Iterable[Dict[str, Any]]:
    parsed = urlparse(listing_url)
    path = parsed.path or "/"
    if not path.endswith("/"):
        path += "/"
    api_url = f"https://www.reddit.com{path}.json?raw_json=1&limit=50"
    headers = {"User-Agent": USER_AGENT}

    resp = request_with_retries(api_url, headers, expect_json=True)
    if resp is None:
        return []

    try:
        data = resp.json()
    except ValueError:
        print(f"[WARN] JSON decode failed for {api_url}")
        return []

    if not isinstance(data, dict):
        return []

    children = data.get("data", {}).get("children", [])
    results: List[Dict[str, Any]] = []

    for child in children:
        if not isinstance(child, dict):
            continue
        if child.get("kind") != "t3":
            continue
        d = child.get("data", {})

        post_id = d.get("id")
        title = d.get("title") or ""
        content = d.get("selftext") or ""
        permalink = d.get("permalink") or ""
        author = d.get("author")
        author_flair = d.get("author_flair_text")
        link_flair = d.get("link_flair_text")

        if not post_id or not title:
            continue

        reddit_id = post_id
        if not reddit_id.startswith("t3_"):
            reddit_id = f"t3_{reddit_id}"

        meta = {
            "author": author,
            "author_flair": author_flair,
            "link_flair": link_flair,
            "created_utc": d.get("created_utc"),
            "edited_utc": d.get("edited") if isinstance(d.get("edited"), (int, float)) else None,
            "score": d.get("score"),
            "upvote_ratio": d.get("upvote_ratio"),
            "num_comments": d.get("num_comments"),
            "over_18": d.get("over_18"),
            "spoiler": d.get("spoiler"),
            "locked": d.get("locked"),
            "archived": d.get("archived"),
            "distinguished": d.get("distinguished"),
        }

        results.append(
            {
                "reddit_id": reddit_id,
                "title": title,
                "content": content,
                "permalink": permalink,
                "meta": meta,
            }
        )

    return results


def parse_comment_tree(data: List[Any]) -> List[Dict[str, Any]]:
    if not isinstance(data, list) or not data:
        return []

    comment_listing = data[1].get("data", {}).get("children", []) if len(data) > 1 else []
    collected: List[Dict[str, Any]] = []

    stack: List[Any] = list(comment_listing)
    while stack:
        node = stack.pop()
        if not isinstance(node, dict):
            continue
        if node.get("kind") != "t1":
            continue
        d = node.get("data", {})
        author = d.get("author") or "[deleted]"
        body = d.get("body") or ""
        score = d.get("score") if d.get("score") is not None else d.get("ups", 0)

        collected.append(
            {
                "comment_id": d.get("id"),
                "parent_id": d.get("parent_id"),
                "username": author,
                "is_submitter": d.get("is_submitter"),
                "votes": int(score or 0),
                "controversiality": d.get("controversiality"),
                "depth": d.get("depth"),
                "distinguished": d.get("distinguished"),
                "created_utc": d.get("created_utc"),
                "comment": body,
            }
        )

        replies = d.get("replies")
        if isinstance(replies, dict):
            children = replies.get("data", {}).get("children", [])
            for child in children:
                stack.append(child)

    return collected


def fetch_comments_from_api(permalink: str) -> List[Dict[str, Any]]:
    if not permalink:
        return []

    url = f"https://www.reddit.com{permalink}.json?raw_json=1"
    headers = {"User-Agent": USER_AGENT}
    resp = request_with_retries(url, headers, expect_json=True)
    if resp is None:
        return []

    try:
        data = resp.json()
    except ValueError:
        print(f"[WARN] JSON decode failed for {url}")
        return []

    return parse_comment_tree(data)


def process_once(conn: sqlite3.Connection, listing_url: str, max_posts: Optional[int] = None) -> Tuple[int, int, int]:
    errors = 0
    total_articles = 0
    total_comments = 0

    parsed = urlparse(listing_url)
    is_reddit = parsed.scheme in ("http", "https") and parsed.netloc.endswith("reddit.com")

    posts: List[Dict[str, Any]]

    if is_reddit:
        print(f"[+] Fetching posts from Reddit API for: {listing_url}")
        posts = list(fetch_posts_from_subreddit_api(listing_url))
    else:
        print(f"[+] Fetching listing page: {listing_url}")
        html = fetch_html_from_url(listing_url)
        store_raw_html(conn, listing_url, html)
        print("[+] Parsing posts from HTML listing...")
        posts = list(extract_posts_from_listing(html))

    print(f"[+] Found {len(posts)} posts to process.")
    if max_posts is not None and max_posts > 0:
        posts = posts[:max_posts]
        print(f"[+] Limiting to first {len(posts)} posts due to max_posts={max_posts}.")

    for i, post in enumerate(posts, 1):
        try:
            reddit_id = post["reddit_id"]
            title = post["title"]
            content = post.get("content", "")
            permalink = post.get("permalink") or ""
            meta = post.get("meta") or {}
            url = f"https://www.reddit.com{permalink}" if permalink.startswith("/") else permalink

            print(f"  [{i}/{len(posts)}] {reddit_id} :: {title[:60]!r}")

            article_id = upsert_article(conn, reddit_id, title, content, url, meta=meta)
            total_articles += 1

            if permalink:
                comments = fetch_comments_from_api(permalink)
                print(f"     -> {len(comments)} comments")
                for c in comments:
                    insert_comment(conn, article_id, c)
                    total_comments += 1

            if WAIT_MIN is not None:
                if WAIT_MAX is not None:
                    delay = random.uniform(WAIT_MIN, WAIT_MAX)
                else:
                    delay = WAIT_MIN
            else:
                delay = float(ARTICLE_PAUSE_SECONDS)

            if delay < 0:
                delay = 0

            if delay > 0:
                print(f"     .. sleeping {delay:.2f}s before next post")
                time.sleep(delay)

        except Exception as e:
            errors += 1
            print(f"[ERROR] Failed processing post index {i}: {e}")

    return total_articles, total_comments, errors


def main(listing_url: str, db_path: str, interval: Optional[int] = None, max_posts: Optional[int] = None) -> None:
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        init_db(conn)

        while True:
            started_at = now_utc_iso()
            print(f"[+] Run started at {started_at}")
            total_articles, total_comments, errors = process_once(conn, listing_url, max_posts)
            finished_at = now_utc_iso()

            conn.execute(
                """
                INSERT INTO runs (started_at, finished_at, total_articles, total_comments, errors, version)
                VALUES (?, ?, ?, ?, ?, ?);
                """,
                (started_at, finished_at, total_articles, total_comments, errors, COLLECTOR_VERSION),
            )
            conn.commit()

            print(
                f"[+] Run finished at {finished_at} :: articles={total_articles}, "
                f"comments={total_comments}, errors={errors}"
            )

            if not interval or interval <= 0:
                break

            print(f"[+] Sleeping {interval} minutes before next run...")
            time.sleep(interval * 60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("listing_url")
    parser.add_argument("db_path", nargs="?", default=DEFAULT_DB_PATH)
    parser.add_argument("interval", nargs="?", type=int, default=None)
    parser.add_argument("max_posts", nargs="?", type=int, default=None)
    parser.add_argument("--wait", nargs="*", type=float, default=None)
    args = parser.parse_args()

    if args.wait:
        if len(args.wait) == 1:
            WAIT_MIN = args.wait[0]
        elif len(args.wait) >= 2:
            WAIT_MIN, WAIT_MAX = args.wait[0], args.wait[1]

    listing_url = args.listing_url
    db_path = args.db_path
    interval = args.interval
    max_posts = args.max_posts

    main(listing_url, db_path, interval, max_posts)

