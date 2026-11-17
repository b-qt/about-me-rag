"""Snapshot public GitHub repositories as raw material for knowledge/projects.csv (offline helper).

Writes private/raw/github_snapshot.md for review; nothing is indexed automatically.
Repos without a language are left blank rather than guessed, badges/HTML/images are
stripped from READMEs, and this project's own repo is skipped so the assistant never
retrieves its own marketing text.

    python scripts/fetch_github.py                 # uses GITHUB_USERNAME (default b-qt)
    GITHUB_TOKEN=... python scripts/fetch_github.py   # optional, avoids rate limits
"""
from __future__ import annotations

import json
import os
import re
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
USERNAME = os.getenv("GITHUB_USERNAME", "b-qt")
SKIP = {"about-me-rag", USERNAME}  # this project, and the profile README repo
API = "https://api.github.com"


def get(url: str, raw: bool = False):
    headers = {"User-Agent": "about-me-rag", "Accept": "application/vnd.github.raw" if raw else "application/vnd.github+json"}
    if os.getenv("GITHUB_TOKEN"):
        headers["Authorization"] = f"Bearer {os.environ['GITHUB_TOKEN']}"
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=30) as response:
        body = response.read().decode("utf-8", errors="ignore")
    return body if raw else json.loads(body)


def clean_readme(text: str, limit: int = 1500) -> str:
    text = re.sub(r"<!--.*?-->", "", text, flags=re.S)
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text)            # images and badges
    text = re.sub(r"\[!\[[^\]]*\]\([^)]*\)\]\([^)]*\)", "", text)
    text = re.sub(r"<[^>]+>", "", text)                         # HTML tags
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)         # links -> text
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()[:limit]


def main() -> None:
    repos = get(f"{API}/users/{USERNAME}/repos?per_page=100&sort=updated")
    out = [f"# GitHub snapshot for {USERNAME}\n"]
    for repo in repos:
        if repo["name"] in SKIP or repo.get("fork"):
            continue
        try:
            readme = clean_readme(get(f"{API}/repos/{USERNAME}/{repo['name']}/readme", raw=True))
        except Exception:
            readme = ""
        out.append(
            f"## {repo['name']}\n- URL: {repo['html_url']}\n- Language: {repo.get('language') or ''}\n"
            f"- Created: {repo['created_at'][:10]} · Last push: {repo['pushed_at'][:10]}\n"
            f"- Description: {repo.get('description') or ''}\n\n{readme}\n"
        )
    target = ROOT / "private" / "raw" / "github_snapshot.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(out), encoding="utf-8")
    print(f"Wrote {target.relative_to(ROOT)}. Review it, then update knowledge/projects.csv.")


if __name__ == "__main__":
    main()
