---
title: "Side project: news-monitor (Spanish Market Intelligence Refinery)"
project: news-monitor
url: https://github.com/b-qt/news-monitor
---
# news-monitor: Spanish Market Intelligence Refinery (2026)

Tagline: "Stop reading the news, start understanding the sentiment."

## What it does

Ingests news from Google RSS feeds in Spain, sorts it into tech, economy (finance) and real-estate news, enriches each headline with sentiment analysis and shows the result in a dashboard in near real time.

## How it works

- Ingestion with Mage AI every 6 hours, pulling postings through RSS.
- AI enrichment with Robertuito, an NLP model trained on native Spanish text, to score the mood of every headline.
- Storage in a dual PostgreSQL / DuckDB warehouse.
- Ingestion and transformation are kept as separate, decoupled stages.
- Design principle: every data point served to the dashboard must be honest, tested and deduplicated.
