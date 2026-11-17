---
title: "Side project: job-hunter-pipeline ((Northern) Spain Job Monitor)"
project: job-hunter-pipeline
url: https://github.com/b-qt/job-hunter-pipeline
---
# job-hunter-pipeline: (Northern) Spain Job Monitor (2026)

Tagline: "Turning the job hunt into a quiet pipeline."

## What it does

Monitors the Spanish news cycle and the tech job market, focused on Bilbao and Gijón for her planned move to Bilbao, and turns it into a dashboard of insights.

## How it works

- Ingestion: Mage.ai collects postings and headlines from LinkedIn, InfoJobs and Google News every 6 hours.
- Transformation: dbt runs 13 data-quality tests, removes duplicates and fixes messy dates before data is published.
- Visualisation: a Streamlit dashboard.
