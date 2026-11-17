# knowledge/: everything the assistant is allowed to know

Only files in this folder are indexed. Anything in it can be repeated to any visitor of the public app, so keep it free of salaries, phone numbers, personal emails, signatures and ID numbers (`tests/test_privacy.py` checks for these).

| Path | What goes there | How it is used |
|---|---|---|
| `timeline.csv` | One row per role or degree, with Where/How/Who/Why columns | Source of truth for the Growth and Mind Map pages; sent to the model with every question |
| `projects.csv` | One row per side project | Growth page and project questions |
| `profile/` | Summary, skills, languages, links | Chat |
| `experience/`, `education/` | One markdown file per row of `timeline.csv` (`timeline_id` in the front matter links them) | Chat; org, title and dates come from the CSV |
| `projects/` | Longer write-ups of notable projects | Chat |
| `targets/` | Job postings she is considering, cleaned up | Used only for "fit" questions; always labelled NOT past work |

Supported formats: `.md`, `.txt`, `.pdf`. The index rebuilds automatically when any file here changes.

## timeline.csv columns

- `type`: `work` or `education`
- `start`, `end`: `YYYY` or `YYYY-MM`; leave `end` empty for a current role
- `level`: seniority for the growth chart (1 intern, 2 specialist/consultant, 3 engineer, 4 senior)
- `stack`: tools separated by `;`
- `where`, `how`, `who`, `why`: the Mind Map; leave a cell empty rather than guess
- `key_metric`: the measurable result, if any
- `status`: `verified`, or `needs_confirmation: <what to check>`
