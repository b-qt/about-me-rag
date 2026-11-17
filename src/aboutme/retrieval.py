"""Hybrid retrieval: vector search (meaning) + BM25 (exact names like "GIMAC" or "Arrow"),
fused with reciprocal rank fusion. Job postings are excluded unless the question is
about fit or names a target company, which stops them being mistaken for past work."""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

import chromadb

from .embeddings import Embedder

WORD = re.compile(r"[a-z0-9áéíóúñç+#]+")
STOPWORDS = set(
    "a an and are as at be by did do does for from has have her hers how i in is it its me my of on or "
    "she that the their them they this to was were what when where which who why will with you your "
    "about tell give can could would should".split()
)
FACT_TYPES = ["profile", "experience", "education", "project"]
CORE_TYPES = {"profile", "experience", "education"}
PROJECT_HINTS = ("project", "github", "repo", "side", "portfolio", "built in her free", "hobby", "personal work")
ALL_TYPES = FACT_TYPES + ["target_role"]
TARGET_HINTS = (
    "job", "posting", "vacanc", "apply", "applying", "application", "role fit", "fit for", "a fit",
    "good fit", "suitable", "match", "align", "target", "interview", "position", "hiring", "candidate",
    "requirement", "gap",
)
REQUIREMENT_SECTION = re.compile(r"requirement|qualification", re.I)


def stem(token: str) -> str:
    """Fold simple plurals so "projects" matches "project"."""
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 3 and token.endswith("s") and not token.endswith(("ss", "us", "is")):
        return token[:-1]
    return token


def tokenize(text: str) -> list[str]:
    return [stem(t) for t in WORD.findall(text.lower()) if t not in STOPWORDS]


@dataclass(frozen=True)
class Hit:
    chunk_id: str
    text: str
    meta: dict
    score: float


class BM25:
    def __init__(self, ids: list[str], texts: list[str], k1: float = 1.5, b: float = 0.75) -> None:
        self.ids = ids
        self.docs = [Counter(tokenize(t)) for t in texts]
        self.lengths = [sum(d.values()) for d in self.docs]
        self.avg = sum(self.lengths) / max(len(self.lengths), 1)
        df = Counter(term for d in self.docs for term in d)
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        self.k1, self.b = k1, b

    def scores(self, query: str) -> dict[str, float]:
        terms = tokenize(query)
        out: dict[str, float] = {}
        for cid, doc, length in zip(self.ids, self.docs, self.lengths):
            s = 0.0
            for t in terms:
                if t in doc:
                    tf = doc[t]
                    s += self.idf[t] * tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * length / self.avg))
            if s > 0:
                out[cid] = s
        return out


def wants_targets(question: str, target_orgs: set[str]) -> bool:
    q = question.lower()
    return any(h in q for h in TARGET_HINTS) or any(org in q for org in target_orgs)


class Retriever:
    def __init__(self, collection: chromadb.Collection, embedder: Embedder) -> None:
        self.collection = collection
        self.embedder = embedder
        data = collection.get(include=["documents", "metadatas"])
        self.texts = dict(zip(data["ids"], data["documents"]))
        self.metas = dict(zip(data["ids"], data["metadatas"]))
        self.bm25 = BM25(list(self.texts), list(self.texts.values()))
        self.target_orgs = {
            m["org"].lower() for m in self.metas.values() if m.get("doc_type") == "target_role" and m.get("org")
        }

    def allowed_types(self, question: str, include_targets: bool | None = None) -> list[str]:
        if include_targets is None:
            include_targets = wants_targets(question, self.target_orgs)
        return ALL_TYPES if include_targets else FACT_TYPES

    def _fused(self, question: str, vector: list[float], types: list[str], pool: int) -> dict[str, float]:
        result = self.collection.query(
            query_embeddings=[vector], n_results=min(pool, len(self.texts)),
            where={"doc_type": {"$in": types}},
        )
        lexical = self.bm25.scores(question)
        lexical_ranked = [cid for cid, _ in sorted(lexical.items(), key=lambda x: -x[1])
                          if self.metas[cid]["doc_type"] in types][:pool]
        fused: dict[str, float] = {}
        for ranking in (result["ids"][0], lexical_ranked):
            for rank, cid in enumerate(ranking):
                fused[cid] = fused.get(cid, 0.0) + 1.0 / (60 + rank + 1)
        return fused

    def search(self, question: str, k: int = 6, include_targets: bool | None = None) -> list[Hit]:
        """Top-k chunks. At least half the slots go to profile/experience/education (unless the
        question is about side projects), and job postings take at most half."""
        types = self.allowed_types(question, include_targets)
        pool = max(k * 3, 12)
        vector = self.embedder.embed([question])[0].tolist()
        fused = self._fused(question, vector, types, pool)
        ranked = sorted(fused, key=lambda c: -fused[c])

        core_min = 0 if about_projects(question) else (k + 1) // 2
        core_fused = self._fused(question, vector, sorted(CORE_TYPES), pool) if core_min else {}
        chosen = sorted(core_fused, key=lambda c: -core_fused[c])[:core_min]

        target_max = k // 2 if "target_role" in types else 0
        targets_taken = 0
        for cid in ranked:
            if len(chosen) >= k:
                break
            if cid in chosen:
                continue
            if self.metas[cid]["doc_type"] == "target_role":
                if targets_taken >= target_max:
                    continue
                targets_taken += 1
            chosen.append(cid)
        score = {c: max(fused.get(c, 0.0), core_fused.get(c, 0.0)) for c in chosen}
        chosen.sort(key=lambda c: -score[c])
        return [Hit(c, self.texts[c], self.metas[c], score[c]) for c in chosen]


    def is_fit_question(self, question: str) -> bool:
        return wants_targets(question, self.target_orgs)

    def postings(self) -> dict[str, list[str]]:
        """doc_id -> chunk ids (in document order) for every saved job posting."""
        out: dict[str, list[str]] = {}
        for cid, meta in sorted(self.metas.items(), key=lambda x: (x[1]["doc_id"], x[1].get("position", 0))):
            if meta["doc_type"] == "target_role":
                out.setdefault(meta["doc_id"], []).append(cid)
        return out

    def search_fit(self, question: str, k: int = 6) -> tuple[list[Hit], list[Hit]]:
        """Evidence from her own record, kept separate from the job postings.

        If the question names a posting's organisation, return that posting in full (up to 4 chunks);
        otherwise return one chunk per posting, preferring its requirements section, so every
        posting can be compared."""
        q = question.lower()
        postings = self.postings()
        named = [d for d, ids in postings.items() if self.metas[ids[0]].get("org", "").lower() in q]
        chosen: list[str] = []
        if named:
            for doc_id in named:
                chosen.extend(postings[doc_id][:4])
        else:
            vector = self.embedder.embed([question])[0].tolist()
            fused = self._fused(question, vector, ["target_role"], len(self.texts))
            for ids in postings.values():
                reqs = [c for c in ids if REQUIREMENT_SECTION.search(self.metas[c].get("section", ""))]
                chosen.append(reqs[0] if reqs else max(ids, key=lambda c: fused.get(c, 0.0)))
        jobs = [Hit(c, self.texts[c], self.metas[c], 0.0) for c in chosen]
        return self._fit_evidence(question, jobs, projects=max(2, k // 3)), jobs

    def _fit_evidence(self, question: str, jobs: list[Hit], projects: int) -> list[Hit]:
        """Broad evidence for a comparison: the passage of each role, degree and profile section
        that best matches the postings' requirements, plus the most relevant side projects."""
        query = "\n".join([question] + [h.text for h in jobs])
        vector = self.embedder.embed([query])[0].tolist()
        fused = self._fused(query, vector, FACT_TYPES, len(self.texts))
        best_per_doc: dict[str, str] = {}
        for cid in sorted(fused, key=lambda c: -fused[c]):
            meta = self.metas[cid]
            if meta["doc_type"] == "profile":
                if meta.get("section") in {"Links"}:
                    continue
                key = f"{meta['doc_id']}#{meta.get('section')}"  # every skills/summary/languages section
            else:
                key = meta.get("project") or meta["doc_id"]  # a project's write-up and CSV row count once
            best_per_doc.setdefault(key, cid)
        core = [c for c in best_per_doc.values() if self.metas[c]["doc_type"] in CORE_TYPES]
        side = [c for c in best_per_doc.values() if self.metas[c]["doc_type"] == "project"][:projects]
        core.sort(key=lambda c: (self.metas[c]["doc_type"] != "profile", -fused[c]))
        return [Hit(c, self.texts[c], self.metas[c], fused[c]) for c in core + side]


def about_projects(question: str) -> bool:
    q = question.lower()
    return any(h in q for h in PROJECT_HINTS)
