"""
semantic_index.py
-----------------
Semantic + structured retrieval engine for the fragility and consequence models
shown in the Explorer's browse tree.

This replaces the description half of the old rapidfuzz search. It indexes the
*tree-visible* data files (the same files ``render_seismic_tree``,
``render_wind_tree`` and ``render_consequence_tree`` render) into an in-process
Qdrant collection backed by fastembed, and exposes a small hybrid API:

    * ``search(query, mode=...)``  — hybrid dense + BM25 (RRF-fused) for
                                     descriptions, exact/substring matching for
                                     component IDs.
    * ``filter_only(filters)``     — component IDs matching the facets, with no
                                     text query.
    * ``allowed_files(filters)``   — data files whose records match the facets;
                                     the browse tree is narrowed to these.

The module also defines the tree corpus that the tree renderers and the index
share: the list of data files, the dataset ID and asset type of each file, and
the list of excluded datasets.

Design notes
~~~~~~~~~~~~
* **Framework-free.** Nothing here imports Streamlit, so the engine can be
  driven from the command line to sanity-check retrieval quality before any UI
  is wired up::

      python -m dlml.web.st_search.semantic_index "steel column connection"
      python -m dlml.web.st_search.semantic_index --mode id "B.10.31"
      python -m dlml.web.st_search.semantic_index --hazard seismic "exterior wall debris"

* **Corpus parity with the tree.** ``build_tree_corpus`` reads exactly the files
  the tree renderers load (``tree_corpus_files``): the fragility and consequence
  files under ``seismic/``, ``hurricane/building/component/`` and
  ``hurricane/building/portfolio/``. The Hazus hurricane *portfolio* files hold
  about 25.8k templated, near-duplicate models each. They are shown in the tree
  and indexed with ``embed=False``, so they are found by ID and facet search but
  their descriptions are not embedded.

* **Rich embedding context.** ``ComponentGroups`` in the source JSON carries
  human-readable labels ("B.10.31 - Steel Columns", "RWC - Roof-Wall
  Connection") that the current tree throws away. We parse them back out and
  prepend the label chain to each component's description before embedding, so a
  query like "roof to wall connection" matches even when the raw description
  never spells it out.

* **In-process Qdrant.** ``location=":memory:"`` builds a fresh collection from
  the embedded records at every process start. Pass ``path=...`` instead to
  persist on disk. No server required.
"""

from __future__ import annotations

import argparse
import dataclasses
import functools
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from dlml._catalog import data_root

# qdrant-client + fastembed are imported lazily inside SemanticIndex so that the
# corpus-building half of this module (and `import` in general) does not require
# the heavy ONNX runtime to be installed.

# ─── Constants ───────────────────────────────────────────────────────────────

#: Default fastembed model. Small (384-dim), CPU-friendly, no torch dependency —
#: chosen to stay within Streamlit Community Cloud's memory/cold-start budget.
DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"

#: Sparse model for the lexical half of hybrid search. BM25 is pure IDF-weighted
#: term matching (no neural net — computed via py-rust-stemmers), so it adds
#: almost nothing to cold start or memory. It rescues rare salient tokens the
#: small dense model dilutes: dense ranks "elevator failure" by the common word
#: "failure", BM25 weights the rare "elevator" far higher. Set to None for
#: dense-only. Results from the two are fused with Reciprocal Rank Fusion.
DEFAULT_SPARSE_MODEL = "Qdrant/bm25"

DEFAULT_COLLECTION = "components"

#: Top-level keys in a fragility.json that are not components.
_NON_COMPONENT_KEYS = {"References"}

# Scores assigned to non-vector (ID) matches so they can be ranked and
# merged on the same 0–1 scale as cosine similarity.
_SCORE_EXACT = 1.0
_SCORE_PREFIX = 0.9
_SCORE_SUBSTRING = 0.7


# ─── Records & filters ───────────────────────────────────────────────────────


@dataclass
class ComponentRecord:
    """One indexable component, flattened from a fragility.json entry."""

    component_id: str
    description: str
    short_name: str          # source library, e.g. "FEMA P-58 2nd Edition"
    file_path: str
    hazard: str              # "seismic" | "hurricane" | "flood" | ""
    category: str            # "FEMA" | "HAZUS" | "SIMCENTER" | "RESEARCH"
    group: str               # 1-segment prefix, e.g. "B"        (tree level 3)
    group_label: str         # human label,      e.g. "B - Shell"
    subgroup: str            # 2-segment prefix, e.g. "B.10"     (tree level 4)
    subgroup_label: str      # human label,      e.g. "B.10 - Super Structure"
    dataset: str = "fragility"  # "fragility" | "consequence" — which file it came from
    type: str = "Damage"     # "Damage" | "Consequence"
    #: Second segment of the dataset ID, e.g. "building" or "power_network".
    asset_type: str = ""
    #: Whether this record's description is embedded for semantic search. False
    #: for the huge hurricane building portfolio (about 25.8k templated
    #: near-duplicate models per file) — those stay findable via ID/substring +
    #: facet filters and the browse tree without paying the (~minutes) embedding
    #: cost at startup.
    embed: bool = True
    #: Extra label names (deepest group chain) used only to enrich the embedding.
    _name_chain: List[str] = field(default_factory=list, repr=False)

    @property
    def embed_text(self) -> str:
        """
        Text fed to the embedding model.

        Prepends the source and the human-readable group/sub-group name chain to
        the raw description so the vector carries context the bare attribute
        string lacks. Tweak this method to experiment with retrieval quality.
        """
        names = " / ".join(self._name_chain) if self._name_chain else ""
        parts = [self.short_name, names, self.description]
        return " | ".join(p for p in parts if p)

    @property
    def payload(self) -> dict:
        """Qdrant payload — every field a facet filter or the UI might need."""
        return {
            "component_id": self.component_id,
            "description": self.description,
            "short_name": self.short_name,
            "file_path": self.file_path,
            "hazard": self.hazard,
            "asset_type": self.asset_type,
            "category": self.category,
            "group": self.group,
            "group_label": self.group_label,
            "subgroup": self.subgroup,
            "subgroup_label": self.subgroup_label,
            "dataset": self.dataset,
            "type": self.type,
        }


@dataclass
class SearchFilters:
    """Structured facets. Any ``None`` field is ignored (matches everything)."""

    hazard: Optional[str] = None
    category: Optional[str] = None
    source: Optional[str] = None       # matches short_name
    group: Optional[str] = None        # 1-segment prefix
    subgroup: Optional[str] = None     # 2-segment prefix
    dataset: Optional[str] = None      # "fragility" | "consequence"
    type: Optional[str] = None
    asset_type: Optional[str] = None   # e.g. "building", "power_network"

    def is_empty(self) -> bool:
        return all(getattr(self, f.name) is None for f in dataclasses.fields(self))

    def conditions(self) -> dict[str, str]:
        """Map the payload key of each set facet to its value.

        Every field is stored under its own name in the payload, except
        ``source``, which is stored as ``short_name``.
        """
        conditions: dict[str, str] = {}
        for f in dataclasses.fields(self):
            value = getattr(self, f.name)
            if value is not None:
                conditions["short_name" if f.name == "source" else f.name] = value
        return conditions

    def matches(self, payload: dict) -> bool:
        """Pure-Python evaluation of the facets against a payload."""
        return all(payload.get(key) == want for key, want in self.conditions().items())


@dataclass
class SearchHit:
    """A single ranked result."""

    component_id: str
    score: float
    payload: dict

    @property
    def description(self) -> str:
        return self.payload.get("description", "")

    @property
    def file_path(self) -> str:
        return self.payload.get("file_path", "")


# ─── Corpus building ─────────────────────────────────────────────────────────


def _human_name(label: str) -> str:
    """'B.10.31 - Steel Columns' -> 'Steel Columns'. Falls back to the label."""
    parts = label.split(" - ", 1)
    return parts[1].strip() if len(parts) == 2 else label.strip()


def _prefix_of(label: str) -> str:
    """'B.10.31 - Steel Columns' -> 'B.10.31'."""
    return label.split(" - ", 1)[0].strip()


def _flatten_group_labels(node, acc: Dict[str, str]) -> Dict[str, str]:
    """
    Walk a ComponentGroups structure and build ``prefix -> full label``.

    ComponentGroups nests arbitrarily: ``{label: {label: [label, ...]}}``. Every
    string label anywhere in the tree is parsed into its dotted prefix and
    recorded, so a component ID can later be resolved to the most specific label
    available.
    """
    if isinstance(node, dict):
        for label, child in node.items():
            if isinstance(label, str):
                acc[_prefix_of(label)] = label
            _flatten_group_labels(child, acc)
    elif isinstance(node, list):
        for label in node:
            if isinstance(label, str):
                acc[_prefix_of(label)] = label
    return acc


def dataset_id_from_path(file_path: str, base: str | Path | None = None) -> str:
    """
    Return the dataset ID of a data file.

    The ID is the file's parent folder relative to *base* (default
    :func:`data_root`), as a POSIX string, e.g.
    ``seismic/building/portfolio/Hazus v5.1``. Its second segment is the
    asset type.
    """
    base_dir = Path(base) if base is not None else data_root()
    return Path(file_path).parent.relative_to(base_dir).as_posix()


def asset_type_from_path(file_path: str, base: str | Path | None = None) -> str:
    """
    Return the asset type of a data file, the second segment of its dataset ID.

    Returns ``""`` for a dataset ID with one segment. Raises ``ValueError`` for
    a path outside *base*, as :func:`dataset_id_from_path` does.
    """
    segments = dataset_id_from_path(file_path, base).split("/")
    return segments[1] if len(segments) > 1 else ""


def _category_from_path(file_path: str) -> str:
    """Source category for the search facet (the CLI's ``--category``).

    Every dataset not recognized as FEMA, Hazus, or SimCenter gets the
    RESEARCH category.
    """
    if "FEMA" in file_path:
        return "FEMA"
    if "Hazus" in file_path:
        return "HAZUS"
    if "SimCenter" in file_path:
        return "SIMCENTER"
    return "RESEARCH"


def _record_from_component(
    comp_id: str,
    comp_data: dict,
    *,
    short_name: str,
    file_path: str,
    hazard: str,
    asset_type: str,
    category: str,
    label_map: Dict[str, str],
    source_type: str,
    dataset: str,
    embed: bool,
) -> Optional[ComponentRecord]:
    """Build one ComponentRecord, or None if it has no usable description."""
    description = (comp_data.get("Description") or "").strip()
    if not description:
        return None

    segments = comp_id.split(".")
    group = segments[0]
    subgroup = ".".join(segments[:2]) if len(segments) >= 2 else group

    # Resolve labels for the group and sub-group prefixes.
    group_label = label_map.get(group, group)
    subgroup_label = label_map.get(subgroup, subgroup)

    # Collect the human-name chain across every resolvable prefix depth
    # (group, sub-group, leaf-group …) for the embedding context. De-duplicate
    # while preserving order.
    name_chain: List[str] = []
    for depth in range(1, len(segments)):
        prefix = ".".join(segments[:depth])
        label = label_map.get(prefix)
        if label:
            name = _human_name(label)
            if name and name not in name_chain:
                name_chain.append(name)

    # Type: prefer explicit JSON, else a keyword heuristic.
    comp_type = source_type
    text = (short_name + " " + description).lower()
    if "consequence" in text:
        comp_type = "Consequence"

    return ComponentRecord(
        component_id=comp_id,
        description=description,
        short_name=short_name,
        file_path=file_path,
        hazard=hazard,
        asset_type=asset_type,
        category=category,
        group=group,
        group_label=group_label,
        subgroup=subgroup,
        subgroup_label=subgroup_label,
        dataset=dataset,
        embed=embed,
        type=comp_type,
        _name_chain=name_chain,
    )


def _records_from_file(
    file_path: str,
    dataset: str = "fragility",
    base: str | Path | None = None,
) -> List[ComponentRecord]:
    try:
        with open(file_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception as exc:  # pragma: no cover - defensive
        print(f"Error reading {file_path}: {exc}")
        return []

    meta: dict = data.get("_GeneralInformation", {}) or {}
    short_name: str = meta.get("ShortName", Path(file_path).parent.name)
    label_map = _flatten_group_labels(meta.get("ComponentGroups", {}), {})

    dataset_id = dataset_id_from_path(file_path, base)
    segments = dataset_id.split("/")
    hazard = segments[0]
    asset_type = segments[1] if len(segments) > 1 else ""
    # The hurricane building portfolio holds about 25.8k templated
    # near-duplicate models per file. Index it for ID/facet search and the
    # tree, but do not embed it, which keeps cold start fast.
    embed = not dataset_id.startswith("hurricane/building/portfolio/")
    category = _category_from_path(file_path)
    source_type = "Consequence" if "consequence" in str(meta.get("Type", "")).lower() else "Damage"

    records: List[ComponentRecord] = []
    for comp_id, comp_data in data.items():
        if comp_id.startswith("_") or comp_id in _NON_COMPONENT_KEYS:
            continue
        if not isinstance(comp_data, dict):
            continue
        rec = _record_from_component(
            comp_id,
            comp_data,
            short_name=short_name,
            file_path=file_path,
            hazard=hazard,
            asset_type=asset_type,
            category=category,
            label_map=label_map,
            source_type=source_type,
            dataset=dataset,
            embed=embed,
        )
        if rec is not None:
            records.append(rec)
    return records


#: Candidate filenames per dataset. Consequence has two forms: Hazus hurricane
#: "coupled" uses ``consequence_repair.*`` (damage-state based); "original" uses
#: ``loss_repair.*`` (a continuous loss function). A directory has at most one.
_DATASET_FILENAMES = {
    "fragility": ("fragility.json",),
    "consequence": ("consequence_repair.json", "loss_repair.json"),
}

EXCLUDED_DATASETS: frozenset[str] = frozenset()
"""Dataset IDs (paths relative to the data root, in the form
``seismic/building/portfolio/Hazus v5.1``) that the Explorer leaves out of
the tree and the search index. Meant for datasets whose models duplicate
those of another dataset. The set is empty."""


def tree_corpus_files(
    base_path: str | Path | None = None, dataset: str = "fragility"
) -> List[str]:
    """
    Return the tree-visible data files for *dataset*, in a stable order.

    Scope (mirrors tree_visuals):
      * seismic   — every matching file under ``seismic/``
      * hurricane — the ``building/component/`` library and the
                    ``building/portfolio/`` Hazus models.

    The consequence dataset matches both ``consequence_repair.json`` and
    ``loss_repair.json``; directories without a match are not listed. Datasets
    in ``EXCLUDED_DATASETS`` are skipped.

    ``base_path`` defaults to the packaged dlml data root (:func:`data_root`),
    so discovery is independent of the current working directory. Pass an
    explicit path only to point the corpus at a different data tree (e.g. the
    CLI ``--base`` override or a test fixture).
    """
    filenames = _DATASET_FILENAMES[dataset]
    base = Path(base_path) if base_path is not None else data_root()
    roots = [
        base / "seismic",
        base / "hurricane" / "building" / "component",
        base / "hurricane" / "building" / "portfolio",
    ]
    files: List[str] = []
    for root in roots:
        if not root.exists():
            continue
        for filename in filenames:
            files.extend(
                p
                for p in sorted(str(q) for q in root.rglob(filename))
                if dataset_id_from_path(p, base) not in EXCLUDED_DATASETS
            )
    return files


def build_tree_corpus(base_path: str | Path | None = None) -> List[ComponentRecord]:
    """
    Parse every tree-visible record into a flat list of ComponentRecords.

    Includes both the fragility (damage) models and the consequence (repair)
    models, each tagged via ``ComponentRecord.dataset`` so search can filter by
    one or span both. A component that exists in both files (e.g. FEMA
    ``B.10.31.001``) yields two distinct records.
    """
    records: List[ComponentRecord] = []
    for dataset in ("fragility", "consequence"):
        for fp in tree_corpus_files(base_path, dataset):
            records.extend(_records_from_file(fp, dataset, base=base_path))
    return records


def prefetch_embedding_models() -> None:
    """
    Download the search models into the local cache if not already present.

    Called by ``dlml explorer`` at startup so the first search does not stall on
    a silent download from HuggingFace. Safe to call repeatedly: fastembed
    caches the models on disk, so later calls just load from that cache. Requires
    the search extra (fastembed); instantiating each model triggers its download.
    """
    from fastembed import SparseTextEmbedding, TextEmbedding

    TextEmbedding(DEFAULT_MODEL)
    if DEFAULT_SPARSE_MODEL:
        SparseTextEmbedding(DEFAULT_SPARSE_MODEL)


# ─── The index ───────────────────────────────────────────────────────────────


class SemanticIndex:
    """
    Hybrid retrieval over a list of ComponentRecords.

    Vectors (via Qdrant + fastembed) power description search; exact/substring
    matching powers ID search. Facets are evaluated as Qdrant payload
    filters for vector queries and in Python everywhere else.
    """

    def __init__(
        self,
        records: List[ComponentRecord],
        *,
        model_name: str = DEFAULT_MODEL,
        sparse_model_name: Optional[str] = DEFAULT_SPARSE_MODEL,
        collection_name: str = DEFAULT_COLLECTION,
        location: str = ":memory:",
        path: Optional[str] = None,
    ) -> None:
        self.records = records
        self.collection_name = collection_name
        self.model_name = model_name
        self.sparse_model_name = sparse_model_name

        # Lazy import keeps the corpus helpers usable without qdrant installed.
        try:
            from qdrant_client import QdrantClient, models
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "qdrant-client with fastembed is required for SemanticIndex. "
                "Install it with:  pip install 'qdrant-client[fastembed]'"
            ) from exc
        self._models = models

        self.client = (
            QdrantClient(path=path) if path else QdrantClient(location=location)
        )
        self.client.set_model(model_name)
        # Registering a sparse model makes add()/query() embed and search both
        # dense and sparse vectors; query() then fuses them with RRF server-side.
        if sparse_model_name:
            self.client.set_sparse_model(sparse_model_name)

        self._index_records()

    @property
    def hybrid(self) -> bool:
        """True when sparse (BM25) fusion is active alongside dense vectors."""
        return self.sparse_model_name is not None

    # -- build -----------------------------------------------------------------

    def _index_records(self) -> None:
        """
        Embed and upsert the embeddable records. Integer ids; component_id in payload.

        Only records flagged ``embed=True`` are sent to the model — the huge
        hurricane portfolio is skipped (it stays in ``self.records`` for ID and
        facet search but never reaches the embedder). ``add`` embeds with both
        the dense and (if registered) sparse models.
        """
        to_embed = [r for r in self.records if r.embed]
        if not to_embed:
            return
        self.client.add(
            collection_name=self.collection_name,
            documents=[r.embed_text for r in to_embed],
            metadata=[r.payload for r in to_embed],
            ids=list(range(len(to_embed))),
        )

    # -- filters ---------------------------------------------------------------

    def _qdrant_filter(self, filters: Optional[SearchFilters]):
        if filters is None or filters.is_empty():
            return None
        m = self._models
        conditions = [
            m.FieldCondition(key=key, match=m.MatchValue(value=value))
            for key, value in filters.conditions().items()
        ]
        return m.Filter(must=conditions) if conditions else None

    # -- queries ---------------------------------------------------------------

    def search(
        self,
        query: str,
        *,
        mode: str = "description",
        filters: Optional[SearchFilters] = None,
        limit: int = 50,
        score_cutoff: float = 0.0,
    ) -> List[SearchHit]:
        """
        Run a hybrid search.

        Parameters
        ----------
        query : str
            The user's query text.
        mode : {"description", "id"}
            * ``description`` — semantic vector search over the description.
            * ``id``          — exact / prefix / substring match on component IDs.
        filters : SearchFilters, optional
            Structured facet constraints applied to the candidate pool.
        limit : int
            Maximum number of hits to return.
        score_cutoff : float
            Drop hits scoring below this (0–1).
        """
        query = (query or "").strip()
        if not query:
            return []

        if mode == "id":
            return self._search_field(query, "component_id", filters, limit, score_cutoff)
        return self._search_vector(query, filters, limit, score_cutoff)

    def _search_vector(
        self,
        query: str,
        filters: Optional[SearchFilters],
        limit: int,
        score_cutoff: float,
    ) -> List[SearchHit]:
        # When a sparse model is registered, ``query`` runs dense and BM25
        # searches and fuses them with RRF. NOTE: fused scores are reciprocal-
        # rank values (small magnitude, ~0.01–0.03), not cosine similarities —
        # use them for ordering, not as an absolute "match %". With sparse
        # disabled, scores are cosine similarity (0–1).
        responses = self.client.query(
            collection_name=self.collection_name,
            query_text=query,
            query_filter=self._qdrant_filter(filters),
            limit=limit,
        )
        hits: List[SearchHit] = []
        for r in responses:
            if r.score < score_cutoff:
                continue
            payload = r.metadata or {}
            hits.append(
                SearchHit(
                    component_id=payload.get("component_id", ""),
                    score=float(r.score),
                    payload=payload,
                )
            )
        return hits

    def _search_field(
        self,
        query: str,
        field_name: str,
        filters: Optional[SearchFilters],
        limit: int,
        score_cutoff: float,
    ) -> List[SearchHit]:
        """Exact/prefix/substring match against a single payload string field."""
        q = query.casefold()
        hits: List[SearchHit] = []
        for rec in self.records:
            payload = rec.payload
            if filters is not None and not filters.matches(payload):
                continue
            value = str(payload.get(field_name, "")).casefold()
            if not value:
                continue
            if value == q:
                score = _SCORE_EXACT
            elif value.startswith(q):
                score = _SCORE_PREFIX
            elif q in value:
                score = _SCORE_SUBSTRING
            else:
                continue
            if score < score_cutoff:
                continue
            hits.append(SearchHit(rec.component_id, score, payload))
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:limit]

    def filter_only(self, filters: Optional[SearchFilters] = None) -> List[str]:
        """
        Return component IDs matching the facets, with no text query.

        Order follows the corpus (stable). The browse tree accepts such a set
        for facets that narrow within a data file, but no Explorer UI selection
        produces one: Hazard, Asset type and Source select whole files through
        :meth:`allowed_files`.
        """
        if filters is None or filters.is_empty():
            return [r.component_id for r in self.records]
        return [r.component_id for r in self.records if filters.matches(r.payload)]

    @functools.cached_property
    def file_facets(self) -> list[dict]:
        """
        One payload-like dict per distinct data file in the records.

        Each dict holds the file-level fields ``file_path``, ``hazard``,
        ``asset_type``, ``short_name`` and ``dataset``, so a file-level filter
        is evaluated with :meth:`SearchFilters.matches` once per file instead
        of once per record.
        """
        facets: dict[str, dict] = {}
        for r in self.records:
            if r.file_path not in facets:
                facets[r.file_path] = {
                    "file_path": r.file_path,
                    "hazard": r.hazard,
                    "asset_type": r.asset_type,
                    "short_name": r.short_name,
                    "dataset": r.dataset,
                }
        return list(facets.values())

    def allowed_files(self, filters: Optional[SearchFilters] = None) -> set:
        """
        Return the ``file_path`` values of the records that match the facets.

        Only the file-level facets (hazard, asset type, source and dataset) are
        evaluated, once for each entry of :attr:`file_facets`; the other fields
        of *filters* are ignored.
        """
        file_filters = (
            SearchFilters()
            if filters is None
            else dataclasses.replace(
                filters, category=None, group=None, subgroup=None, type=None
            )
        )
        return {
            row["file_path"] for row in self.file_facets if file_filters.matches(row)
        }

    # -- introspection ---------------------------------------------------------

    def summary(self) -> Dict[str, int]:
        by_hazard: Dict[str, int] = {}
        by_source: Dict[str, int] = {}
        for r in self.records:
            by_hazard[r.hazard] = by_hazard.get(r.hazard, 0) + 1
            by_source[r.short_name] = by_source.get(r.short_name, 0) + 1
        return {"total": len(self.records), **{f"hazard:{k}": v for k, v in by_hazard.items()},
                **{f"source:{k}": v for k, v in by_source.items()}}


# ─── CLI sanity harness ──────────────────────────────────────────────────────


def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Sanity-check the component semantic index from the command line."
    )
    parser.add_argument("query", nargs="?", default="", help="search query")
    parser.add_argument("--mode", default="description",
                        choices=["description", "id"])
    parser.add_argument("--hazard", default=None, help="facet: seismic / hurricane")
    parser.add_argument("--source", default=None, help="facet: source short name")
    parser.add_argument("--category", default=None,
                        help="facet: FEMA / HAZUS / SIMCENTER / RESEARCH")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--base", default=None,
                        help="override the data root (defaults to the packaged "
                             "dlml data)")
    parser.add_argument("--no-hybrid", action="store_true",
                        help="disable BM25 sparse fusion (dense-only, for A/B)")
    args = parser.parse_args()

    effective_base = args.base if args.base is not None else str(data_root())
    print(f"Building corpus from '{effective_base}' …")
    records = build_tree_corpus(args.base)
    print(f"  {len(records)} components from {len(tree_corpus_files(args.base))} files")

    sparse = None if args.no_hybrid else DEFAULT_SPARSE_MODEL
    mode_desc = "dense-only" if args.no_hybrid else f"hybrid (dense + {DEFAULT_SPARSE_MODEL})"
    print(f"Embedding with {DEFAULT_MODEL}, {mode_desc} (first run downloads models) …")
    index = SemanticIndex(records, sparse_model_name=sparse)

    summary = index.summary()
    print("Corpus summary:")
    for k, v in summary.items():
        print(f"  {k:<40} {v}")

    if not args.query:
        print("\nNo query given — corpus built successfully. Pass a query to search.")
        return

    filters = SearchFilters(
        hazard=args.hazard, source=args.source, category=args.category
    )
    print(f"\nQuery: {args.query!r}  (mode={args.mode})")
    hits = index.search(args.query, mode=args.mode, filters=filters, limit=args.limit)
    if not hits:
        print("  no results")
        return
    for i, h in enumerate(hits, 1):
        p = h.payload
        desc = (h.description[:90] + "…") if len(h.description) > 90 else h.description
        print(f"  {i:>2}. {h.score:.3f}  [{p.get('category')}] "
              f"{h.component_id}  ({p.get('subgroup_label')})")
        print(f"        {desc}")


if __name__ == "__main__":
    _main()
