import logging
import os
import re
from typing import List, Tuple, Optional, Iterable
import pandas as pd
from spacy.tokens import Span, Token
import networkx as nx

from amoc.graph import Graph, Node, Edge, NodeType, NodeSource
from amoc.viz.graph_plots import plot_amoc_triplets
from amoc.llm.vllm_client import VLLMClient
from amoc.nlp.spacy_utils import (
    get_concept_lemmas,
    canonicalize_node_text,
    get_content_words_from_sent,
)
from collections import deque
from amoc.config.paths import OUTPUT_ANALYSIS_DIR


def _sanitize_filename_component(component: str, max_len: int = 80) -> str:
    component = (component or "").replace("\n", " ").strip()
    component = component[:max_len]
    component = re.sub(r"[\\/:*?\"<>|]", "_", component)
    component = re.sub(r"\s+", "_", component)
    return component or "unknown"


class AMoCv4:
    GENERIC_RELATION_LABELS = {
        "has_property",
        "is_type_of",
        "is_part_of",
        "related_to",
        "has_attribute",
        "is_a",
        "appears",
        "contains",
        "includes",
        "include",
        "contain",
        "part_of",
        "associated_with",
        "is associated with",
        "|eot id|",
        "refers to",
        "is an",
        "is mentioned in",
    }

    ENFORCE_ATTACHMENT_CONSTRAINT = False
    ACTIVATION_MAX_DISTANCE = 2
    RELATION_BLACKLIST = {"describes", "is_at_stake"}

    def __init__(
        self,
        persona_description: str,
        story_text: str,
        vllm_client: VLLMClient,
        max_distance_from_active_nodes: int,
        max_new_concepts: int,
        max_new_properties: int,
        context_length: int,
        edge_forget: int,
        nr_relevant_edges: int,
        spacy_nlp,
        debug: bool = False,
        persona_age: Optional[int] = None,
        strict_reactivate_function: bool = True,
        strict_attachament_constraint: bool = True,
        single_anchor_hub: bool = True,
    ) -> None:
        self.persona = persona_description
        self.story_text = story_text
        self.client = vllm_client
        self.model_name = vllm_client.model_name
        self.persona_age = persona_age

        if not isinstance(story_text, str) or not story_text.strip():
            raise ValueError("story_text must be a non-empty string")

        self.max_distance_from_active_nodes = max_distance_from_active_nodes
        self.max_new_concepts = max_new_concepts
        self.max_new_properties = max_new_properties
        self.context_length = context_length
        self.edge_forget = edge_forget
        self.nr_relevant_edges = nr_relevant_edges

        self.graph = Graph()
        self.spacy_nlp = spacy_nlp

        if self.spacy_nlp is None:
            raise RuntimeError("AMoCv4 requires a spaCy nlp object (spacy_nlp).")

        self.debug = debug
        # Cache story lemmas for quick membership checks (currently unused)
        story_doc = self.spacy_nlp(story_text)
        self.story_lemmas = {tok.lemma_.lower() for tok in story_doc if tok.is_alpha}
        self._prev_active_nodes_for_plot: set[Node] = set()
        self._cumulative_deactivated_nodes_for_plot: set[Node] = set()
        self._viz_positions: dict[str, tuple[float, float]] = {}
        self._recently_deactivated_nodes_for_inference: set[Node] = set()
        self._anchor_nodes: set[Node] = set()
        self.strict_reactivate_function = strict_reactivate_function
        self.strict_attachament_constraint = strict_attachament_constraint
        self.single_anchor_hub = single_anchor_hub
        self._current_sentence_text: str = ""
        # Separate memory (cumulative) vs salience (active) graphs for auditing.
        self.cumulative_graph = nx.MultiDiGraph()
        self.active_graph = nx.MultiDiGraph()
        # Stable triplet introduction index: (subj, rel, obj) -> introduced_at_sentence
        self._triplet_intro: dict[tuple[str, str, str], int] = {}

    def _node_token_for_matrix(self, node: Node) -> str:
        return (node.get_text_representer() or "").strip().lower()

    def _distances_from_sources_active_edges(
        self, sources: set[Node], max_distance: int
    ) -> dict[Node, int]:
        if not sources:
            return {}
        distances: dict[Node, int] = {s: 0 for s in sources}
        queue: deque[Node] = deque(sources)
        while queue:
            node = queue.popleft()
            dist = distances[node]
            if dist >= max_distance:
                continue
            for edge in node.edges:
                if not edge.active:
                    continue
                neighbor = (
                    edge.dest_node if edge.source_node == node else edge.source_node
                )
                if neighbor in distances:
                    continue
                distances[neighbor] = dist + 1
                queue.append(neighbor)
        return distances

    def _record_sentence_activation(
        self,
        sentence_id: int,
        explicit_nodes: List[Node],
        newly_inferred_nodes: set[Node],
    ) -> None:
        def _to_landscape_score(raw_score: float) -> float:
            # Transform AMoC "distance" style (0=most active) into Landscape style (5=most active).
            val = 5.0 - float(raw_score)
            if val < 0.0:
                return 0.0
            if val > 5.0:
                return 5.0
            return val

        explicit_set = set(explicit_nodes)
        distances = self._distances_from_sources_active_edges(
            explicit_set, max_distance=self.ACTIVATION_MAX_DISTANCE
        )

        token_to_raw_score: dict[str, int] = {}
        node_raw_score: dict[Node, int] = {}

        # Step 2: explicit nodes reset to 0 (non-negotiable)
        for node in explicit_set:
            token = self._node_token_for_matrix(node)
            if token:
                token_to_raw_score[token] = 0
                node_raw_score[node] = 0

        # Step 5: newly inferred nodes start at 1 (never 0)
        for node in newly_inferred_nodes:
            if node in explicit_set:
                continue
            token = self._node_token_for_matrix(node)
            if token:
                token_to_raw_score[token] = 1
                node_raw_score[node] = 1

        # Step 3/4: carried-over nodes within range, score = distance
        for node, dist in distances.items():
            if node in explicit_set:
                continue
            if dist <= 0:
                continue
            token = self._node_token_for_matrix(node)
            if not token:
                continue
            if token in token_to_raw_score:
                continue
            token_to_raw_score[token] = dist
            node_raw_score[node] = dist

        # Convert node scores to Landscape scale and record.
        for token, raw_score in token_to_raw_score.items():
            self._amoc_matrix_records.append(
                {
                    "sentence": sentence_id,
                    "token": token,
                    "score": _to_landscape_score(raw_score),
                }
            )

        # Add verb (edge-label) activations: take the max activation of connected nodes minus 0.5.
        verb_scores: dict[str, float] = {}
        for edge in self.graph.edges:
            if not edge.active:
                continue
            label = (edge.label or "").strip().lower()
            if not label:
                continue
            src_tok = self._node_token_for_matrix(edge.source_node)
            dst_tok = self._node_token_for_matrix(edge.dest_node)
            if not src_tok or not dst_tok:
                continue
            src_raw = node_raw_score.get(edge.source_node, edge.source_node.score)
            dst_raw = node_raw_score.get(edge.dest_node, edge.dest_node.score)
            src_act = _to_landscape_score(src_raw)
            dst_act = _to_landscape_score(dst_raw)
            verb_act = max(src_act, dst_act) - 0.5
            if verb_act < 0.0:
                verb_act = 0.0
            prev = verb_scores.get(label)
            if prev is None or verb_act > prev:
                verb_scores[label] = verb_act

        for token, score in verb_scores.items():
            self._amoc_matrix_records.append(
                {"sentence": sentence_id, "token": token, "score": score}
            )

    def _infer_edges_to_recently_deactivated(
        self,
        current_sentence_nodes: List[Node],
        current_sentence_words: List[str],
        current_text: str,
    ) -> List[Edge]:
        if not self.ENFORCE_ATTACHMENT_CONSTRAINT:
            return []
        recent = [
            n
            for n in self._recently_deactivated_nodes_for_inference
            if n in self.graph.nodes
        ]
        if not recent or not current_sentence_nodes:
            return []

        candidate_pairs: set[frozenset[Node]] = set()
        for node in current_sentence_nodes:
            for other in recent:
                if node == other:
                    continue
                if self._has_edge_between(node, other):
                    continue
                candidate_pairs.add(frozenset((node, other)))
        if not candidate_pairs:
            return []

        nodes_for_prompt = {n for pair in candidate_pairs for n in pair}

        def _node_line(node: Node) -> str:
            return f" - ({node.get_text_representer()}, {node.node_type})\n"

        nodes_from_text = "".join(
            _node_line(n)
            for n in sorted(nodes_for_prompt, key=lambda x: x.get_text_representer())
        )
        graph_nodes_repr = self.graph.get_nodes_str(list(nodes_for_prompt))
        graph_edges_repr, _ = self.graph.get_edges_str(
            list(nodes_for_prompt), only_text_based=False
        )

        try:
            new_relationships = self.client.get_new_relationships(
                nodes_from_text,
                graph_nodes_repr,
                graph_edges_repr,
                current_text,
                self.persona,
            )
        except Exception:
            logging.error("Targeted LLM edge inference failed", exc_info=True)
            return []

        added: List[Edge] = []
        for idx, relationship in enumerate(new_relationships):
            if relationship is None or isinstance(relationship, (int, float, bool)):
                continue
            if isinstance(relationship, dict):
                subj = relationship.get("subject") or relationship.get("head")
                rel = relationship.get("relation") or relationship.get("predicate")
                obj = relationship.get("object") or relationship.get("tail")
                if not (subj and rel and obj):
                    continue
                relationship = (str(subj), str(rel), str(obj))
            if not isinstance(relationship, (list, tuple)) or len(relationship) != 3:
                continue

            subj, rel, obj = relationship
            subj = self._normalize_endpoint_text(subj, is_subject=True) or None
            obj = self._normalize_endpoint_text(obj, is_subject=False) or None
            if subj is None or obj is None:
                continue
            if not subj or not obj:
                continue
            if not isinstance(subj, str) or not isinstance(obj, str):
                continue
            edge_label = rel.replace("(edge)", "").strip()
            if not self._is_valid_relation_label(edge_label):
                continue

            subj_node = self._find_node_by_text(subj, nodes_for_prompt)
            obj_node = self._find_node_by_text(obj, nodes_for_prompt)
            if subj_node is None or obj_node is None:
                continue
            pair_key = frozenset((subj_node, obj_node))
            if pair_key not in candidate_pairs:
                continue
            if self._has_edge_between(subj_node, obj_node):
                continue

            edge = self._add_edge(subj_node, obj_node, edge_label, self.edge_forget)
            if edge:
                added.append(edge)
        return added

    def _passes_attachment_constraint(
        self,
        subject: str,
        obj: str,
        current_sentence_words: List[str],
        current_sentence_nodes: List[Node],
        graph_active_nodes: List[Node],
        graph_active_edge_nodes: Optional[set[Node]] = None,
    ) -> bool:
        # Connectivity constraint: avoid introducing edges that would form a new
        # disconnected component. Accept an edge only if it (a) touches the current
        # sentence and (b) attaches to a node already in the active graph (or, if
        # no edges are active, any existing node in the graph). This guard is always
        # enforced to keep the graph as a single component, regardless of the flag.

        active_edge_nodes = graph_active_edge_nodes or set()
        active_nodes_pool: set[Node] = set(graph_active_nodes) | active_edge_nodes
        anchor_nodes = self._anchor_nodes

        # If the graph is empty, allow seeding.
        if not self.graph.nodes:
            return True

        # If no active edges exist, fall back to the current anchor to keep
        # connectivity. If the anchor is empty (pre-seed), we allow the first edge.
        if not active_edge_nodes:
            active_nodes_pool |= anchor_nodes

        subject = canonicalize_node_text(self.spacy_nlp, subject)
        obj = canonicalize_node_text(self.spacy_nlp, obj)

        def _lemma_key(text: str) -> tuple[str, ...]:
            return tuple(get_concept_lemmas(self.spacy_nlp, text))

        sentence_lemma_keys = {tuple(n.lemmas) for n in current_sentence_nodes}
        active_lemma_keys = {tuple(n.lemmas) for n in active_nodes_pool}
        anchor_lemma_keys = {tuple(n.lemmas) for n in anchor_nodes}

        subj_key = _lemma_key(subject)
        obj_key = _lemma_key(obj)

        touches_sentence = (
            subject in current_sentence_words
            or obj in current_sentence_words
            or subj_key in sentence_lemma_keys
            or obj_key in sentence_lemma_keys
        )
        touches_active = subj_key in active_lemma_keys or obj_key in active_lemma_keys
        attaches_anchor = (
            not anchor_lemma_keys
            or subj_key in anchor_lemma_keys
            or obj_key in anchor_lemma_keys
        )

        if self.strict_attachament_constraint:
            return touches_sentence and touches_active and attaches_anchor
        # Relaxed guard: allow edges that touch either the current sentence OR
        # the active neighborhood, but still require they attach to the anchor
        # (once seeded) so the graph stays connected.
        return attaches_anchor and (touches_sentence or touches_active)

    def _add_edge(
        self,
        source_node: Node,
        dest_node: Node,
        label: str,
        edge_forget: int,
        created_at_sentence: Optional[int] = None,
    ) -> Optional[Edge]:
        # Keep the graph as a single connected component: once seeded, at least
        # one endpoint of every new edge must already exist in the graph.
        if self.graph.nodes:
            if (
                source_node not in self.graph.nodes
                and dest_node not in self.graph.nodes
            ):
                return None

        # Enforce anchor connectivity: if an anchor exists, at least one endpoint
        # must be in it. Otherwise, reject.
        if self._anchor_nodes and not (
            source_node in self._anchor_nodes or dest_node in self._anchor_nodes
        ):
            # Anchor constraint failure: log and drop
            self._anchor_drop_log.append(
                (
                    self._current_sentence_index or -1,
                    self._current_sentence_text,
                    source_node.get_text_representer(),
                    label,
                    dest_node.get_text_representer(),
                )
            )
            return None

        use_sentence = (
            created_at_sentence
            if created_at_sentence is not None
            else getattr(self, "_current_sentence_index", None)
        )
        edge = self.graph.add_edge(
            source_node,
            dest_node,
            label,
            edge_forget,
            created_at_sentence=use_sentence,
        )
        if edge:
            trip_id = (
                edge.source_node.get_text_representer(),
                edge.label,
                edge.dest_node.get_text_representer(),
            )
            if trip_id not in self._triplet_intro:
                self._triplet_intro[trip_id] = (
                    use_sentence if use_sentence is not None else -1
                )
            self._record_edge_in_graphs(edge, self._current_sentence_index)
            # Seed anchor on the very first edge. If single-anchor mode is on,
            # keep only the initial hub; otherwise grow the anchor set when touched.
            if not self._anchor_nodes:
                if self.single_anchor_hub:
                    self._anchor_nodes = {source_node}
                else:
                    self._anchor_nodes.update({source_node, dest_node})
            elif not self.single_anchor_hub and (
                source_node in self._anchor_nodes or dest_node in self._anchor_nodes
            ):
                self._anchor_nodes.update({source_node, dest_node})
        return edge

    def reset_graph(self) -> None:
        self.graph = Graph()
        self._anchor_nodes = set()

    def resolve_pronouns(self, text: str) -> str:
        resolved = self.client.resolve_pronouns(text, self.persona)
        if not isinstance(resolved, str) or not resolved.strip():
            return text
        return resolved

    def _graph_edges_to_triplets(
        self, only_active: bool = False
    ) -> List[Tuple[str, str, str]]:
        triplets: List[Tuple[str, str, str]] = []
        for edge in self.graph.edges:
            if only_active and not edge.active:
                continue
            if not edge.label or not str(edge.label).strip():
                continue
            if edge.source_node == edge.dest_node:
                continue
            triplets.append(
                (
                    edge.source_node.get_text_representer(),
                    edge.label,
                    edge.dest_node.get_text_representer(),
                )
            )
        return triplets

    # Step 5 from paper - only explicit nodes from the current sentence stay active
    def _restrict_active_to_current_explicit(self, explicit_nodes: List[Node]) -> None:
        explicit_set = set(explicit_nodes)
        inactive_score = self.max_distance_from_active_nodes + 1
        for node in self.graph.nodes:
            if node in explicit_set and node.node_source == NodeSource.TEXT_BASED:
                node.score = 0
            else:
                node.score = inactive_score

    def _get_nodes_with_active_edges(self) -> set[Node]:
        active_nodes: set[Node] = set()
        for edge in self.graph.edges:
            if edge.active:
                active_nodes.add(edge.source_node)
                active_nodes.add(edge.dest_node)
        return active_nodes

    def _normalize_label(self, label: str) -> str:
        norm = (label or "").strip().lower()
        norm = re.sub(r"[\s\-]+", "_", norm)
        return norm

    def _edge_key(self, edge: Edge) -> tuple[str, str, str]:
        return (
            edge.source_node.get_text_representer(),
            edge.dest_node.get_text_representer(),
            edge.label,
        )

    def _record_edge_in_graphs(self, edge: Edge, sentence_idx: Optional[int]) -> None:
        u, v, lbl = self._edge_key(edge)
        introduced = self._triplet_intro.get((u, lbl, v))
        if introduced is None:
            introduced = edge.created_at_sentence if edge.created_at_sentence is not None else -1
        self._triplet_intro[(u, lbl, v)] = int(introduced)

        # Only update last_active when the edge is currently active in this sentence.
        existing_last_active = None
        edge_key = f"{lbl}__introduced_{introduced}"
        if self.cumulative_graph.has_edge(u, v, key=edge_key):
            existing_last_active = self.cumulative_graph[u][v][edge_key].get("last_active_sentence")
        last_active = existing_last_active
        if edge.active and sentence_idx is not None:
            last_active = sentence_idx
        edge_key = f"{lbl}__introduced_{introduced}"
        # Cumulative memory (never removed)
        if self.cumulative_graph.has_edge(u, v, key=edge_key):
            # Update attributes if already present
            data = self.cumulative_graph[u][v][edge_key]
            if data.get("introduced_at_sentence", introduced) == -1:
                data["introduced_at_sentence"] = introduced
            data["last_active_sentence"] = last_active
        else:
            self.cumulative_graph.add_edge(
                u,
                v,
                key=edge_key,
                relation=lbl,
                introduced_at_sentence=int(introduced),
                last_active_sentence=int(last_active),
            )
        # Active projection (salience)
        if edge.active:
            self.active_graph.add_edge(
                u,
                v,
                key=edge_key,
                relation=lbl,
                introduced_at_sentence=int(introduced),
                last_active_sentence=int(last_active),
            )
        else:
            if self.active_graph.has_edge(u, v, key=edge_key):
                try:
                    self.active_graph.remove_edge(u, v, key=edge_key)
                except Exception:
                    pass

    def _graph_to_triplets(self, graph: nx.MultiDiGraph) -> List[Tuple[str, str, str]]:
        trips: List[Tuple[str, str, str]] = []
        for u, v, key, data in graph.edges(keys=True, data=True):
            rel = data.get("relation") or key
            trips.append((u, rel, v))
        return trips

    def _is_generic_relation(self, label: str) -> bool:
        norm = self._normalize_label(label)
        return norm in self.GENERIC_RELATION_LABELS

    def _is_blacklisted_relation(self, label: str) -> bool:
        norm = self._normalize_label(label)
        return norm in self.RELATION_BLACKLIST

    def _is_verb_relation(self, label: str) -> bool:
        doc = self.spacy_nlp(label)
        has_verb = False
        for tok in doc:
            if not getattr(tok, "is_alpha", False):
                continue
            if tok.pos_ in {"ADJ", "NOUN", "PROPN", "ADV"}:
                return False
            if tok.pos_ in {"VERB", "AUX"}:
                has_verb = True
        return has_verb

    def _is_valid_relation_label(self, label: str) -> bool:
        if not label:
            return False
        if self._is_generic_relation(label):
            return False
        if self._is_blacklisted_relation(label):
            return False
        if not self._is_verb_relation(label):
            return False
        return True

    def _normalize_endpoint_text(self, text: str, is_subject: bool) -> Optional[str]:
        if not text:
            return None
        doc = self.spacy_nlp(text)
        if not doc:
            return None
        allowed_subject = {"NOUN", "PROPN", "PRON"}
        allowed_object = {"NOUN", "PROPN", "PRON", "ADJ"}
        for tok in doc:
            if not getattr(tok, "is_alpha", False):
                continue
            pos = tok.pos_
            if is_subject and pos not in allowed_subject:
                continue
            if not is_subject and pos not in allowed_object:
                continue
            lemma = (getattr(tok, "lemma_", "") or "").strip().lower()
            if not lemma or lemma in self.spacy_nlp.Defaults.stop_words:
                continue
            return lemma
        return None

    def _has_edge_between(self, a: Node, b: Node) -> bool:
        for edge in self.graph.edges:
            if (edge.source_node == a and edge.dest_node == b) or (
                edge.source_node == b and edge.dest_node == a
            ):
                return True
        return False

    def _find_node_by_text(
        self, text: str, candidates: Iterable[Node]
    ) -> Optional[Node]:
        canon = canonicalize_node_text(self.spacy_nlp, text)
        lemmas = tuple(get_concept_lemmas(self.spacy_nlp, canon))
        for node in candidates:
            if lemmas == tuple(node.lemmas):
                return node
        return None

    def _appears_in_story(self, text: str) -> bool:
        if not text:
            return False
        doc = self.spacy_nlp(text)
        return any(
            tok.lemma_.lower() in self.story_lemmas for tok in doc if tok.is_alpha
        )

    def _classify_canonical_node_text(self, canon: str) -> Optional[NodeType]:
        if not canon:
            return None
        doc = self.spacy_nlp(canon)
        if not doc:
            return None
        token = next((t for t in doc if getattr(t, "is_alpha", False)), None) or doc[0]
        lemma = (getattr(token, "lemma_", "") or "").lower()
        if lemma in self.spacy_nlp.Defaults.stop_words:
            return None
        if token.pos_ in {"NOUN", "PROPN"}:
            return NodeType.CONCEPT
        if token.pos_ == "ADJ":
            return NodeType.PROPERTY
        return None

    def _canonicalize_and_classify_node_text(
        self, text: str
    ) -> tuple[str, Optional[NodeType]]:
        canon = canonicalize_node_text(self.spacy_nlp, text)
        return canon, self._classify_canonical_node_text(canon)

    def _plot_graph_snapshot(
        self,
        sentence_index: int,
        sentence_text: str,
        output_dir: Optional[str],
        highlight_nodes: Optional[Iterable[str]],
        deactivated_concepts: Optional[List[str]] = None,
        new_nodes: Optional[List[str]] = None,
        only_active: bool = False,
        largest_component_only: bool = False,
        mode: str = "sentence_active",
        triplets_override: Optional[List[Tuple[str, str, str]]] = None,
    ) -> None:
        # Route per-sentence plots into mode-specific subfolders for clarity.
        plot_dir = output_dir
        if output_dir and mode in {"sentence_active", "sentence_cumulative"}:
            subdir = "active" if mode == "sentence_active" else "cummulative"
            plot_dir = os.path.join(output_dir, subdir)
        triplets = (
            triplets_override
            if triplets_override is not None
            else self._graph_edges_to_triplets(only_active=only_active)
        )
        age_for_filename = self.persona_age if self.persona_age is not None else -1
        try:
            saved_path = plot_amoc_triplets(
                triplets=(
                    self._graph_edges_to_triplets(only_active=True)
                    if only_active
                    else triplets
                ),
                persona=self.persona,
                model_name=self.model_name,
                age=age_for_filename,
                blue_nodes=highlight_nodes,
                output_dir=plot_dir,
                step_tag=(
                    f"sent{sentence_index+1}_{mode}"
                    if mode
                    else f"sent{sentence_index+1}"
                ),
                sentence_text=sentence_text,
                deactivated_concepts=deactivated_concepts,
                new_nodes=new_nodes,
                largest_component_only=largest_component_only,
                positions=self._viz_positions,
            )
            if triplets:
                logging.info(
                    "[Plot] Saved sentence %d graph to %s", sentence_index, saved_path
                )
            else:
                logging.info(
                    "[Plot] Sentence %d graph skipped (no active edges)", sentence_index
                )
        except Exception:
            logging.error("Failed to plot graph snapshot", exc_info=True)

    def analyze(
        self,
        replace_pronouns: bool = True,
        plot_after_each_sentence: bool = False,
        graphs_output_dir: Optional[str] = None,
        highlight_nodes: Optional[Iterable[str]] = None,
        matrix_suffix: Optional[str] = None,
        largest_component_only: bool = False,
    ) -> List[Tuple[str, str, str]]:
        if not hasattr(self, "_amoc_matrix_records"):
            self._amoc_matrix_records = []
        # Always start each analyze run with a fresh layout cache so plots begin
        # from a clean radial arrangement.
        self._viz_positions = {}
        self._cumulative_deactivated_nodes_for_plot = set()
        self._prev_active_nodes_for_plot = set()

        def _clean_resolved_sentence(orig_text: str, candidate: str) -> str:
            if not isinstance(candidate, str) or not candidate.strip():
                return orig_text
            cleaned = re.sub(r"<[^>]+>", " ", candidate)
            cleaned = re.sub(r"\s+", " ", cleaned).strip()

            # Pick the candidate sentence with the highest content-word overlap
            # with the original to avoid prompt echoes in the header.
            orig_doc = self.spacy_nlp(orig_text)
            orig_tokens = {t.lemma_.lower() for t in orig_doc if t.is_alpha}
            best_sent = None
            best_overlap = -1
            cand_doc = self.spacy_nlp(cleaned)
            for sent in cand_doc.sents:
                toks = {t.lemma_.lower() for t in sent if t.is_alpha}
                overlap = len(orig_tokens & toks)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_sent = sent.text.strip()

            chosen = best_sent or cleaned

            # Trim runaway echoes to a reasonable length.
            max_len = max(len(orig_text) * 2 + 40, 400)
            if len(chosen) > max_len:
                chosen = chosen[:max_len].rstrip(" ,.;") + "..."
            return chosen or orig_text

        doc = self.spacy_nlp(self.story_text)
        resolved_sentences: list[tuple[Span, str, str]] = []
        for orig_sent in doc.sents:
            resolved_text = orig_sent.text
            if replace_pronouns:
                candidate = self.resolve_pronouns(orig_sent.text)
                if isinstance(candidate, str) and candidate.strip():
                    resolved_text = _clean_resolved_sentence(orig_sent.text, candidate)
            resolved_doc = self.spacy_nlp(resolved_text)
            if not resolved_doc:
                resolved_text = orig_sent.text
                resolved_doc = self.spacy_nlp(resolved_text)
            resolved_span = resolved_doc[0 : len(resolved_doc)]
            resolved_sentences.append((resolved_span, resolved_text, orig_sent.text))

        prev_sentences: list[str] = []
        current_sentence = ""
        self._sentence_triplets: list[tuple[int, str, str, str, str, bool, bool]] = (
            []
        )  # sentence_idx, sentence_text, subj, rel, obj, active, anchor_kept
        for i, (sent, resolved_text, original_text) in enumerate(resolved_sentences):
            # Working-memory projection is rebuilt each sentence.
            self.active_graph = nx.MultiDiGraph()
            self._current_sentence_index = i + 1
            self._current_sentence_text = original_text
            self._anchor_drop_log: list[tuple[int, str, str, str, str]] = (
                []
            )  # sent_idx, sent_text, subj, rel, obj
            nodes_before_sentence = set(self.graph.nodes)
            logging.info("Processing sentence %d: %s", i, resolved_text)
            if i == 0:
                current_sentence = sent
                prev_sentences.append(resolved_text)
                self.init_graph(sent)

                (
                    current_sentence_text_based_nodes,
                    current_sentence_text_based_words,
                ) = self.get_senteces_text_based_nodes(
                    [sent], create_unexistent_nodes=True
                )
                inferred_concept_relationships, inferred_property_relationships = (
                    self.infer_new_relationships_step_0(sent)
                )

                self.add_inferred_relationships_to_graph_step_0(
                    inferred_concept_relationships, NodeType.CONCEPT, sent
                )
                self.add_inferred_relationships_to_graph_step_0(
                    inferred_property_relationships, NodeType.PROPERTY, sent
                )
                self._restrict_active_to_current_explicit(
                    current_sentence_text_based_nodes
                )
                self.graph.set_nodes_score_based_on_distance_from_active_nodes(
                    current_sentence_text_based_nodes
                )
            else:
                added_edges = []
                current_sentence = sent
                prev_sentences.append(resolved_text)
                if len(prev_sentences) > self.context_length:
                    prev_sentences.pop(0)

                current_sentence_text_based_nodes, current_sentence_text_based_words = (
                    self.get_senteces_text_based_nodes(
                        [current_sentence], create_unexistent_nodes=True
                    )
                )

                current_all_text = resolved_text
                # Step 3: build active subgraph using only explicit (text-based) nodes.
                graph_active_nodes = self.graph.get_active_nodes(
                    self.max_distance_from_active_nodes, only_text_based=True
                )
                active_nodes_text = self.graph.get_nodes_str(graph_active_nodes)
                active_nodes_edges_text, _ = self.graph.get_edges_str(
                    graph_active_nodes, only_text_based=True
                )

                nodes_from_text = ""
                for idx, node in enumerate(current_sentence_text_based_nodes):
                    nodes_from_text += f" - ({current_sentence_text_based_words[idx]}, {node.node_type})\n"

                new_relationships = self.client.get_new_relationships(
                    nodes_from_text,  # 1. Nodes from Text
                    active_nodes_text,  # 2. Nodes from Graph (explicit only)
                    active_nodes_edges_text,  # 3. Edges from Graph (explicit only)
                    current_all_text,  # 4. Text
                    self.persona,  # 5. Persona
                )

                text_based_activated_nodes = current_sentence_text_based_nodes
                sentence_lemma_keys = {
                    tuple(n.lemmas) for n in current_sentence_text_based_nodes
                }
                for idx, relationship in enumerate(new_relationships):
                    # Skip None or scalar junk (int, float, bool, etc.)
                    if relationship is None or isinstance(
                        relationship, (int, float, bool)
                    ):
                        logging.error(
                            f"[AMoC] Skipping non-iterable relationship at {idx}: {relationship!r}"
                        )
                        continue

                    # If relationship is a dict, try to convert it
                    if isinstance(relationship, dict):
                        subj = relationship.get("subject") or relationship.get("head")
                        rel = relationship.get("relation") or relationship.get(
                            "predicate"
                        )
                        obj = relationship.get("object") or relationship.get("tail")
                        if not (subj and rel and obj):
                            logging.error(
                                f"[AMoC] Skipping malformed dict relationship at {idx}: {relationship!r}"
                            )
                            continue
                        relationship = (str(subj), str(rel), str(obj))

                    # Must be list/tuple from this point on
                    if not isinstance(relationship, (list, tuple)):
                        logging.error(
                            f"[AMoC] Skipping unexpected relationship type at {idx}: {type(relationship)} → {relationship!r}"
                        )
                        continue

                    # Must have exactly 3 elements
                    if len(relationship) != 3:
                        logging.error(
                            f"[AMoC] Skipping relationship with wrong length at {idx}: {relationship!r}"
                        )
                        continue

                    # Unpack
                    subj, rel, obj = relationship

                    subj = self._normalize_endpoint_text(subj, is_subject=True) or None
                    obj = self._normalize_endpoint_text(obj, is_subject=False) or None
                    if subj is None or obj is None:
                        continue
                    # Validate subject/object strings
                    if not subj or not obj:
                        continue
                    if subj == obj:
                        continue
                    if not isinstance(subj, str) or not isinstance(obj, str):
                        continue

                    if not self._passes_attachment_constraint(
                        subj,
                        obj,
                        current_sentence_text_based_words,
                        current_sentence_text_based_nodes,
                        graph_active_nodes,
                        self._get_nodes_with_active_edges(),
                    ):
                        continue

                    # Continue with your original code
                    source_node = self.get_node_from_new_relationship(
                        subj,
                        graph_active_nodes,
                        current_sentence_text_based_nodes,
                        current_sentence_text_based_words,
                        node_source=NodeSource.TEXT_BASED,
                        create_node=True,
                    )

                    dest_node = self.get_node_from_new_relationship(
                        obj,
                        graph_active_nodes,
                        current_sentence_text_based_nodes,
                        current_sentence_text_based_words,
                        node_source=NodeSource.TEXT_BASED,
                        create_node=True,
                    )
                    edge_label = rel.replace("(edge)", "").strip()
                    if not self._is_valid_relation_label(edge_label):
                        continue
                    if source_node is None or dest_node is None:
                        continue

                    if tuple(source_node.lemmas) in sentence_lemma_keys:
                        source_node.node_source = NodeSource.TEXT_BASED
                    if tuple(dest_node.lemmas) in sentence_lemma_keys:
                        dest_node.node_source = NodeSource.TEXT_BASED

                    potential_new_edge = self._add_edge(
                        source_node, dest_node, edge_label, self.edge_forget
                    )
                    if potential_new_edge:
                        added_edges.append(potential_new_edge)

                # infer new relationships logic...
                inferred_concept_relationships, inferred_property_relationships = (
                    self.infer_new_relationships(
                        current_all_text,
                        current_sentence_text_based_nodes,
                        current_sentence_text_based_words,
                        self.graph.get_nodes_str(
                            self.graph.get_active_nodes(
                                self.max_distance_from_active_nodes,
                                only_text_based=True,
                            )
                        ),
                        self.graph.get_edges_str(
                            self.graph.get_active_nodes(
                                self.max_distance_from_active_nodes,
                                only_text_based=True,
                            ),
                            only_text_based=True,
                        )[0],
                    )
                )

                self.add_inferred_relationships_to_graph(
                    inferred_concept_relationships,
                    NodeType.CONCEPT,
                    current_sentence_text_based_nodes,
                    current_sentence_text_based_words,
                    graph_active_nodes,
                    added_edges,
                )
                self.add_inferred_relationships_to_graph(
                    inferred_property_relationships,
                    NodeType.PROPERTY,
                    current_sentence_text_based_nodes,
                    current_sentence_text_based_words,
                    graph_active_nodes,
                    added_edges,
                )

                if self.ENFORCE_ATTACHMENT_CONSTRAINT:
                    targeted_edges = self._infer_edges_to_recently_deactivated(
                        current_sentence_text_based_nodes,
                        current_sentence_text_based_words,
                        current_all_text,
                    )
                    added_edges.extend(targeted_edges)

                self.graph.set_nodes_score_based_on_distance_from_active_nodes(
                    text_based_activated_nodes
                )
                self.reactivate_relevant_edges(
                    self.graph.get_active_nodes(
                        self.max_distance_from_active_nodes, only_text_based=True
                    ),
                    " ".join(prev_sentences),
                    added_edges,
                )
                self._restrict_active_to_current_explicit(
                    current_sentence_text_based_nodes
                )
                self.graph.set_nodes_score_based_on_distance_from_active_nodes(
                    current_sentence_text_based_nodes
                )

            if self.debug:
                logging.info(
                    "Active graph after sentence %d:\n%s",
                    i,
                    self.graph.get_active_graph_repr(),
                )

            sentence_id = i + 1
            newly_inferred_nodes = {
                n
                for n in (set(self.graph.nodes) - nodes_before_sentence)
                if n.node_source == NodeSource.INFERENCE_BASED
            }
            # Refresh active projection for this step
            self._record_sentence_activation(
                sentence_id=sentence_id,
                explicit_nodes=current_sentence_text_based_nodes,
                newly_inferred_nodes=newly_inferred_nodes,
            )

            current_active_nodes = self._get_nodes_with_active_edges()
            if i == 0:
                deactivated_concepts = None
                recently_deactivated_nodes: set[Node] = set()
                new_nodes_for_plot = None
            else:
                appeared = current_active_nodes - self._prev_active_nodes_for_plot
                gone = self._prev_active_nodes_for_plot - current_active_nodes
                self._cumulative_deactivated_nodes_for_plot.update(gone)
                self._cumulative_deactivated_nodes_for_plot.difference_update(
                    current_active_nodes
                )
                deactivated_concepts = sorted(
                    {
                        node.get_text_representer()
                        for node in self._cumulative_deactivated_nodes_for_plot
                    }
                )
                recently_deactivated_nodes = set(gone)
                new_nodes_for_plot = sorted(
                    {node.get_text_representer() for node in appeared}
                )
            # Only keep the deactivated set for targeted inference when the
            # attachment constraint is enforced.
            if self.ENFORCE_ATTACHMENT_CONSTRAINT:
                self._recently_deactivated_nodes_for_inference = (
                    recently_deactivated_nodes
                )
            else:
                self._recently_deactivated_nodes_for_inference = set()
            self._prev_active_nodes_for_plot = current_active_nodes

            if plot_after_each_sentence:
                # Active (salience) view
                self._plot_graph_snapshot(
                    sentence_index=i,
                    sentence_text=sent.text,
                    output_dir=graphs_output_dir,
                    highlight_nodes=highlight_nodes,
                    deactivated_concepts=deactivated_concepts,
                    new_nodes=new_nodes_for_plot,
                    only_active=True,
                    largest_component_only=largest_component_only,
                    mode="sentence_active",
                    triplets_override=self._graph_to_triplets(self.active_graph),
                )
                # Cumulative memory view
                self._plot_graph_snapshot(
                    sentence_index=i,
                    sentence_text=sent.text,
                    output_dir=graphs_output_dir,
                    highlight_nodes=highlight_nodes,
                    deactivated_concepts=deactivated_concepts,
                    new_nodes=new_nodes_for_plot,
                    only_active=False,
                    largest_component_only=largest_component_only,
                    mode="sentence_cumulative",
                    triplets_override=self._graph_to_triplets(self.cumulative_graph),
                )

            # Capture triplets for this sentence (all edges, with current active flag)
            for edge in self.graph.edges:
                self._sentence_triplets.append(
                    (
                        self._current_sentence_index,
                        original_text,
                        edge.source_node.get_text_representer(),
                        edge.label,
                        edge.dest_node.get_text_representer(),
                        edge.active,
                        True,  # anchor_kept
                    )
                )
            for sent_idx, sent_text, subj, rel, obj in getattr(
                self, "_anchor_drop_log", []
            ):
                self._sentence_triplets.append(
                    (
                        sent_idx,
                        sent_text,
                        subj,
                        rel,
                        obj,
                        False,  # inactive/not added
                        False,  # anchor_kept flag
                    )
                )

        # save score matrix
        df = pd.DataFrame(self._amoc_matrix_records)
        matrix = (
            df.pivot(index="token", columns="sentence", values="score")
            .sort_index()
            .fillna(0.0)
        )

        matrix_dir = os.path.join(OUTPUT_ANALYSIS_DIR, "matrix")
        os.makedirs(matrix_dir, exist_ok=True)
        safe_model = _sanitize_filename_component(self.model_name, max_len=60)
        safe_persona = _sanitize_filename_component(self.persona, max_len=60)
        age_for_filename = self.persona_age if self.persona_age is not None else -1
        suffix = (
            f"_{_sanitize_filename_component(matrix_suffix)}" if matrix_suffix else ""
        )
        matrix_filename = (
            f"amoc_matrix_{safe_model}_{safe_persona}_{age_for_filename}{suffix}.csv"
        )
        matrix_path = os.path.join(matrix_dir, matrix_filename)

        matrix.to_csv(matrix_path)
        logging.info(
            "[Matrix] Saved activation matrix for persona '%s' to %s",
            self.persona,
            matrix_path,
        )
        logging.info("AMoC activation matrix:\n%s", matrix.to_string())
        # Collect final active triplets
        final_triplets = []
        for u, v, key, data in self.active_graph.edges(keys=True, data=True):
            introduced = data.get("introduced_at_sentence", -1)
            last_active = data.get("last_active_sentence", -1)
            rel = data.get("relation") or key
            final_triplets.append(
                (
                    u,
                    rel,
                    v,
                    True,
                    introduced,
                    last_active,
                )
            )

        cumulative_triplets = self._graph_to_triplets(self.cumulative_graph)

        return final_triplets, self._sentence_triplets, cumulative_triplets

    def infer_new_relationships_step_0(
        self, sent: Span
    ) -> Tuple[List[Tuple[str, str, str]], List[Tuple[str, str, str]]]:
        current_sentence_text_based_nodes, current_sentence_text_based_words = (
            self.get_senteces_text_based_nodes([sent], create_unexistent_nodes=False)
        )

        nodes_from_text = ""
        for i, node in enumerate(current_sentence_text_based_nodes):
            nodes_from_text += (
                f" - ({current_sentence_text_based_words[i]}, {node.node_type})\n"
            )

        for _ in range(3):
            try:
                object_properties_dict = (
                    self.client.infer_objects_and_properties_first_sentence(
                        nodes_from_text, sent.text, self.persona
                    )
                )
                break
            except:
                continue
        else:
            return [], []

        for _ in range(3):
            try:
                new_relationships = (
                    self.client.generate_new_inferred_relationships_first_sentence(
                        nodes_from_text,
                        object_properties_dict["concepts"][: self.max_new_concepts],
                        object_properties_dict["properties"][: self.max_new_properties],
                        sent.text,
                        self.persona,
                    )
                )
                return (
                    new_relationships["concept_relationships"],
                    new_relationships["property_relationships"],
                )
            except:
                continue
        return [], []

    def infer_new_relationships(
        self,
        text: str,
        current_sentence_text_based_nodes: List[Node],
        current_sentence_text_based_words: List[str],
        graph_nodes_representation: str,
        graph_edges_representation: str,
    ) -> Tuple[List[Tuple[str, str, str]], List[Tuple[str, str, str]]]:
        nodes_from_text = ""
        for i, node in enumerate(current_sentence_text_based_nodes):
            nodes_from_text += (
                f" - ({current_sentence_text_based_words[i]}, {node.node_type})\n"
            )

        for _ in range(3):
            try:
                object_properties_dict = self.client.infer_objects_and_properties(
                    nodes_from_text,
                    graph_nodes_representation,
                    graph_edges_representation,
                    text,
                    self.persona,
                )
                break
            except:
                continue

        for _ in range(3):
            try:
                new_relationships = self.client.generate_new_inferred_relationships(
                    nodes_from_text,
                    graph_nodes_representation,
                    graph_edges_representation,
                    object_properties_dict["concepts"][: self.max_new_concepts],
                    object_properties_dict["properties"][: self.max_new_properties],
                    text,
                    self.persona,
                )
                return (
                    new_relationships["concept_relationships"],
                    new_relationships["property_relationships"],
                )
            except:
                continue
        return [], []

    def reactivate_relevant_edges(
        self,
        active_nodes: List[Node],
        prev_sentences_text: str,
        newly_added_edges: List[Edge],
    ) -> None:
        edges_text, edges = self.graph.get_edges_str(
            self.graph.nodes, only_active=False
        )
        raw_indices = self.client.get_relevant_edges(
            edges_text, prev_sentences_text, self.persona
        )

        if self.strict_reactivate_function:
            valid_indices: List[int] = []
            for idx in raw_indices:
                try:
                    i = int(idx)
                except Exception:
                    continue
                if 1 <= i <= len(edges):
                    valid_indices.append(i)

            valid_indices = valid_indices[: self.nr_relevant_edges]
            active_node_set = set(active_nodes)
            if not valid_indices:
                # Fallback (no LLM indices): keep only edges that are already active
                # AND stay within the current active-node neighborhood, plus newly added.
                selected = set()
                for idx, edge in enumerate(edges, start=1):
                    if edge in newly_added_edges:
                        selected.add(idx)
                    elif (
                        edge.active
                        and edge.source_node in active_node_set
                        and edge.dest_node in active_node_set
                    ):
                        selected.add(idx)
            else:
                selected = set(valid_indices)
                for i in selected:
                    edges[i - 1].forget_score = self.edge_forget
                    edges[i - 1].active = True
                    self._record_edge_in_graphs(edges[i - 1], self._current_sentence_index)
        else:
            # Legacy behavior from the original paper code: trust the LLM indices,
            # reactivate them, and fade everything else not newly added.
            legacy_indices: List[int] = []
            for idx in raw_indices[: self.nr_relevant_edges]:
                try:
                    i = int(idx)
                except Exception:
                    continue
                if 1 <= i <= len(edges):
                    legacy_indices.append(i)
            selected = set(legacy_indices)
            for i in selected:
                edges[i - 1].forget_score = self.edge_forget
                edges[i - 1].active = True
                self._record_edge_in_graphs(edges[i - 1], self._current_sentence_index)

        # Fade away edges that were not selected and are not newly added
        for j in range(1, len(edges) + 1):
            if j not in selected and edges[j - 1] not in newly_added_edges:
                edges[j - 1].fade_away()
                self._record_edge_in_graphs(edges[j - 1], self._current_sentence_index)

    def init_graph(self, sent: Span) -> None:
        current_sentence_text_based_nodes, current_sentence_text_based_words = (
            self.get_senteces_text_based_nodes([sent], create_unexistent_nodes=True)
        )

        nodes_from_text = ""
        for i, node in enumerate(current_sentence_text_based_nodes):
            nodes_from_text += (
                f" - ({current_sentence_text_based_words[i]}, {node.node_type})\n"
            )

        relationships = self.client.get_new_relationships_first_sentence(
            nodes_from_text, sent.text, self.persona
        )
        # print(f"First sentence edges:\n{relationships}")

        for relationship in relationships:
            if len(relationship) != 3:
                continue
            if not relationship[0] or not relationship[2]:
                continue
            if relationship[0] == relationship[2]:
                continue
            if not isinstance(relationship[0], str) or not isinstance(
                relationship[2], str
            ):
                continue
            norm_subj = self._normalize_endpoint_text(relationship[0], is_subject=True)
            norm_obj = self._normalize_endpoint_text(relationship[2], is_subject=False)
            if norm_subj is None or norm_obj is None:
                continue
            if not self._passes_attachment_constraint(
                norm_subj,
                norm_obj,
                current_sentence_text_based_words,
                current_sentence_text_based_nodes,
                list(self.graph.nodes),
                self._get_nodes_with_active_edges(),
            ):
                continue
            source_node = self.get_node_from_text(
                norm_subj,
                current_sentence_text_based_nodes,
                current_sentence_text_based_words,
                node_source=NodeSource.TEXT_BASED,
                create_node=True,
            )
            dest_node = self.get_node_from_text(
                norm_obj,
                current_sentence_text_based_nodes,
                current_sentence_text_based_words,
                node_source=NodeSource.TEXT_BASED,
                create_node=True,
            )
            edge_label = relationship[1].replace("(edge)", "").strip()
            if not self._is_valid_relation_label(edge_label):
                continue
            if source_node is None or dest_node is None:
                continue
            self._add_edge(source_node, dest_node, edge_label, self.edge_forget)

    def add_inferred_relationships_to_graph_step_0(
        self,
        inferred_relationships: List[Tuple[str, str, str]],
        node_type: NodeType,
        sent: Span,
    ) -> None:
        current_sentence_text_based_nodes, current_sentence_text_based_words = (
            self.get_senteces_text_based_nodes([sent], create_unexistent_nodes=False)
        )
        for relationship in inferred_relationships:
            # print(relationship)
            if len(relationship) != 3:
                continue
            if not relationship[0] or not relationship[2]:
                continue
            if relationship[0] == relationship[2]:
                continue
            if not isinstance(relationship[0], str) or not isinstance(
                relationship[2], str
            ):
                continue
            norm_subj = self._normalize_endpoint_text(relationship[0], is_subject=True)
            norm_obj = self._normalize_endpoint_text(relationship[2], is_subject=False)
            if norm_subj is None or norm_obj is None:
                continue
            if not (
                self._appears_in_story(relationship[0])
                or self._appears_in_story(relationship[2])
            ):
                continue
            if not self._passes_attachment_constraint(
                relationship[0],
                relationship[2],
                current_sentence_text_based_words,
                current_sentence_text_based_nodes,
                list(self.graph.nodes),
                self._get_nodes_with_active_edges(),
            ):
                continue
            subj, subj_type = self._canonicalize_and_classify_node_text(relationship[0])
            obj, obj_type = self._canonicalize_and_classify_node_text(relationship[2])
            if subj_type is None or obj_type is None:
                continue
            source_node = self.get_node_from_text(
                norm_subj,
                current_sentence_text_based_nodes,
                current_sentence_text_based_words,
                node_source=NodeSource.INFERENCE_BASED,
                create_node=False,
            )
            dest_node = self.get_node_from_text(
                norm_obj,
                current_sentence_text_based_nodes,
                current_sentence_text_based_words,
                node_source=NodeSource.INFERENCE_BASED,
                create_node=False,
            )
            edge_label = relationship[1].replace("(edge)", "").strip()
            if not self._is_valid_relation_label(edge_label):
                continue
            if source_node is None:
                source_node = self.graph.add_or_get_node(
                    get_concept_lemmas(self.spacy_nlp, subj),
                    subj,
                    subj_type,
                    NodeSource.INFERENCE_BASED,
                )

            if dest_node is None:
                dest_node = self.graph.add_or_get_node(
                    get_concept_lemmas(self.spacy_nlp, obj),
                    obj,
                    obj_type,
                    NodeSource.INFERENCE_BASED,
                )

            self._add_edge(source_node, dest_node, edge_label, self.edge_forget)

    def add_inferred_relationships_to_graph(
        self,
        inferred_relationships: List[Tuple[str, str, str]],
        node_type: NodeType,
        curr_sentences_nodes: List[Node],
        curr_sentences_words: List[str],
        active_graph_nodes: List[Node],
        added_edges: List[Edge],
    ) -> None:
        for relationship in inferred_relationships:
            # print(relationship)
            if len(relationship) != 3:
                continue
            if not relationship[0] or not relationship[2]:
                continue
            if relationship[0] == relationship[2]:
                continue
            if not isinstance(relationship[0], str) or not isinstance(
                relationship[2], str
            ):
                continue
            norm_subj = self._normalize_endpoint_text(relationship[0], is_subject=True)
            norm_obj = self._normalize_endpoint_text(relationship[2], is_subject=False)
            if norm_subj is None or norm_obj is None:
                continue
            if not (
                self._appears_in_story(relationship[0])
                or self._appears_in_story(relationship[2])
            ):
                continue
            if not self._passes_attachment_constraint(
                relationship[0],
                relationship[2],
                curr_sentences_words,
                curr_sentences_nodes,
                active_graph_nodes,
                self._get_nodes_with_active_edges(),
            ):
                continue
            subj, subj_type = self._canonicalize_and_classify_node_text(relationship[0])
            obj, obj_type = self._canonicalize_and_classify_node_text(relationship[2])
            if subj_type is None or obj_type is None:
                continue
            source_node = self.get_node_from_new_relationship(
                norm_subj,
                active_graph_nodes,
                curr_sentences_nodes,
                curr_sentences_words,
                node_source=NodeSource.INFERENCE_BASED,
                create_node=False,
            )
            dest_node = self.get_node_from_new_relationship(
                norm_obj,
                active_graph_nodes,
                curr_sentences_nodes,
                curr_sentences_words,
                node_source=NodeSource.INFERENCE_BASED,
                create_node=False,
            )
            edge_label = relationship[1].replace("(edge)", "").strip()
            if not self._is_valid_relation_label(edge_label):
                continue
            if source_node is None:
                source_node = self.graph.add_or_get_node(
                    get_concept_lemmas(self.spacy_nlp, subj),
                    subj,
                    subj_type,
                    NodeSource.INFERENCE_BASED,
                )

            if dest_node is None:
                dest_node = self.graph.add_or_get_node(
                    get_concept_lemmas(self.spacy_nlp, obj),
                    obj,
                    obj_type,
                    NodeSource.INFERENCE_BASED,
                )

            potential_edge = self._add_edge(
                source_node, dest_node, edge_label, self.edge_forget
            )
            if potential_edge:
                added_edges.append(potential_edge)

    def get_node_from_text(
        self,
        text: str,
        curr_sentences_nodes: List[Node],
        curr_sentences_words: List[str],
        node_source: NodeSource,
        create_node: bool,
    ) -> Optional[Node]:
        if text in curr_sentences_words:
            return curr_sentences_nodes[curr_sentences_words.index(text)]
        if create_node:
            canon, inferred_type = self._canonicalize_and_classify_node_text(text)
            if inferred_type is None:
                return None
            lemmas = get_concept_lemmas(self.spacy_nlp, canon)
            return self.graph.add_or_get_node(lemmas, canon, inferred_type, node_source)
        return None

    def get_node_from_new_relationship(
        self,
        text: str,
        graph_active_nodes: List[Node],
        curr_sentences_nodes: List[Node],
        curr_sentences_words: List[str],
        node_source: NodeSource,
        create_node: bool,
    ) -> Optional[Node]:
        if text in curr_sentences_words:
            return curr_sentences_nodes[curr_sentences_words.index(text)]
        else:
            canon, inferred_type = self._canonicalize_and_classify_node_text(text)
            if inferred_type is None:
                return None
            lemmas = get_concept_lemmas(self.spacy_nlp, canon)
            for node in graph_active_nodes:
                if lemmas == node.lemmas:
                    return node
        if create_node:
            canon, inferred_type = self._canonicalize_and_classify_node_text(text)
            if inferred_type is None:
                return None
            lemmas = get_concept_lemmas(self.spacy_nlp, canon)
            return self.graph.add_or_get_node(lemmas, canon, inferred_type, node_source)
        return None

    def is_content_word_and_non_stopword(
        self,
        token: Token,
        pos_list: List[str] = [
            "NOUN",
            "PROPN",
            "ADJ",
        ],
    ) -> bool:
        return (token.pos_ in pos_list) and (
            token.lemma_ not in self.spacy_nlp.Defaults.stop_words
        )

    def get_senteces_text_based_nodes(
        self, previous_sentences: List[Span], create_unexistent_nodes: bool = True
    ) -> Tuple[List[Node], List[str]]:
        text_based_nodes = []
        text_based_words = []
        for sent in previous_sentences:
            content_words = get_content_words_from_sent(self.spacy_nlp, sent)
            for word in content_words:
                node = self.graph.get_node([word.lemma_])
                if node is not None:
                    node.add_actual_text(word.text)
                    text_based_nodes.append(node)
                    text_based_words.append(word.text)
                else:
                    if create_unexistent_nodes:
                        if word.pos_ == "ADJ":
                            new_node = self.graph.add_or_get_node(
                                [word.lemma_],
                                word.text,
                                NodeType.PROPERTY,
                                NodeSource.TEXT_BASED,
                            )
                        else:
                            new_node = self.graph.add_or_get_node(
                                [word.lemma_],
                                word.text,
                                NodeType.CONCEPT,
                                NodeSource.TEXT_BASED,
                            )
                        text_based_nodes.append(new_node)
                        text_based_words.append(word.text)
        return text_based_nodes, text_based_words
