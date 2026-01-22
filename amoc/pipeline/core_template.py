"""
AMoCv4 Core Template - Per-Sentence Graph Construction

This template provides the structure for rebuilding core.py with proper
per-sentence graph constraints. Fill in each method following the docstrings.

NODE STATES (per sentence):
- EXPLICIT: Nodes from current sentence text (score=0, always visible)
- CARRY-OVER: Nodes reachable from explicit via active edges within max_distance
- INACTIVE: Nodes with score > max_distance (must be excluded from per-sentence graph)

KEY INVARIANTS:
1. Per-sentence graph contains ONLY explicit + carry-over nodes
2. Edges in per-sentence graph must have BOTH endpoints active
3. Per-sentence graph must always be connected (when non-empty)
4. Inactive nodes must not appear in nodes, edges, metrics, or metadata
"""

import logging
import os
import re
from typing import List, Tuple, Optional, Iterable
import pandas as pd
from spacy.tokens import Span, Token
import networkx as nx

from amoc.graph import Graph, Node, Edge, NodeType, NodeSource
from amoc.graph.per_sentence_graph import (
    PerSentenceGraph,
    PerSentenceGraphBuilder,
    build_per_sentence_graph,
)
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
    """Sanitize a string for use in filenames."""
    pass  # TODO: implement


class AMoCv4:
    """
    Associative Memory of Comprehension v4.

    Builds per-sentence knowledge graphs from narrative text, tracking
    node activation states and edge salience across sentences.
    """

    # Relations to filter out (too generic)
    GENERIC_RELATION_LABELS = {
        "has_property", "is_type_of", "is_part_of", "related_to",
        "has_attribute", "is_a", "appears", "contains", "includes",
        # ... add more as needed
    }

    ENFORCE_ATTACHMENT_CONSTRAINT = False  # Class-level flag for targeted inference
    ACTIVATION_MAX_DISTANCE = 2  # Max BFS distance for activation matrix
    RELATION_BLACKLIST = {"describes", "is_at_stake"}

    # =========================================================================
    # INITIALIZATION
    # =========================================================================

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
        matrix_dir_base: Optional[str] = None,
    ) -> None:
        """
        Initialize AMoCv4.

        Key instance variables to set up:
        - self.graph: Graph() - the cumulative knowledge graph
        - self._anchor_nodes: set[Node] - nodes that guarantee connectivity
        - self._explicit_nodes_current_sentence: set[Node] - current sentence's nodes
        - self._per_sentence_view: Optional[PerSentenceGraph] - current sentence's view
        - self.strict_attachament_constraint: bool - controls strict mode features
        """
        pass  # TODO: implement

    # =========================================================================
    # PER-SENTENCE GRAPH CONSTRUCTION (strict mode only)
    # =========================================================================

    def _build_per_sentence_view(
        self, explicit_nodes: List[Node], sentence_index: int
    ) -> Optional[PerSentenceGraph]:
        """
        Build the per-sentence graph view with all invariants enforced.

        ONLY active when strict_attachament_constraint is True.
        When False, returns None and legacy behavior is used.

        The view guarantees:
        1. Only explicit + carry-over nodes are visible
        2. Inactive nodes are completely excluded
        3. Only edges where BOTH endpoints are visible are included
        4. The graph is connected (or empty)
        """
        pass  # TODO: implement

    def _get_attachable_nodes_for_sentence(self) -> set[Node]:
        """
        Get the set of nodes that new edges can attach to.

        Returns: explicit | carry-over | anchor nodes

        This is the core connectivity guarantee - new edges must have
        at least one endpoint in this set.
        """
        pass  # TODO: implement

    # =========================================================================
    # ATTACHMENT CONSTRAINT (controls what edges can be added)
    # =========================================================================

    def _passes_attachment_constraint(
        self,
        subject: str,
        obj: str,
        current_sentence_words: List[str],
        current_sentence_nodes: List[Node],
        graph_active_nodes: List[Node],
        graph_active_edge_nodes: Optional[set[Node]] = None,
    ) -> bool:
        """
        Check if an edge (subject, *, obj) passes the attachment constraint.

        STRICT MODE (strict_attachament_constraint=True):
        - At least one endpoint must be in the attachable set
        - Attachable = explicit | carry-over | active_edge_nodes | anchors

        PERMISSIVE MODE (strict_attachament_constraint=False):
        - Only requires at least one endpoint exists in memory

        Returns True if the edge is allowed.
        """
        pass  # TODO: implement

    def _add_edge(
        self,
        source_node: Node,
        dest_node: Node,
        label: str,
        edge_forget: int,
        created_at_sentence: Optional[int] = None,
    ) -> Optional[Edge]:
        """
        Add an edge to the graph.

        STRICT MODE: Checks that at least one endpoint is attachable
        (in main component, explicit nodes, or anchors).

        PERMISSIVE MODE: No connectivity check.

        Returns the Edge if added, None if rejected.
        """
        pass  # TODO: implement

    # =========================================================================
    # CONNECTIVITY ENFORCEMENT
    # =========================================================================

    def _enforce_graph_connectivity(self) -> None:
        """
        Remove edges that create disconnected components.

        Finds the main component (containing anchor nodes) and removes
        any edges not in that component.

        Called after first sentence and in reactivation.
        """
        pass  # TODO: implement

    def _get_nodes_with_active_edges(self) -> set[Node]:
        """
        Get all nodes that are endpoints of active edges.

        Also includes explicit nodes from current sentence.
        """
        pass  # TODO: implement

    def _can_attach(self, node: Node) -> bool:
        """Check if a node is in the attachable set."""
        pass  # TODO: implement

    # =========================================================================
    # NODE SCORE MANAGEMENT
    # =========================================================================

    def _restrict_active_to_current_explicit(self, explicit_nodes: List[Node]) -> None:
        """
        Reset node scores: explicit nodes get 0, all others get inactive score.

        This is Step 5 from the paper - only explicit nodes stay active.
        """
        pass  # TODO: implement

    def _distances_from_sources_active_edges(
        self, sources: set[Node], max_distance: int
    ) -> dict[Node, int]:
        """
        BFS from source nodes, following only ACTIVE edges.

        Returns dict mapping each reachable node to its distance.
        """
        pass  # TODO: implement

    # =========================================================================
    # EDGE REACTIVATION
    # =========================================================================

    def reactivate_relevant_edges(
        self,
        active_nodes: List[Node],
        prev_sentences_text: str,
        newly_added_edges: List[Edge],
    ) -> None:
        """
        Reactivate edges based on LLM relevance judgment.

        NON-STRICT MODE (strict_reactivate_function=False):
        - All edges become active (monotonic accumulation)

        STRICT MODE:
        - Query LLM for relevant edges
        - Selected edges get reactivated
        - Non-selected become inactive but stay in memory
        - Bridge edges kept active for connectivity
        """
        pass  # TODO: implement

    # =========================================================================
    # MAIN ANALYSIS LOOP
    # =========================================================================

    def analyze(
        self,
        replace_pronouns: bool = True,
        plot_after_each_sentence: bool = False,
        graphs_output_dir: Optional[str] = None,
        highlight_nodes: Optional[Iterable[str]] = None,
        matrix_suffix: Optional[str] = None,
        largest_component_only: bool = False,
    ) -> List[Tuple[str, str, str]]:
        """
        Main analysis loop - process story sentence by sentence.

        For each sentence:
        1. Extract explicit nodes from sentence text
        2. Get active nodes from previous state (carry-over)
        3. Query LLM for new relationships
        4. Add edges (with attachment constraint check)
        5. Infer new concepts/properties
        6. Update node scores
        7. Reactivate relevant edges
        8. Build per-sentence view (if strict mode)
        9. Record metrics and plot

        Returns: (final_triplets, sentence_triplets, cumulative_triplets)
        """
        pass  # TODO: implement

    # =========================================================================
    # FIRST SENTENCE INITIALIZATION
    # =========================================================================

    def init_graph(self, sent: Span) -> None:
        """
        Initialize graph from first sentence.

        1. Extract text-based nodes
        2. Query LLM for relationships
        3. Add edges (with constraint check)
        """
        pass  # TODO: implement

    def infer_new_relationships_step_0(
        self, sent: Span
    ) -> Tuple[List[Tuple[str, str, str]], List[Tuple[str, str, str]]]:
        """
        Infer new concepts and properties for first sentence.

        Returns: (concept_relationships, property_relationships)
        """
        pass  # TODO: implement

    def add_inferred_relationships_to_graph_step_0(
        self,
        inferred_relationships: List[Tuple[str, str, str]],
        node_type: NodeType,
        sent: Span,
    ) -> None:
        """Add inferred relationships from first sentence to graph."""
        pass  # TODO: implement

    # =========================================================================
    # SUBSEQUENT SENTENCE PROCESSING
    # =========================================================================

    def infer_new_relationships(
        self,
        text: str,
        current_sentence_text_based_nodes: List[Node],
        current_sentence_text_based_words: List[str],
        graph_nodes_representation: str,
        graph_edges_representation: str,
    ) -> Tuple[List[Tuple[str, str, str]], List[Tuple[str, str, str]]]:
        """
        Infer new concepts and properties for subsequent sentences.

        Uses both sentence nodes and active graph context.
        """
        pass  # TODO: implement

    def add_inferred_relationships_to_graph(
        self,
        inferred_relationships: List[Tuple[str, str, str]],
        node_type: NodeType,
        curr_sentences_nodes: List[Node],
        curr_sentences_words: List[str],
        active_graph_nodes: List[Node],
        added_edges: List[Edge],
    ) -> None:
        """Add inferred relationships to graph (subsequent sentences)."""
        pass  # TODO: implement

    # =========================================================================
    # NODE EXTRACTION AND LOOKUP
    # =========================================================================

    def get_senteces_text_based_nodes(
        self, previous_sentences: List[Span], create_unexistent_nodes: bool = True
    ) -> Tuple[List[Node], List[str]]:
        """
        Extract text-based nodes from sentences.

        Returns: (nodes, words) where words[i] corresponds to nodes[i]
        """
        pass  # TODO: implement

    def get_node_from_text(
        self,
        text: str,
        curr_sentences_nodes: List[Node],
        curr_sentences_words: List[str],
        node_source: NodeSource,
        create_node: bool,
    ) -> Optional[Node]:
        """Look up or create a node from text."""
        pass  # TODO: implement

    def get_node_from_new_relationship(
        self,
        text: str,
        graph_active_nodes: List[Node],
        curr_sentences_nodes: List[Node],
        curr_sentences_words: List[str],
        node_source: NodeSource,
        create_node: bool,
    ) -> Optional[Node]:
        """Look up node in sentence nodes first, then active graph nodes."""
        pass  # TODO: implement

    def _find_node_by_text(
        self, text: str, candidates: Iterable[Node]
    ) -> Optional[Node]:
        """Find a node by text among candidates using lemma matching."""
        pass  # TODO: implement

    # =========================================================================
    # TEXT NORMALIZATION AND VALIDATION
    # =========================================================================

    def _normalize_endpoint_text(self, text: str, is_subject: bool) -> Optional[str]:
        """
        Normalize edge endpoint text.

        Extracts the first valid lemma based on POS tag.
        Subject: NOUN, PROPN, PRON
        Object: NOUN, PROPN, PRON, ADJ
        """
        pass  # TODO: implement

    def _normalize_label(self, label: str) -> str:
        """Normalize relation label (lowercase, underscores)."""
        pass  # TODO: implement

    def _canonicalize_and_classify_node_text(
        self, text: str
    ) -> tuple[str, Optional[NodeType]]:
        """Canonicalize text and determine node type (CONCEPT or PROPERTY)."""
        pass  # TODO: implement

    def _classify_canonical_node_text(self, canon: str) -> Optional[NodeType]:
        """Classify canonical text as CONCEPT or PROPERTY based on POS."""
        pass  # TODO: implement

    # =========================================================================
    # RELATION VALIDATION
    # =========================================================================

    def _is_valid_relation_label(self, label: str) -> bool:
        """
        Check if relation label is valid.

        Must: not be generic, not be blacklisted, contain a verb.
        """
        pass  # TODO: implement

    def _is_generic_relation(self, label: str) -> bool:
        """Check if label is in GENERIC_RELATION_LABELS."""
        pass  # TODO: implement

    def _is_blacklisted_relation(self, label: str) -> bool:
        """Check if label is in RELATION_BLACKLIST."""
        pass  # TODO: implement

    def _is_verb_relation(self, label: str) -> bool:
        """Check if label contains a verb (and no nouns/adjs)."""
        pass  # TODO: implement

    # =========================================================================
    # EDGE AND GRAPH UTILITIES
    # =========================================================================

    def _has_edge_between(self, a: Node, b: Node) -> bool:
        """Check if an edge exists between two nodes (either direction)."""
        pass  # TODO: implement

    def _edge_key(self, edge: Edge) -> tuple[str, str, str]:
        """Get (source_text, dest_text, label) key for an edge."""
        pass  # TODO: implement

    def _graph_edges_to_triplets(
        self, only_active: bool = False
    ) -> List[Tuple[str, str, str]]:
        """Convert graph edges to (subject, relation, object) triplets."""
        pass  # TODO: implement

    def _graph_to_triplets(self, graph: nx.MultiDiGraph) -> List[Tuple[str, str, str]]:
        """Convert NetworkX graph to triplets."""
        pass  # TODO: implement

    def _cumulative_triplets_upto(
        self, sentence_idx: Optional[int] = None
    ) -> List[Tuple[str, str, str]]:
        """Get all cumulative triplets up to sentence index."""
        pass  # TODO: implement

    def _appears_in_story(self, text: str) -> bool:
        """Check if any lemma in text appears in the story."""
        pass  # TODO: implement

    # =========================================================================
    # RECORDING AND METRICS
    # =========================================================================

    def _record_edge_in_graphs(self, edge: Edge, sentence_idx: Optional[int]) -> None:
        """
        Record edge in active_graph and cumulative records.

        Updates:
        - self._triplet_intro (introduction sentence)
        - self._cumulative_triplet_records (state log)
        - self.active_graph (add/remove based on edge.active)
        """
        pass  # TODO: implement

    def _record_sentence_activation(
        self,
        sentence_id: int,
        explicit_nodes: List[Node],
        newly_inferred_nodes: set[Node],
    ) -> None:
        """
        Record activation matrix for this sentence.

        Computes landscape scores (5=most active, 0=inactive) and
        records to self._amoc_matrix_records.
        """
        pass  # TODO: implement

    # =========================================================================
    # VISUALIZATION
    # =========================================================================

    def _plot_graph_snapshot(
        self,
        sentence_index: int,
        sentence_text: str,
        output_dir: Optional[str],
        highlight_nodes: Optional[Iterable[str]],
        only_active: bool = False,
        largest_component_only: bool = False,
        mode: str = "sentence_active",
        triplets_override: Optional[List[Tuple[str, str, str]]] = None,
        active_edges: Optional[set[tuple[str, str]]] = None,
        explicit_nodes: Optional[List[str]] = None,
        salient_nodes: Optional[List[str]] = None,
        inactive_nodes: Optional[List[str]] = None,
    ) -> None:
        """Save a graph visualization snapshot."""
        pass  # TODO: implement

    # =========================================================================
    # PRONOUN RESOLUTION
    # =========================================================================

    def resolve_pronouns(self, text: str) -> str:
        """Resolve pronouns in text using LLM."""
        pass  # TODO: implement

    # =========================================================================
    # TARGETED INFERENCE (optional feature)
    # =========================================================================

    def _infer_edges_to_recently_deactivated(
        self,
        current_sentence_nodes: List[Node],
        current_sentence_words: List[str],
        current_text: str,
    ) -> List[Edge]:
        """
        Infer edges between current sentence nodes and recently deactivated nodes.

        Only active when ENFORCE_ATTACHMENT_CONSTRAINT is True.
        """
        pass  # TODO: implement

    # =========================================================================
    # MISC UTILITIES
    # =========================================================================

    def reset_graph(self) -> None:
        """Reset the graph and anchor nodes."""
        pass  # TODO: implement

    def _node_token_for_matrix(self, node: Node) -> str:
        """Get the token string for a node (for activation matrix)."""
        pass  # TODO: implement

    def is_content_word_and_non_stopword(
        self,
        token: Token,
        pos_list: List[str] = ["NOUN", "PROPN", "ADJ"],
    ) -> bool:
        """Check if token is a content word and not a stopword."""
        pass  # TODO: implement
