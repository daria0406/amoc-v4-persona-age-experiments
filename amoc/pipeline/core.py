import logging
import re
from typing import List, Tuple, Optional, Iterable

from spacy.tokens import Span, Token

from amoc.graph import Graph, Node, Edge, NodeType, NodeSource
from amoc.viz.graph_plots import plot_amoc_triplets
from amoc.llm.vllm_client import VLLMClient
from amoc.nlp.spacy_utils import (
    get_concept_lemmas,
    has_noun,
    get_content_words_from_sent,
)


class AMoCv4:
    GENERIC_RELATION_LABELS = {
        "has_property",
        "is_type_of",
        "is_part_of",
        "related_to",
        "has_attribute",
        "is_a",
    }

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

    def reset_graph(self) -> None:
        self.graph = Graph()

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
            triplets.append(
                (
                    edge.source_node.get_text_representer(),
                    edge.label,
                    edge.dest_node.get_text_representer(),
                )
            )
        return triplets

    def _normalize_label(self, label: str) -> str:
        norm = (label or "").strip().lower()
        norm = re.sub(r"[\s\-]+", "_", norm)
        return norm

    def _is_generic_relation(self, label: str) -> bool:
        norm = self._normalize_label(label)
        return norm in self.GENERIC_RELATION_LABELS

    def _appears_in_story(self, text: str) -> bool:
        if not text:
            return False
        doc = self.spacy_nlp(text)
        return any(
            tok.lemma_.lower() in self.story_lemmas for tok in doc if tok.is_alpha
        )

    def _plot_graph_snapshot(
        self,
        sentence_index: int,
        output_dir: Optional[str],
        highlight_nodes: Optional[Iterable[str]],
        only_active: bool = True,
    ) -> None:
        triplets = self._graph_edges_to_triplets(only_active=only_active)
        age_for_filename = self.persona_age if self.persona_age is not None else -1
        try:
            saved_path = plot_amoc_triplets(
                triplets=triplets,
                persona=self.persona,
                model_name=self.model_name,
                age=age_for_filename,
                blue_nodes=highlight_nodes,
                output_dir=output_dir,
                step_tag=f"sent{sentence_index}",
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
        replace_pronouns: bool = False,
        plot_after_each_sentence: bool = False,
        graphs_output_dir: Optional[str] = None,
        highlight_nodes: Optional[Iterable[str]] = None,
    ) -> List[Tuple[str, str, str]]:
        text = self.story_text
        if replace_pronouns:
            text = self.resolve_pronouns(text)
        doc = self.spacy_nlp(text)
        prev_sentences = []
        current_sentence = ""
        for i, sent in enumerate(doc.sents):
            logging.info("Processing sentence %d: %s" % (i, sent))
            if i == 0:
                current_sentence = sent
                prev_sentences.append(sent)
                self.init_graph(sent)

                inferred_concept_relationships, inferred_property_relationships = (
                    self.infer_new_relationships_step_0(sent)
                )

                self.add_inferred_relationships_to_graph_step_0(
                    inferred_concept_relationships, NodeType.CONCEPT, sent
                )
                self.add_inferred_relationships_to_graph_step_0(
                    inferred_property_relationships, NodeType.PROPERTY, sent
                )
            else:
                added_edges = []
                current_sentence = sent
                prev_sentences.append(sent)
                if len(prev_sentences) > self.context_length:
                    prev_sentences.pop(0)

                current_sentence_text_based_nodes, current_sentence_text_based_words = (
                    self.get_senteces_text_based_nodes(
                        [current_sentence], create_unexistent_nodes=True
                    )
                )

                current_all_text = sent.text
                graph_active_nodes = self.graph.get_active_nodes(
                    self.max_distance_from_active_nodes
                )
                # Restrict prompt context to text-based active nodes/edges
                text_active_nodes = self.graph.get_active_nodes(
                    self.max_distance_from_active_nodes, only_text_based=True
                )
                active_nodes_text = self.graph.get_nodes_str(text_active_nodes)
                active_nodes_edges_text, _ = self.graph.get_edges_str(
                    text_active_nodes, only_text_based=True
                )

                nodes_from_text = ""
                for idx, node in enumerate(current_sentence_text_based_nodes):
                    nodes_from_text += f" - ({current_sentence_text_based_words[idx]}, {node.node_type})\n"

                # Fetch new relationships (CORRECTED ARGUMENT ORDER)
                # Signature: (nodes_from_text, nodes_from_graph, edges_from_graph, text, persona)
                # Use full active graph context (text + inferred) to allow richer connections.
                full_active_nodes = self.graph.get_active_nodes(
                    self.max_distance_from_active_nodes, only_text_based=False
                )
                full_active_nodes_text = self.graph.get_nodes_str(full_active_nodes)
                full_active_edges_text, _ = self.graph.get_edges_str(
                    full_active_nodes, only_text_based=False
                )

                new_relationships = self.client.get_new_relationships(
                    nodes_from_text,  # 1. Nodes from Text
                    full_active_nodes_text,  # 2. Nodes from Graph
                    full_active_edges_text,  # 3. Edges from Graph
                    current_all_text,  # 4. Text
                    self.persona,  # 5. Persona
                )

                text_based_activated_nodes = current_sentence_text_based_nodes
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

                    # Validate subject/object strings
                    if not subj or not obj:
                        continue
                    if subj == obj:
                        continue
                    if not isinstance(subj, str) or not isinstance(obj, str):
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
                        relationship[2],
                        graph_active_nodes,
                        current_sentence_text_based_nodes,
                        current_sentence_text_based_words,
                        node_source=NodeSource.TEXT_BASED,
                        create_node=True,
                    )
                    edge_label = relationship[1].replace("(edge)", "").strip()
                    if source_node is None or dest_node is None:
                        continue

                    if relationship[0] in current_sentence_text_based_words:
                        source_node.node_source = NodeSource.TEXT_BASED
                    if relationship[2] in current_sentence_text_based_words:
                        dest_node.node_source = NodeSource.TEXT_BASED

                    potential_new_edge = self.graph.add_edge(
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

                self.graph.set_nodes_score_based_on_distance_from_active_nodes(
                    text_based_activated_nodes
                )
                self.reactivate_relevant_edges(
                    self.graph.get_active_nodes(self.max_distance_from_active_nodes),
                    " ".join([s.text for s in prev_sentences]),
                    added_edges,
                )
                self.graph.set_nodes_score_based_on_distance_from_active_nodes(
                    text_based_activated_nodes
                )

            if plot_after_each_sentence:
                self._plot_graph_snapshot(
                    sentence_index=i,
                    output_dir=graphs_output_dir,
                    highlight_nodes=highlight_nodes,
                    only_active=True,
                )

        # Return triplets for external saving
        return [
            (
                edge.source_node.get_text_representer(),
                edge.label,
                edge.dest_node.get_text_representer(),
                edge.active,
            )
            for edge in self.graph.edges
        ]

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
        relevant_edges_index = self.client.get_relevant_edges(
            edges_text, prev_sentences_text, self.persona
        )[: self.nr_relevant_edges]
        for i in relevant_edges_index:
            # print("Reactivating edge: ", edges[i-1]) # Reduced verbosity
            edges[i - 1].forget_score = self.edge_forget
            edges[i - 1].active = True

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
            source_node = self.get_node_from_text(
                relationship[0],
                current_sentence_text_based_nodes,
                current_sentence_text_based_words,
                node_source=NodeSource.TEXT_BASED,
                create_node=True,
            )
            dest_node = self.get_node_from_text(
                relationship[2],
                current_sentence_text_based_nodes,
                current_sentence_text_based_words,
                node_source=NodeSource.TEXT_BASED,
                create_node=True,
            )
            edge_label = relationship[1].replace("(edge)", "").strip()
            if self._is_generic_relation(edge_label):
                continue
            if source_node is None or dest_node is None:
                continue
            self.graph.add_edge(source_node, dest_node, edge_label, self.edge_forget)

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
            if not (
                self._appears_in_story(relationship[0])
                or self._appears_in_story(relationship[2])
            ):
                continue
            source_node = self.get_node_from_text(
                relationship[0],
                current_sentence_text_based_nodes,
                current_sentence_text_based_words,
                node_source=NodeSource.INFERENCE_BASED,
                create_node=False,
            )
            dest_node = self.get_node_from_text(
                relationship[2],
                current_sentence_text_based_nodes,
                current_sentence_text_based_words,
                node_source=NodeSource.INFERENCE_BASED,
                create_node=False,
            )
            edge_label = relationship[1].replace("(edge)", "").strip()
            if self._is_generic_relation(edge_label):
                continue
            if source_node is None:
                source_node = self.graph.add_or_get_node(
                    get_concept_lemmas(self.spacy_nlp, relationship[0]),
                    relationship[0],
                    node_type,
                    NodeSource.INFERENCE_BASED,
                )

            if dest_node is None:
                dest_node = self.graph.add_or_get_node(
                    get_concept_lemmas(self.spacy_nlp, relationship[2]),
                    relationship[2],
                    node_type,
                    NodeSource.INFERENCE_BASED,
                )

            self.graph.add_edge(source_node, dest_node, edge_label, self.edge_forget)

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
            if not (
                self._appears_in_story(relationship[0])
                or self._appears_in_story(relationship[2])
            ):
                continue
            source_node = self.get_node_from_new_relationship(
                relationship[0],
                active_graph_nodes,
                curr_sentences_nodes,
                curr_sentences_words,
                node_source=NodeSource.INFERENCE_BASED,
                create_node=False,
            )
            dest_node = self.get_node_from_new_relationship(
                relationship[2],
                active_graph_nodes,
                curr_sentences_nodes,
                curr_sentences_words,
                node_source=NodeSource.INFERENCE_BASED,
                create_node=False,
            )
            edge_label = relationship[1].replace("(edge)", "").strip()
            if self._is_generic_relation(edge_label):
                continue
            if source_node is None:
                source_node = self.graph.add_or_get_node(
                    get_concept_lemmas(self.spacy_nlp, relationship[0]),
                    relationship[0],
                    node_type,
                    NodeSource.INFERENCE_BASED,
                )

            if dest_node is None:
                dest_node = self.graph.add_or_get_node(
                    get_concept_lemmas(self.spacy_nlp, relationship[2]),
                    relationship[2],
                    node_type,
                    NodeSource.INFERENCE_BASED,
                )

            potential_edge = self.graph.add_edge(
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
            lemmas = get_concept_lemmas(self.spacy_nlp, text)
            if has_noun(self.spacy_nlp, text):
                new_node = self.graph.add_or_get_node(
                    lemmas, text, NodeType.CONCEPT, node_source
                )
            else:
                new_node = self.graph.add_or_get_node(
                    lemmas, text, NodeType.PROPERTY, node_source
                )
            return new_node
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
            lemmas = get_concept_lemmas(self.spacy_nlp, text)
            for node in graph_active_nodes:
                if lemmas == node.lemmas:
                    return node
        if create_node:
            lemmas = get_concept_lemmas(self.spacy_nlp, text)
            if has_noun(self.spacy_nlp, text):
                new_node = self.graph.add_or_get_node(
                    lemmas, text, NodeType.CONCEPT, node_source
                )
            else:
                new_node = self.graph.add_or_get_node(
                    lemmas, text, NodeType.PROPERTY, node_source
                )
            return new_node
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
