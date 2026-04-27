from __future__ import annotations
from typing import TYPE_CHECKING, Set, List, Dict, Optional, Tuple, FrozenSet, Callable
from collections import deque
import networkx as nx
import logging

if TYPE_CHECKING:
    from amoc.core.node import Node
    from amoc.core.edge import Edge
    from amoc.core.graph import Graph

from amoc.core.node import NodeType
from amoc.config.constants import MAX_CARRYOVER_NODES, CARRYOVER_SENIORITY_WEIGHT


class PerSentenceGraph:
    def __init__(
        self,
        sentence_index: int,
        explicit_nodes: FrozenSet[Node],
        carryover_nodes: FrozenSet[Node],
        active_nodes: FrozenSet[Node],
        active_edges: FrozenSet[Edge],
        anchor_nodes: FrozenSet[Node],
    ) -> None:
        self.sentence_index = sentence_index
        self.explicit_nodes = explicit_nodes
        self.carryover_nodes = carryover_nodes
        self.active_nodes = active_nodes
        self.active_edges = active_edges
        self.anchor_nodes = frozenset()

        adjacency = {n: set() for n in self.active_nodes}
        for edge in self.active_edges:
            if edge.source_node in adjacency and edge.dest_node in adjacency:
                adjacency[edge.source_node].add(edge.dest_node)
                adjacency[edge.dest_node].add(edge.source_node)
        self._adjacency: Dict[Node, Set[Node]] = adjacency

        # Compute degrees
        degrees = {n: len(neighbors) for n, neighbors in adjacency.items()}
        self._node_degrees: Dict[Node, int] = degrees

        # Build NetworkX graph for connectivity
        G = nx.Graph()
        for node in self.active_nodes:
            G.add_node(node)
        for edge in self.active_edges:
            G.add_edge(edge.source_node, edge.dest_node, edge=edge)
        self._nx_graph: Optional[nx.Graph] = G

    def is_connected(self) -> bool:
        if self._nx_graph.number_of_nodes() <= 1:
            return True
        return nx.is_connected(self._nx_graph)

    def is_empty(self) -> bool:
        return len(self.active_edges) == 0

    def get_node_degree(self, node: Node) -> int:
        return self._node_degrees.get(node, 0)

    def get_neighbors(self, node: Node) -> Set[Node]:
        return self._adjacency.get(node, set())

    def node_is_visible(self, node: Node) -> bool:
        return node in self.active_nodes

    def edge_is_visible(self, edge: Edge) -> bool:
        return edge in self.active_edges

    def get_triplets(self) -> List[Tuple[str, str, str]]:
        return [
            (
                edge.source_node.get_text_representer(),
                edge.label,
                edge.dest_node.get_text_representer(),
            )
            for edge in self.active_edges
        ]

    def to_networkx(self) -> nx.Graph:
        return self._nx_graph.copy()


class PerSentenceGraphBuilder:
    def __init__(
        self,
        cumulative_graph: Graph,
        max_distance: int,
        anchor_nodes: Set[Node],
        sentence_index: int,         
        repair_callback=None,
    ):
        self.cumulative_graph = cumulative_graph
        self.max_distance = max_distance
        self.anchor_nodes = frozenset()
        self.repair_callback = repair_callback

        self._explicit_nodes: Set[Node] = set()
        self._carryover_nodes: Set[Node] = set()
        self._sentence_index: int = sentence_index   
        self._distances: Optional[Dict[Node, int]] = None

    def set_explicit_nodes(self, nodes: List[Node]) -> "PerSentenceGraphBuilder":
        self._explicit_nodes = set(nodes)
        return self

    def compute_carryover_nodes(self) -> "PerSentenceGraphBuilder":
        if not self._explicit_nodes:
            self._carryover_nodes = set()
            self._distances = {}
            return self

        # BFS to find all reachable nodes within distance
        distances = {n: 0 for n in self._explicit_nodes}
        queue = deque(self._explicit_nodes)

        while queue:
            node = queue.popleft()
            current_dist = distances[node]

            if current_dist >= self.max_distance:
                continue

            for edge in node.edges:
                if not (edge.active and edge.visibility_score > 0):
                    continue

                neighbor = (
                    edge.dest_node if edge.source_node == node else edge.source_node
                )

                if neighbor in distances:
                    continue

                distances[neighbor] = current_dist + 1
                queue.append(neighbor)

        # Store distances for later capping
        self._distances = distances

        # All reachable nodes - current explicit nodes
        reachable_nodes = set(distances.keys()) - self._explicit_nodes

        # Carryover = all reachable nodes with at least one active edge
        self._carryover_nodes = {
            node
            for node in reachable_nodes
            if any(e.active and e.visibility_score > 0 for e in node.edges)
        }

        # Log nodes that were reachable but excluded (no active edges)
        excluded = reachable_nodes - self._carryover_nodes
        for node in excluded:
            logging.info(
                f"excluded from carryover: {node.get_text_representer()} (reachable but no active edges)"
            )

        # cap carryover nodes based on activation score and seniority
        self.cap_carryover_nodes()

        return self

    # cap carryover nodes to MAX_CARRYOVER_NODES, using activation score and seniority
    def cap_carryover_nodes(self) -> None:
        if len(self._carryover_nodes) <= MAX_CARRYOVER_NODES:
            logging.info(f"Sentence {self._sentence_index}: carryover size {len(self._carryover_nodes)} ≤ {MAX_CARRYOVER_NODES}, no capping needed")
            return
            
        # Compute priority for each carryover node
        scored = []
        for node in self._carryover_nodes:
            # activation from BFS distance: 5 - distance
            distance = self._distances.get(node, self.max_distance + 1)
            if distance > self.max_distance:
                activation = 0.0
            else:
                activation = 5.0 - float(distance)

            # seniority: how many sentences since node first appeared
            seniority = 0
            if hasattr(node, 'first_seen_sentence') and node.first_seen_sentence is not None:
                seniority = self._sentence_index - node.first_seen_sentence
                if seniority < 0:
                    seniority = 0

            score = activation - CARRYOVER_SENIORITY_WEIGHT * seniority
            scored.append((score, node))

        # Sort descending by score, keep top MAX_CARRYOVER_NODES
        scored.sort(key=lambda x: x[0], reverse=True)
        best_nodes = {node for _, node in scored[:MAX_CARRYOVER_NODES]}

        # Log dropped nodes for debugging
        dropped = self._carryover_nodes - best_nodes
        if dropped:
            logging.debug(
                f"Sentence {self._sentence_index}: dropped {len(dropped)} old carryover nodes: "
                f"{[n.get_text_representer() for n in dropped]}"
            )

        self._carryover_nodes = best_nodes

    def get_active_nodes(self) -> Set[Node]:
        return self._explicit_nodes | self._carryover_nodes

    def get_attachable_nodes(self) -> Set[Node]:
        return self._explicit_nodes | self._carryover_nodes

    def can_add_edge(self, source: Node, dest: Node) -> bool:
        attachable = self.get_attachable_nodes()
        return source in attachable or dest in attachable

    def build(self, sentence_index: int) -> PerSentenceGraph:
        current_index = self._sentence_index

        global_active_nodes, global_active_edges = (
            self.cumulative_graph.get_active_subgraph_wrapper()
        )
        global_active_nodes = set(global_active_nodes)
        global_active_edges = set(global_active_edges)

        explicit_nodes = set(self._explicit_nodes)
        carryover_nodes = set(self._carryover_nodes)
        view_nodes = explicit_nodes | carryover_nodes

        view_edges = {
            e
            for e in global_active_edges
            if e.source_node in view_nodes and e.dest_node in view_nodes
        }

        return PerSentenceGraph(
            sentence_index=current_index,
            explicit_nodes=frozenset(explicit_nodes),
            carryover_nodes=frozenset(carryover_nodes),
            active_nodes=frozenset(view_nodes),
            active_edges=frozenset(view_edges),
            anchor_nodes=frozenset(),
        )


def build_per_sentence_graph(
    cumulative_graph: Graph,
    explicit_nodes: List[Node],
    max_distance: int,
    anchor_nodes: Set[Node],
    sentence_index: int,
    repair_callback: Optional[
        Callable[
            [List[Set[Node]], Set[Node], Set[Edge], int],
            Optional[Set[Edge]],
        ]
    ] = None,
) -> PerSentenceGraph:

    builder = PerSentenceGraphBuilder(
        cumulative_graph=cumulative_graph,
        max_distance=max_distance,
        anchor_nodes=anchor_nodes,
        sentence_index=sentence_index,   
        repair_callback=repair_callback,
    )

    return (
        builder.set_explicit_nodes(explicit_nodes)
        .compute_carryover_nodes()
        .build(sentence_index)
    )