import os
import re
import math
from typing import List, Tuple, Iterable, Optional, Dict
import networkx as nx
import matplotlib.pyplot as plt
from amoc.config.paths import OUTPUT_ANALYSIS_DIR  # reuse existing output base

DEFAULT_BLUE_NODES: Iterable[str] = ()


def plot_amoc_triplets(
    triplets: List[Tuple[str, str, str]],
    persona: str,
    model_name: str,
    age: int,
    blue_nodes: Optional[Iterable[str]] = None,
    output_dir: str | None = None,
    step_tag: Optional[str] = None,
    largest_component_only: bool = False,
    sentence_text: str = "",
    deactivated_concepts: Optional[List[str]] = None,
    positions: Optional[Dict[str, Tuple[float, float]]] = None,
) -> str:
    blue_nodes = set(blue_nodes) if blue_nodes is not None else set(DEFAULT_BLUE_NODES)
    out_dir = output_dir or os.path.join(OUTPUT_ANALYSIS_DIR, "graphs")
    os.makedirs(out_dir, exist_ok=True)

    def _sanitize(component: str, max_len: int = 50) -> str:
        component = (component or "").replace("\n", " ").strip()
        component = component[:max_len]
        # Replace path separators and common bad filename chars
        return re.sub(r"[\\/:*?\"<>|]", "_", component)

    safe_model = _sanitize(model_name, max_len=60)
    safe_persona = _sanitize(persona, max_len=40)
    suffix = f"_{step_tag}" if step_tag else ""
    filename = f"amoc_graph_{safe_model}_{safe_persona}_{age}{suffix}.png"
    save_path = os.path.join(out_dir, filename)

    G = nx.MultiDiGraph()
    for src, rel, dst in triplets:
        G.add_edge(src, dst, label=rel)

    # Remove isolates, optionally keep only largest component
    G.remove_nodes_from(list(nx.isolates(G)))
    if G.number_of_nodes() == 0:
        return save_path
    components = list(nx.connected_components(G.to_undirected()))
    if largest_component_only and components:
        largest = max(components, key=len)
        G = G.subgraph(largest).copy()

    plt.figure(figsize=(22, 18))

    node_colors = ["#a0cbe2" if node in blue_nodes else "#ffe8a0" for node in G.nodes()]

    pos: Dict[str, Tuple[float, float]] = {}
    nodes = list(G.nodes())

    if len(nodes) == 1:
        pos[nodes[0]] = (0.0, 0.0)
    else:
        # Start from any existing angles
        angles: Dict[str, float] = {}
        for node in nodes:
            if node in positions:
                x, y = positions[node]
                angles[node] = math.atan2(y, x)

        # Assign angles to new nodes deterministically, evenly spaced
        missing = [n for n in nodes if n not in angles]
        missing.sort()
        n_missing = len(missing)
        if n_missing > 0:
            for idx, node in enumerate(missing):
                angle = 2 * math.pi * idx / n_missing
                angles[node] = angle

        # Use a radius that scales with node count to reduce cramming
        radius = max(10.0, 3.0 + 0.8 * len(nodes))
        for node in nodes:
            angle = angles.get(node, 0.0)
            pos[node] = (radius * math.cos(angle), radius * math.sin(angle))

    nx.draw_networkx_nodes(
        G,
        pos,
        node_size=3800,
        node_color=node_colors,
        linewidths=2.0,
        edgecolors="black",
    )
    # Draw edges with per-edge curvature to separate parallel edges
    edge_labels = nx.get_edge_attributes(G, "label")
    parallel_groups: Dict[Tuple[str, str], List[Tuple[str, str, int]]] = {}
    for u, v, k in G.edges(keys=True):
        parallel_groups.setdefault((u, v), []).append((u, v, k))

    for (u, v), edges in parallel_groups.items():
        total = len(edges)
        offsets = [
            0.0 if total == 1 else (i - (total - 1) / 2) * 0.25 for i in range(total)
        ]
        for (u, v, k), rad in zip(edges, offsets):
            nx.draw_networkx_edges(
                G,
                pos,
                edgelist=[(u, v, k)],
                edge_color="black",
                arrows=False,
                width=1.2,
                connectionstyle=f"arc3,rad={rad}",
            )
            if edge_labels:
                nx.draw_networkx_edge_labels(
                    G,
                    pos,
                    edge_labels=edge_labels,
                    font_color="darkred",
                    font_size=9,
                    label_pos=0.5,
                    bbox=dict(facecolor="white", edgecolor="none", pad=0.2),
                )

    nx.draw_networkx_labels(
        G, pos, font_size=11, font_weight="bold", font_color="black"
    )

    title_persona = (persona[:150] + "...") if len(persona) > 150 else persona
    plt.title(f"AMoC Knowledge Graph: {model_name}", size=20, pad=20)
    sup_lines = [f"Persona: {title_persona}"]
    if sentence_text:
        sup_lines.append(f"Sentence: {sentence_text}")
    if deactivated_concepts is not None:
        if deactivated_concepts:
            sup_lines.append("Deactivated concepts: " + ", ".join(deactivated_concepts))
        else:
            sup_lines.append("Deactivated concepts: none")
    plt.suptitle(
        "\n".join(sup_lines),
        y=0.98,
        fontsize=12,
        style="italic",
        color="darkblue",
    )
    plt.axis("off")
    plt.savefig(save_path, format="PNG", dpi=300, bbox_inches="tight")
    plt.close()

    if positions is not None:
        positions.clear()
        positions.update(pos)

    return save_path
