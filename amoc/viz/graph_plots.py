import os
import re
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

    # Radial-ish layout: place the highest-degree node at the center shell, others around it.
    pos: Dict[str, Tuple[float, float]]
    if G.number_of_nodes() > 2:
        degrees = dict(G.degree())
        if degrees:
            hub = max(degrees, key=degrees.get)
            others = [n for n in G.nodes() if n != hub]
            pos = nx.shell_layout(G, nlist=[[hub], others], scale=3.0)
        else:
            pos = nx.spring_layout(G, k=2.8, iterations=200, seed=42)
    else:
        pos = nx.spring_layout(G, k=2.8, iterations=200, seed=42)

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
    plt.suptitle(
        f"Persona: {title_persona}",
        y=0.98,
        fontsize=12,
        style="italic",
        color="darkblue",
    )
    plt.axis("off")
    plt.savefig(save_path, format="PNG", dpi=300, bbox_inches="tight")
    plt.close()
    return save_path
