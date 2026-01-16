import os
import re
import math
from typing import List, Tuple, Iterable, Optional, Dict
import networkx as nx
import matplotlib.pyplot as plt
from amoc.config.paths import OUTPUT_ANALYSIS_DIR  # reuse existing output base

DEFAULT_BLUE_NODES: Iterable[str] = ()

TRIVIAL_NODE_TEXTS = {
    "and",
    "or",
    "but",
    "nor",
    "so",
    "yet",
    "through",
    "with",
    "without",
    "to",
    "of",
    "in",
    "on",
    "at",
    "from",
    "into",
    "onto",
    "by",
    "for",
    "about",
    "over",
    "under",
    "after",
    "before",
    "during",
    "while",
    "as",
}


def _pretty_text(text: str) -> str:
    text = (text or "").strip()
    text = text.replace("_", " ")
    text = re.sub(r"\s+", " ", text)
    return text


def _is_trivial_node_text(text: str) -> bool:
    return _pretty_text(text).lower() in TRIVIAL_NODE_TEXTS


def _wrap_angle(angle: float) -> float:
    two_pi = 2.0 * math.pi
    return angle % two_pi


def _circular_distance(a: float, b: float) -> float:
    two_pi = 2.0 * math.pi
    diff = abs(a - b) % two_pi
    return min(diff, two_pi - diff)

def _min_pairwise_distance(pos: Dict[str, Tuple[float, float]]) -> float:
    coords = list(pos.values())
    if len(coords) < 2:
        return float("inf")
    min_dist = float("inf")
    for i in range(len(coords)):
        x1, y1 = coords[i]
        for j in range(i + 1, len(coords)):
            x2, y2 = coords[j]
            dist = math.hypot(x1 - x2, y1 - y2)
            if dist < min_dist:
                min_dist = dist
    return min_dist


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
        src = str(src).strip()
        dst = str(dst).strip()
        rel = str(rel).strip()
        if not src or not dst:
            continue
        if _is_trivial_node_text(src) or _is_trivial_node_text(dst):
            continue
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
    previous_positions = positions or {}
    target_min_dist = 7.0

    if len(nodes) == 1:
        pos[nodes[0]] = (0.0, 0.0)
    else:
        UG = G.to_undirected()

        # Pick a stable "hub" node (center) to produce a radial layout.
        hub_candidates = [n for n in nodes if n in blue_nodes] or nodes
        hub = max(hub_candidates, key=lambda n: (UG.degree(n), str(n)))

        levels = nx.single_source_shortest_path_length(UG, hub)
        max_level = max(levels.values(), default=0)

        parents: Dict[str, str] = {}
        for node in nodes:
            if node == hub:
                continue
            level = levels.get(node)
            if level is None or level == 0:
                continue
            prevs = [nbr for nbr in UG.neighbors(node) if levels.get(nbr) == level - 1]
            if prevs:
                parents[node] = max(prevs, key=lambda n: (UG.degree(n), str(n)))

        # Start from any existing angles for stability across snapshots
        angles: Dict[str, float] = {hub: 0.0}
        for node in nodes:
            if node == hub:
                continue
            if node in previous_positions:
                x, y = previous_positions[node]
                angles[node] = _wrap_angle(math.atan2(y, x))

        candidate_count = max(64, 8 * len(nodes))
        base_candidates = [2.0 * math.pi * i / candidate_count for i in range(candidate_count)]

        for level in range(1, max_level + 1):
            ring = sorted([n for n in nodes if levels.get(n) == level])
            occupied = [angles[n] for n in ring if n in angles]
            missing = [n for n in ring if n not in angles]
            candidates = base_candidates.copy()

            for node in missing:
                parent_angle = angles.get(parents.get(node, ""))
                best_angle: float | None = None
                best_score = -1.0
                for cand in candidates:
                    min_sep = min(
                        (_circular_distance(cand, occ) for occ in occupied),
                        default=math.pi,
                    )
                    score = min_sep
                    if parent_angle is not None:
                        score -= 0.25 * _circular_distance(cand, parent_angle)
                    if score > best_score:
                        best_score = score
                        best_angle = cand
                if best_angle is None:
                    best_angle = 0.0
                angles[node] = best_angle
                occupied.append(best_angle)
                try:
                    candidates.remove(best_angle)
                except ValueError:
                    pass

        # Any nodes not reachable from the hub (disconnected component):
        # place them on an outer ring so they don't overlap the main component.
        unreachable = sorted([n for n in nodes if n not in levels and n != hub])
        if unreachable:
            outer_level = max_level + 1
            for idx, node in enumerate(unreachable):
                angles[node] = 2.0 * math.pi * idx / max(1, len(unreachable))
            max_level = outer_level

        level1_count = sum(1 for n in nodes if levels.get(n) == 1)
        base_radius = max(12.0, 6.0 + 0.9 * level1_count)
        ring_step = max(9.0, 6.0 + 0.35 * len(nodes))

        max_label_len = max((len(_pretty_text(n)) for n in nodes), default=0)
        target_min_dist = 7.0 + max(0, max_label_len - 10) * 0.12

        scale = 1.0
        for _ in range(25):
            pos = {hub: (0.0, 0.0)}
            for node in nodes:
                if node == hub:
                    continue
                level = levels.get(node, max_level)
                r = (base_radius + max(0, level - 1) * ring_step) * scale
                angle = angles.get(node, 0.0)
                pos[node] = (r * math.cos(angle), r * math.sin(angle))
            if _min_pairwise_distance(pos) >= target_min_dist:
                break
            scale *= 1.12

    nx.draw_networkx_nodes(
        G,
        pos,
        node_size=3800,
        node_color=node_colors,
        linewidths=2.0,
        edgecolors="black",
    )
    ax = plt.gca()

    # Draw edges with per-edge curvature to separate parallel edges
    edge_labels = nx.get_edge_attributes(G, "label")
    placed_label_points: List[Tuple[float, float]] = []
    node_clearance = max(4.0, target_min_dist * 0.55)
    label_clearance = 2.2

    def _label_collides(x: float, y: float) -> bool:
        for nx_, ny_ in pos.values():
            if math.hypot(x - nx_, y - ny_) < node_clearance:
                return True
        for lx, ly in placed_label_points:
            if math.hypot(x - lx, y - ly) < label_clearance:
                return True
        return False

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
            label = edge_labels.get((u, v, k))
            if label:
                label = _pretty_text(label)
                x1, y1 = pos[u]
                x2, y2 = pos[v]
                dx, dy = x2 - x1, y2 - y1
                length = math.hypot(dx, dy)
                if length <= 1e-9:
                    continue

                ux, uy = dx / length, dy / length
                nx_, ny_ = -uy, ux  # perpendicular

                base_shift = 0.9 + 1.6 * abs(rad) + 0.02 * len(label)
                t_candidates = [0.35, 0.5, 0.65]
                mult_candidates = [0.0, 1.0, -1.0, 1.8, -1.8, 2.6, -2.6]

                chosen_xy: Tuple[float, float] | None = None
                for t in t_candidates:
                    px = x1 + dx * t
                    py = y1 + dy * t
                    # Push labels "outward" relative to the origin to reduce central clutter.
                    sign = 1.0 if (px * nx_ + py * ny_) >= 0 else -1.0
                    for mult in mult_candidates:
                        lx = px + sign * mult * base_shift * nx_
                        ly = py + sign * mult * base_shift * ny_
                        if not _label_collides(lx, ly):
                            chosen_xy = (lx, ly)
                            break
                    if chosen_xy is not None:
                        break

                if chosen_xy is None:
                    chosen_xy = (x1 + dx * 0.5, y1 + dy * 0.5)

                ax.text(
                    chosen_xy[0],
                    chosen_xy[1],
                    label,
                    color="darkred",
                    fontsize=9,
                    ha="center",
                    va="center",
                    rotation=0,
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=0.2),
                )
                placed_label_points.append(chosen_xy)

    node_labels = {n: _pretty_text(n) for n in G.nodes()}
    nx.draw_networkx_labels(
        G,
        pos,
        labels=node_labels,
        font_size=11,
        font_weight="bold",
        font_color="black",
    )

    title_persona = (persona[:150] + "...") if len(persona) > 150 else persona
    plt.title(f"AMoC Knowledge Graph: {model_name}", size=20, pad=20)
    sup_lines = [f"Persona: {title_persona}"]
    if sentence_text:
        sup_lines.append(f"Sentence: {sentence_text}")
    if deactivated_concepts is not None:
        if deactivated_concepts:
            sup_lines.append(
                "Deactivated concepts: "
                + ", ".join(_pretty_text(c) for c in deactivated_concepts)
            )
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
