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


def _set_axes_limits(ax, pos: Dict[str, Tuple[float, float]]) -> None:
    xs = [x for x, _ in pos.values()]
    ys = [y for _, y in pos.values()]
    if not xs or not ys:
        return
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    dx = max(1.0, max_x - min_x)
    dy = max(1.0, max_y - min_y)
    pad_x = dx * 0.18
    pad_y = dy * 0.18
    ax.set_xlim(min_x - pad_x, max_x + pad_x)
    ax.set_ylim(min_y - pad_y, max_y + pad_y)


def _node_required_center_distance_data(
    fig, ax, *, node_size: float, pad_px: float = 6.0
) -> float:
    # node_size is matplotlib scatter "s" in points^2 (area).
    fig.canvas.draw()
    dpi = fig.dpi
    radius_pts = math.sqrt(float(node_size) / math.pi)
    radius_px = radius_pts * dpi / 72.0
    # Convert px to data units via the current data transform.
    x0, y0 = ax.transData.transform((0.0, 0.0))
    x1, _ = ax.transData.transform((1.0, 0.0))
    _, y1 = ax.transData.transform((0.0, 1.0))
    px_per_x = max(1e-6, abs(x1 - x0))
    px_per_y = max(1e-6, abs(y1 - y0))
    radius_data = max((radius_px + pad_px) / px_per_x, (radius_px + pad_px) / px_per_y)
    return 2.0 * radius_data


def _ring_required_radius(n_ring: int, target_min_dist: float) -> float:
    if n_ring <= 1:
        return target_min_dist * 1.6
    # For two-node rings, keep edges off a straight line by using a 120° spread.
    angular_sep = (2.0 * math.pi / 3.0) if n_ring == 2 else (2.0 * math.pi / n_ring)
    chord_factor = 2.0 * math.sin(angular_sep / 2.0)
    if chord_factor <= 0.0:
        chord_factor = 1e-6
    return (target_min_dist / chord_factor) * 1.05


def _enforce_minimum_spacing(
    fig,
    ax,
    pos: Dict[str, Tuple[float, float]],
    hub: Optional[str],
    *,
    node_size: float = 3800,
    pad_px: float = 8.0,
    scale_step: float = 1.06,
    max_iter: int = 140,
    freeze_nodes: Optional[set[str]] = None,
) -> None:
    if len(pos) < 2:
        return
    for _ in range(max_iter):
        _set_axes_limits(ax, pos)
        required = _node_required_center_distance_data(
            fig, ax, node_size=node_size, pad_px=pad_px
        )
        if _min_pairwise_distance(pos) >= required:
            return
        # Radial scale (keep angles) until node circles no longer overlap.
        for node, (x, y) in list(pos.items()):
            if freeze_nodes and node in freeze_nodes:
                continue
            if hub is not None and node == hub:
                continue
            pos[node] = (x * scale_step, y * scale_step)


def _level_angle_seed(level: int) -> float:
    # Golden-angle based deterministic seed per ring to avoid collinear
    # single-node rings while keeping layouts stable across runs.
    golden_angle = math.pi * (3.0 - math.sqrt(5.0))
    return _wrap_angle(level * golden_angle)


def _point_segment_distance(
    px: float, py: float, ax: float, ay: float, bx: float, by: float
) -> Tuple[float, float, float]:
    dx = bx - ax
    dy = by - ay
    if dx == 0.0 and dy == 0.0:
        return math.hypot(px - ax, py - ay), ax, ay
    t = ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)
    t = max(0.0, min(1.0, t))
    proj_x = ax + t * dx
    proj_y = ay + t * dy
    return math.hypot(px - proj_x, py - proj_y), proj_x, proj_y


def _push_nodes_off_edges(
    fig,
    ax,
    pos: Dict[str, Tuple[float, float]],
    edges: List[Tuple[str, str]],
    *,
    node_size: float = 3800,
    pad_px: float = 12.0,
    corridor_scale: float = 1.25,
    max_iter: int = 140,
) -> None:
    if len(pos) < 3 or not edges:
        return
    edges_list = [(u, v) for u, v in edges if u in pos and v in pos]
    if not edges_list:
        return
    for _ in range(max_iter):
        _set_axes_limits(ax, pos)
        required = _node_required_center_distance_data(
            fig, ax, node_size=node_size, pad_px=pad_px
        )
        corridor = required * corridor_scale
        moved = False
        for node, (x, y) in list(pos.items()):
            for u, v in edges_list:
                if node == u or node == v:
                    continue
                ux, uy = pos[u]
                vx, vy = pos[v]
                dist, proj_x, proj_y = _point_segment_distance(x, y, ux, uy, vx, vy)
                if dist >= corridor:
                    continue
                dx = x - proj_x
                dy = y - proj_y
                norm = math.hypot(dx, dy)
                if norm < 1e-6:
                    # Nudge sideways if perfectly aligned
                    dx, dy = -(vy - uy), vx - ux
                    norm = math.hypot(dx, dy)
                dx /= norm
                dy /= norm
                gap = corridor - dist
                step = max(gap * 1.25, required * 0.12)
                x += dx * step
                y += dy * step
                moved = True
            pos[node] = (x, y)
        if not moved:
            break


def _enforce_min_edge_length(
    pos: Dict[str, Tuple[float, float]],
    edges: List[Tuple[str, str]],
    *,
    min_len: float,
    hub: Optional[str] = None,
    step_factor: float = 1.04,
    max_iter: int = 80,
) -> None:
    if min_len <= 0 or not edges:
        return
    for _ in range(max_iter):
        moved = False
        for u, v in edges:
            if u not in pos or v not in pos:
                continue
            x1, y1 = pos[u]
            x2, y2 = pos[v]
            dx = x2 - x1
            dy = y2 - y1
            dist = math.hypot(dx, dy)
            if dist >= min_len:
                continue
            if dist < 1e-6:
                dx, dy, dist = 1.0, 0.0, 1.0
            deficit = (min_len - dist) * step_factor
            if hub is not None and u == hub and v != hub:
                # Keep hub fixed; push v away from hub.
                pos[v] = (x2 + (dx / dist) * deficit, y2 + (dy / dist) * deficit)
            elif hub is not None and v == hub and u != hub:
                pos[u] = (x1 - (dx / dist) * deficit, y1 - (dy / dist) * deficit)
            else:
                shift = deficit * 0.5
                pos[u] = (x1 - (dx / dist) * shift, y1 - (dy / dist) * shift)
                pos[v] = (x2 + (dx / dist) * shift, y2 + (dy / dist) * shift)
            moved = True
        if not moved:
            break


def _choose_angles_in_gaps(
    *,
    existing_angles: Dict[str, float],
    missing_nodes: List[str],
    candidate_count: int,
    penalty_target: Dict[str, float] | None = None,
    penalty_weight: float = 0.0,
) -> Dict[str, float]:
    if not missing_nodes:
        return {}
    missing_nodes = list(missing_nodes)
    candidates = [2.0 * math.pi * i / candidate_count for i in range(candidate_count)]
    occupied: List[float] = list(existing_angles.values())
    chosen: Dict[str, float] = {}
    for node in missing_nodes:
        best_angle: float | None = None
        best_score = -1e9
        target = penalty_target.get(node) if penalty_target else None
        for cand in candidates:
            min_sep = min(
                (_circular_distance(cand, occ) for occ in occupied), default=math.pi
            )
            score = min_sep
            if target is not None and penalty_weight:
                score -= penalty_weight * _circular_distance(cand, target)
            if score > best_score:
                best_score = score
                best_angle = cand
        if best_angle is None:
            best_angle = 0.0
        chosen[node] = best_angle
        occupied.append(best_angle)
        try:
            candidates.remove(best_angle)
        except ValueError:
            pass
    return chosen


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
    new_nodes: Optional[List[str]] = None,
    explicit_nodes: Optional[Iterable[str]] = None,
    salient_nodes: Optional[Iterable[str]] = None,
    inactive_nodes: Optional[Iterable[str]] = None,
    positions: Optional[Dict[str, Tuple[float, float]]] = None,
    avoid_edge_overlap: bool = True,
    active_edges: Optional[set[Tuple[str, str]]] = None,
    hub_edge_explanations: Optional[List[str]] = None,
) -> str:

    def expand_by_anchor(
        G_full: nx.Graph,
        G_snapshot: nx.Graph,
        anchors: set[str],
        hops: int = 2,
    ) -> nx.Graph:
        # Only expand within connected components that contain anchors
        # to prevent pulling in disconnected parts of the graph
        nodes_to_keep = set(G_snapshot.nodes())

        # Find all connected components in G_full
        if G_full.number_of_nodes() == 0:
            return G_full.subgraph(nodes_to_keep).copy()

        # Get the connected component(s) containing at least one anchor
        anchor_components: set[frozenset[str]] = set()
        for component in nx.connected_components(G_full):
            component_set = frozenset(component)
            if any(anchor in component_set for anchor in anchors):
                anchor_components.add(component_set)

        # Only consider nodes from components that contain anchors
        valid_nodes = set()
        for comp in anchor_components:
            valid_nodes |= set(comp)

        # Now expand from anchors, but only within valid components
        for anchor in anchors:
            if anchor not in G_full:
                continue
            if anchor not in valid_nodes:
                continue

            reachable = nx.single_source_shortest_path_length(
                G_full, anchor, cutoff=hops
            )
            # Only keep reachable nodes that are in valid components
            nodes_to_keep |= set(reachable.keys()) & valid_nodes

        return G_full.subgraph(nodes_to_keep).copy()

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

    # Directed graph so reciprocal edges can be visualized (e.g., A→B and B→A).
    G = nx.DiGraph()
    edge_labels: Dict[Tuple[str, str], str] = {}
    duplicate_edges: set[Tuple[str, str]] = set()
    active_edge_set = active_edges or set()
    # Build full graph before filtering for anchoring purposes.
    G_full = nx.DiGraph()
    for src, rel, dst in triplets:
        src = str(src).strip()
        dst = str(dst).strip()
        rel = str(rel).strip()
        if not src or not dst:
            continue
        u, v = src, dst
        if (u, v) in edge_labels:
            duplicate_edges.add((u, v))
            continue
        G.add_edge(u, v)
        G_full.add_edge(u, v)
        edge_labels[(u, v)] = rel
        edge_labels[(v, u)] = rel  # allow undirected lookups

    # Inject explicit / salient / inactive nodes so they are retained even if isolated.
    def _inject_nodes(nodes):
        if not nodes:
            return
        for node in nodes:
            if not node:
                continue
            if node not in G:
                G.add_node(node)
            if node not in G_full:
                G_full.add_node(node)

    _inject_nodes(explicit_nodes)
    _inject_nodes(salient_nodes)
    _inject_nodes(inactive_nodes)

    # Remove isolates, optionally keep only largest component, removed it to keep the
    if "sentence" in (step_tag or "") and active_edge_set:
        # Only remove nodes that are neither incident to active edges nor provided anchors
        nodes_with_edges = {u for e in G.edges() for u in e}
        nodes_to_keep = (
            nodes_with_edges
            | set(explicit_nodes or [])
            | set(salient_nodes or [])
            | set(inactive_nodes or [])
        )
        G.remove_nodes_from([n for n in G.nodes() if n not in nodes_to_keep])
    if G.number_of_nodes() == 0:
        return save_path
    components = list(nx.connected_components(G.to_undirected()))
    if largest_component_only and components:
        largest = max(components, key=len)
        G = G.subgraph(largest).copy()

    # SENTENCE ANCHORING    ----------------------------
    anchors: set[str] = set()
    if explicit_nodes:
        anchors |= {n for n in explicit_nodes if n in G_full}
    if not anchors:
        anchors = {n for n in G.nodes() if G.degree(n) > 0}
    if not anchors and G_full.nodes:
        anchors = {next(iter(G_full.nodes()))}
    G_draw = expand_by_anchor(
        G_full=G_full.to_undirected(),
        G_snapshot=G.to_undirected(),
        anchors=anchors,
        hops=2,
    )
    G = G_draw
    # Bridge components for visualization so all provided nodes appear connected.
    components_draw = list(nx.connected_components(G_draw))
    if len(components_draw) > 1:
        anchor_node = None
        for candidate in (explicit_nodes or []):
            if candidate in G_draw:
                anchor_node = candidate
                break
        if anchor_node is None and components_draw:
            anchor_node = next(iter(components_draw[0]))
        if anchor_node is not None:
            for comp in components_draw:
                if anchor_node in comp:
                    continue
                rep = next(iter(comp))
                G_draw.add_edge(anchor_node, rep)
                edge_labels[(anchor_node, rep)] = edge_labels.get(
                    (anchor_node, rep), ""
                )
    G = G_draw
    # --------------------------------------

    fig, ax = plt.subplots(figsize=(22, 18))

    node_colors = ["#a0cbe2" if node in blue_nodes else "#ffe8a0" for node in G.nodes()]

    pos: Dict[str, Tuple[float, float]] = {}
    nodes = list(G.nodes())
    position_cache = positions or {}
    fixed_pos: Dict[str, Tuple[float, float]] = {
        node: position_cache[node] for node in nodes if node in position_cache
    }
    fixed_nodes = set(fixed_pos.keys())
    hub: Optional[str] = None

    max_label_len = max((len(_pretty_text(n)) for n in nodes), default=0)
    target_min_dist = 7.0 + max(0, max_label_len - 10) * 0.12

    if len(nodes) == 1:
        pos[nodes[0]] = (0.0, 0.0)
        hub = nodes[0]
    else:
        UG = G

        # Single-hub radial layout (hub centered) to keep the nodes at fixed positions
        if (
            positions is not None
            and "__HUB__" in positions
            and positions["__HUB__"] in nodes
        ):
            hub = positions["__HUB__"]
        else:
            hub_candidates = [n for n in nodes if n in blue_nodes] or nodes
            hub = max(hub_candidates, key=lambda n: (UG.degree(n), str(n)))
            if positions is not None:
                positions["__HUB__"] = hub

        pos[hub] = (0.0, 0.0)
        freeze_nodes = set(fixed_nodes)
        if hub is not None:
            freeze_nodes.add(hub)

        levels = nx.single_source_shortest_path_length(UG, hub)
        max_level = max(levels.values(), default=0)

        # Include disconnected nodes (if any) on an outer ring.
        unreachable = sorted([n for n in nodes if n not in levels and n != hub])
        if unreachable:
            max_level += 1
            for node in unreachable:
                levels[node] = max_level

        ring_step = max(12.0, target_min_dist * 1.55)
        radii: Dict[int, float] = {}

        # If we have cached coordinates, keep existing nodes fixed and only
        # place newly appearing nodes.
        if fixed_pos:
            pos = dict(fixed_pos)
            if hub not in pos:
                pos[hub] = (0.0, 0.0)
            movable_nodes = [n for n in nodes if n not in pos]

            # Precompute ring radii based on both fixed + new nodes per level,
            # but never pull a new node inward past any already-placed node.
            max_r_by_level: Dict[int, float] = {}
            for node, (x, y) in pos.items():
                if node == hub:
                    continue
                lvl = levels.get(node)
                if lvl is None:
                    continue
                max_r_by_level[lvl] = max(
                    max_r_by_level.get(lvl, 0.0), math.hypot(x, y)
                )
            for level in range(1, max_level + 1):
                ring_nodes = [n for n in nodes if levels.get(n) == level]
                n_ring = len(ring_nodes)
                required_r = _ring_required_radius(n_ring, target_min_dist)
                prev_r = radii.get(level - 1, 0.0)
                radii[level] = max(
                    required_r, prev_r + ring_step, max_r_by_level.get(level, 0.0)
                )

            # Place new nodes ring-by-ring, filling angular gaps around already
            # placed nodes (if any).
            for level in range(1, max_level + 1):
                ring_new = sorted([n for n in movable_nodes if levels.get(n) == level])
                if not ring_new:
                    continue
                r = radii[level]
                default_angle = _level_angle_seed(level)
                existing_angles = {
                    n: _wrap_angle(math.atan2(y, x))
                    for n, (x, y) in pos.items()
                    if n != hub and levels.get(n) == level and (x != 0.0 or y != 0.0)
                }
                chosen = _choose_angles_in_gaps(
                    existing_angles=existing_angles,
                    missing_nodes=ring_new,
                    candidate_count=max(12, len(ring_new) + len(existing_angles) + 6),
                )
                for node in ring_new:
                    angle = chosen.get(node, default_angle)
                    pos[node] = (r * math.cos(angle), r * math.sin(angle))

            # Any node still not placed (shouldn't happen) goes to an outer ring.
            max_existing_r = max(
                (math.hypot(x, y) for x, y in pos.values()), default=0.0
            )
            for node in movable_nodes:
                if node in pos:
                    continue
                r = max_existing_r + ring_step
                pos[node] = (r, 0.0)

            # Collision-avoidance pass that only moves newly placed nodes
            # (keeps cached nodes in place).
            movable_sorted = sorted(movable_nodes)
            for _ in range(180):
                _set_axes_limits(ax, pos)
                required = _node_required_center_distance_data(
                    fig, ax, node_size=3800, pad_px=8.0
                )
                moved = False
                for node in movable_nodes:
                    x, y = pos[node]
                    if x == 0.0 and y == 0.0:
                        x = required
                        y = 0.0
                    # Push outward until no overlap with any other node.
                    for _inner in range(12):
                        min_dist = float("inf")
                        for other, (ox, oy) in pos.items():
                            if other == node:
                                continue
                            d = math.hypot(x - ox, y - oy)
                            if d < min_dist:
                                min_dist = d
                        if min_dist >= required:
                            break
                        x *= 1.10
                        y *= 1.10
                        moved = True
                    pos[node] = (x, y)
                # Also ensure movable nodes don't overlap each other after scaling.
                if not moved:
                    # Quick check
                    any_overlap = False
                    for i1, n1 in enumerate(movable_sorted):
                        x1, y1 = pos[n1]
                        for n2 in movable_sorted[i1 + 1 :]:
                            x2, y2 = pos[n2]
                            if math.hypot(x1 - x2, y1 - y2) < required:
                                any_overlap = True
                                break
                        if any_overlap:
                            break
                    if not any_overlap:
                        break
        else:
            # No cache: compute a full radial layout from scratch.
            # Precompute counts per ring to choose radii that avoid overlap on the ring.
            for level in range(1, max_level + 1):
                ring_nodes = [n for n in nodes if levels.get(n) == level]
                n_ring = len(ring_nodes)
                required_r = _ring_required_radius(n_ring, target_min_dist)
                prev_r = radii.get(level - 1, 0.0)
                radii[level] = max(required_r, prev_r + ring_step)

            pos = {hub: (0.0, 0.0)}
            for level in range(1, max_level + 1):
                ring = sorted([n for n in nodes if levels.get(n) == level])
                if not ring:
                    continue
                r = radii[level]
                n_ring = len(ring)

                if n_ring == 1:
                    angles = [_level_angle_seed(level)]
                elif n_ring == 2:
                    base = _level_angle_seed(level)
                    spread = 2.0 * math.pi / 3.0  # 120° to avoid a straight line
                    angles = [
                        _wrap_angle(base - spread / 2.0),
                        _wrap_angle(base + spread / 2.0),
                    ]
                else:
                    angles = [2.0 * math.pi * idx / n_ring for idx in range(n_ring)]

                for idx, node in enumerate(ring):
                    angle = (
                        angles[idx]
                        if idx < len(angles)
                        else 2.0 * math.pi * idx / n_ring
                    )
                    pos[node] = (r * math.cos(angle), r * math.sin(angle))

            # Scale outward until node circles cannot overlap in screen space.
            for _ in range(120):
                _set_axes_limits(ax, pos)
                required = _node_required_center_distance_data(
                    fig, ax, node_size=3800, pad_px=8.0
                )
                if _min_pairwise_distance(pos) >= required:
                    break
                for node, (x, y) in list(pos.items()):
                    if node == hub:
                        continue
                    pos[node] = (x * 1.08, y * 1.08)
            _set_axes_limits(ax, pos)
            required = _node_required_center_distance_data(
                fig, ax, node_size=3800, pad_px=8.0
            )
            min_dist = _min_pairwise_distance(pos)
            if min_dist > 0 and min_dist < required:
                factor = (required / min_dist) * 1.02
                for node, (x, y) in list(pos.items()):
                    if node == hub:
                        continue
                    pos[node] = (x * factor, y * factor)

    if pos and hub is None:
        hub = next(iter(pos.keys()))
    min_edge_len = max(6.5, target_min_dist * 0.9)
    _enforce_min_edge_length(pos, list(G.edges()), min_len=min_edge_len, hub=hub)
    if avoid_edge_overlap:
        _push_nodes_off_edges(fig, ax, pos, list(G.edges()))
    _enforce_minimum_spacing(
        fig,
        ax,
        pos,
        hub,
        node_size=3800,
        pad_px=8.0,
        scale_step=1.06,
        max_iter=140,
        freeze_nodes=fixed_nodes,
    )
    if avoid_edge_overlap:
        _push_nodes_off_edges(fig, ax, pos, list(G.edges()))

    nx.draw_networkx_nodes(
        G,
        pos,
        node_size=3800,
        node_color=node_colors,
        linewidths=2.0,
        edgecolors="black",
        ax=ax,
    )

    reciprocals: set[Tuple[str, str]] = {
        (u, v) for u, v in G.edges() if (v, u) in G.edges()
    }

    # styling to distinguish between
    edge_colors = []
    edge_widths = []

    for u, v in G.edges():
        if (u, v) in active_edge_set:
            edge_colors.append("black")
            edge_widths.append(1.3)
        else:
            edge_colors.append("#cccccc")
            edge_widths.append(1.2)

    # Draw all edges as straight lines; for reciprocal pairs, place labels on
    # opposite sides of the segment (using a perpendicular offset) to avoid
    # overlap while keeping the geometry straight.
    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=list(G.edges()),
        edge_color=edge_colors,
        arrows=True,
        arrowsize=16,
        width=edge_widths,
        connectionstyle="arc3,rad=0.0",
        ax=ax,
    )

    def _label_offset(u: str, v: str) -> Tuple[float, float]:
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        dx, dy = x2 - x1, y2 - y1
        dist = math.hypot(dx, dy)
        if dist < 1e-6:
            return (x1 + x2) * 0.5, (y1 + y2) * 0.5
        # Perpendicular unit vector
        nx_ = -dy / dist
        ny_ = dx / dist
        base = 0.0  # place label on the edge axis (no perpendicular shift)
        # For reciprocals, push labels to opposite sides and stagger along the edge.
        if (u, v) in reciprocals:
            sign = 1.0 if u < v else -1.0
            t = 0.58 if u < v else 0.42
        else:
            sign = 0.0
            t = 0.5
        mx = x1 * (1.0 - t) + x2 * t
        my = y1 * (1.0 - t) + y2 * t
        return mx + sign * base * nx_, my + sign * base * ny_

    for u, v in G.edges():
        if (u, v) not in edge_labels:
            continue
        lx, ly = _label_offset(u, v)
        ax.text(
            lx,
            ly,
            _pretty_text(edge_labels[(u, v)]),
            fontsize=9,
            color="darkred",
            ha="center",
            va="center",
            bbox=dict(facecolor="white", edgecolor="none", pad=0.2),
        )

    node_labels = {n: _pretty_text(n) for n in G.nodes()}
    nx.draw_networkx_labels(
        G,
        pos,
        labels=node_labels,
        font_size=11,
        font_weight="bold",
        font_color="black",
        ax=ax,
    )

    def _normalize_title_line(text: str, max_len: int = 220) -> str:
        # Avoid extremely long headers (e.g., if a full prompt leaks in) that
        # blow up the canvas and distort node placement by compacting and
        # truncating the line.
        cleaned = re.sub(r"<[^>]+>", " ", text or "")
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        if len(cleaned) > max_len:
            cleaned = cleaned[: max_len - 3].rstrip() + "..."
        return cleaned

    def _format_nodes_line(label: str, nodes: Optional[Iterable[str]]) -> Optional[str]:
        if nodes is None:
            return None
        cleaned = [_pretty_text(n) for n in nodes if n]
        if not cleaned:
            return f"{label}: none"
        return f"{label}: " + ", ".join(cleaned)

    title_persona = (persona[:150] + "...") if len(persona) > 150 else persona
    ax.set_title(f"AMoC Knowledge Graph: {model_name}", size=20, pad=20)
    sup_lines = []
    persona_line = _normalize_title_line(title_persona, max_len=180)
    sentence_idx = None
    if persona_line:
        sup_lines.append(f"Persona: {persona_line}")
    if sentence_text:
        sentence_line = _normalize_title_line(sentence_text, max_len=220)
        if step_tag and "sent" in step_tag:
            # Extract sentence index from step_tag if present (e.g., sent3_active)
            import re as _re

            m = _re.search(r"sent(\d+)", step_tag)
            if m:
                sentence_idx = int(m.group(1))
                sentence_line = f"{sentence_idx}: {sentence_line}"
        sup_lines.append(f"Sentence {sentence_line}")
    inactive_for_title = (
        inactive_nodes if inactive_nodes is not None else deactivated_concepts
    )
    if inactive_for_title is not None:
        sup_lines.append("\n")
    for label, items in [
        ("Explicit this sentence", explicit_nodes),
        (
            "Carry-over from previous sentences",
            salient_nodes if (sentence_idx is not None and sentence_idx > 1) else None,
        ),
        ("Inactive nodes (retained in memory)", inactive_for_title),
    ]:
        line = _format_nodes_line(label, items)
        if line:
            sup_lines.append(line)
    if explicit_nodes is None and salient_nodes is None:
        fallback = _format_nodes_line("Active this sentence", new_nodes)
        if fallback:
            sup_lines.append(fallback)
    # Add hub edge explanations if provided (when explicit nodes needed hub-anchoring)
    if hub_edge_explanations:
        sup_lines.append("\n")
        sup_lines.append("Hub-anchored connections (LLM explanation):")
        for explanation in hub_edge_explanations[:3]:  # Limit to 3 to avoid clutter
            truncated = (explanation[:120] + "...") if len(explanation) > 120 else explanation
            sup_lines.append(f"  • {truncated}")
    plt.suptitle(
        "\n".join(sup_lines),
        y=0.98,
        fontsize=12,
        style="italic",
        color="darkblue",
    )
    ax.axis("off")
    fig.savefig(save_path, format="PNG", dpi=300, bbox_inches="tight")
    plt.close(fig)

    if positions is not None:
        # Keep existing coordinates for nodes not present in this snapshot so
        # layout stays stable as nodes disappear/reappear across sentences.
        positions.update(pos)

    return save_path
