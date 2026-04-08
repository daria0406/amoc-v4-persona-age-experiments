REACTIVATION_VISIBILITY = 2

sentences = [
    "A man very close to Charlemagne wrote most of the things we know about this famous king.",
    "These writings tell us that Charlemagne took pride in his ability to ride, swim, and hunt.",
    "He dressed like other Franks in linen shirt and breeches, a tunic, and hose wrapped with bands.",
    "Only at festivals did he dress in beautiful clothes.",
    "Charlemagne never dined without his children when he was home.",
    "At mealtime he liked to have someone read history and the deeds of great men aloud.",
    "When he went on a journey, the children often went with him.",
    "His sons rode at his side and his daughters followed him.",
    "As soon as his sons were old enough, the king taught them to ride horseback, to practice using weapons, and to hunt.",
    "He had his daughters taught to spin and weave.",
    "As you will see later, he set up a school so those in the palace could learn to read and write.",
    "Charlemagne wanted to bring all the Teutonic peoples together into one Christian empire.",
    "To do this he had to conquer tribes outside his kingdom such as the Saxons.",
    "Time and again he crossed the Rhine and made war on them until they came under his rule.",
    "Charlemagne added the Lombards of northern Italy to his kingdom by capturing the Lombard king.",
    "He put on the king's crown and declared himself king of the Lombards.",
    "Charlemagne had to protect the kingdom against some fierce Asiatic tribes which had settled southeast of Frankland.",
    "To show they were conquered, these tribes paid him a certain amount of goods each year.",
]


def mentions_king(sent):
    s = sent.lower()
    return "charlemagne" in s and "king" in s


def mentions_famous(sent):
    s = sent.lower()
    return "charlemagne" in s and "famous" in s


def estimate_llm_score_king(sent):
    """LLM relevance score for (charlemagne, is, king)."""
    s = sent.lower()
    if "charlemagne" in s or "king" in s or "kingdom" in s or "crown" in s or "rule" in s:
        return 2
    if any(w in s for w in ["conquer", "empire", "reign"]):
        return 2
    if s.startswith("he ") or "his " in s:
        return 1
    return 0


def estimate_llm_score_famous(sent):
    """LLM relevance score for (charlemagne, is, famous)."""
    s = sent.lower()
    if "charlemagne" in s or "famous" in s or "pride" in s or "great" in s:
        return 2
    if s.startswith("he ") or "his " in s:
        return 1
    return 0


def compute_status(vis, created_at, i, asserted, reactivated, vis_before):
    """Match recorder.py status logic exactly."""
    if vis <= 0:
        return "inactive"
    elif created_at == i and asserted and vis == 2:
        return "asserted"
    elif asserted:
        return "explicit"
    elif reactivated:
        return "reactivated"
    elif vis == 1:
        return "decaying"
    else:
        # vis=2, not asserted/reactivated → no carryover label
        return "inactive"


def step_edge(vis, created_at, i, mentions_fn, llm_score_fn, sent):
    """
    Returns (vis_after, status, action, llm_score, asserted, reactivated).
    vis=None means edge not yet created.
    """
    asserted = False
    reactivated = False
    vis_before = vis

    if vis is None:
        if mentions_fn(sent):
            vis = REACTIVATION_VISIBILITY
            created_at = i
            asserted = True
            status = "asserted"
            return vis, created_at, status, "CREATED", "—", asserted, reactivated
        else:
            return None, None, "—", "not yet created", "—", False, False

    if mentions_fn(sent):
        vis_before_mention = vis
        vis = REACTIVATION_VISIBILITY
        asserted = True
        if vis_before_mention <= 0:
            reactivated = True  # was dead, now asserted — treat as reactivated+asserted
        status = compute_status(vis, created_at, i, asserted, reactivated, vis_before)
        return vis, created_at, status, "MENTIONED (skip decay)", "—", asserted, reactivated

    score = llm_score_fn(sent)

    if score == 0 or score == 1:
        vis -= 1
        if vis <= 0:
            vis = 0
        action = "reduce_visibility()"
    elif score == 2:
        if vis <= 0:
            vis = REACTIVATION_VISIBILITY
            reactivated = True
            action = f"reactivated (0→{REACTIVATION_VISIBILITY})"
        else:
            action = "maintain (score=2)"

    status = compute_status(vis, created_at, i, asserted, reactivated, vis_before)
    return vis, created_at, status, action, score, asserted, reactivated


def simulate():
    vis_king = None
    vis_famous = None
    created_king = None
    created_famous = None

    col_sent = 4
    col_vis = 3
    col_stat = 12

    header = (
        f"{'S':>{col_sent}} | "
        f"{'(is,king) vis':>{col_vis}} | "
        f"{'status':<{col_stat}} | "
        f"{'score':>5} | "
        f"{'(is,famous) vis':>{col_vis+2}} | "
        f"{'status':<{col_stat}} | "
        f"{'score':>5} | "
        f"sentence"
    )
    print(header)
    print("-" * len(header))

    for i, sent in enumerate(sentences):
        vis_king, created_king, stat_king, action_king, score_king, _, _ = step_edge(
            vis_king, created_king, i, mentions_king, estimate_llm_score_king, sent
        )
        vis_famous, created_famous, stat_famous, action_famous, score_famous, _, _ = step_edge(
            vis_famous, created_famous, i, mentions_famous, estimate_llm_score_famous, sent
        )

        vk = "—" if vis_king is None else vis_king
        vf = "—" if vis_famous is None else vis_famous
        sk = stat_king if stat_king != "—" else "—"
        sf = stat_famous if stat_famous != "—" else "—"
        short = sent[:55] + "..." if len(sent) > 55 else sent

        print(
            f"{i+1:>{col_sent}} | "
            f"{str(vk):>{col_vis}} | "
            f"{sk:<{col_stat}} | "
            f"{str(score_king):>5} | "
            f"{str(vf):>{col_vis+2}} | "
            f"{sf:<{col_stat}} | "
            f"{str(score_famous):>5} | "
            f"{short}"
        )

    print()
    print("=" * len(header))
    print()
    print("LLM score heuristic key:  2=maintain/reactivate  1=decay  0=decay")
    print("Status key:  asserted=created this sent  explicit=re-mentioned  reactivated=was 0, now 2")
    print("             decaying=vis1  inactive=vis0 or vis2-not-mentioned")


if __name__ == "__main__":
    simulate()
