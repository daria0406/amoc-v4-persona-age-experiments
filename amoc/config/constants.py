MAX_DISTANCE_FROM_ACTIVE_NODES = 2
MAX_NEW_CONCEPTS = 3
MAX_NEW_PROPERTIES = 3
CONTEXT_LENGTH = 1
# edge_visibility = edge forget from old code
EDGE_VISIBILITY = 2
NR_RELEVANT_EDGES = 10
DEBUG = False
DECAY_STEP = 1
# Faster decay step applied to low-relevance (semantic score 1) edges whose
# object is re-mentioned. Raise to make transient edges drop out sooner.
SEMANTIC_FAST_DECAY_STEP = 2
# In the activation matrix, an edge only propagates activation to its neighbour
# while visibility_score > this threshold. Raise it to make carried-over tokens
# fade from the matrix faster (fewer sentences of lingering activation).
MATRIX_MIN_PROPAGATION_VISIBILITY = 1
MAX_REACTIVATION_COUNT = 6
MAX_EDGES_PER_NODE = 5
MAX_CARRYOVER = 10
MAX_TRIPLETS = 25
REACTIVATION_VISIBILITY = 2
MAX_CARRYOVER_NODES = 5          
CARRYOVER_SENIORITY_WEIGHT = 1.0

STORY_TEXT = "A young knight rode through the forest. The knight was unfamiliar with the country. Suddenly, a dragon appeared. The dragon was kidnapping a beautiful princess. The knight wanted to free the princess. The knight wanted to marry the princess. The knight hurried after the dragon. The knight and the dragon fought for life and death. Soon, the knight's armor was completely scorched. At last, the knight killed the dragon. The knight freed the princess. The princess was very thankful to the knight. The princess married the knight."

AGE_REGIMES = {
    "primary_school": (5, 10),
    "secondary_school": (11, 14),
    "high_school": (15, 18),
    "university_freshman": (17, 18),
}

BLUE_NODES = [
    "knight",
    "princess",
    "dragon",
    "forest",
    "armor",
    "beautiful",
    "scorched",
]
