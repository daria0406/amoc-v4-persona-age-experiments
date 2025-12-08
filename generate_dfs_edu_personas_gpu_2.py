import argparse
import json
import multiprocessing
import os
import re
from typing import List, Dict
import pandas as pd
from datasets import load_dataset
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

tokenizer = None  # will init later
OUTPUT_FOLDER = "/export/home/acs/stud/a/ana_daria.zahaleanu/amoc_output/personas_dfs"    
# -------------------------------------------------------------------
# Multiprocessing & environment setup
# -------------------------------------------------------------------
multiprocessing.set_start_method("spawn", force=True)
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["HF_HOME"] = "/export/projects/nlp/.cache"

PERSONAHUB_CONFIGS = [
    "persona",
    "instruction",
    "knowledge",
    "math",
    "npc",
    "reasoning",
    "tool",
]

JUDGE_SYSTEM_PROMPT_PRIMARY = """
You are a strict classifier.

Task: Decide whether the persona describes a primary school student.

Definitions:
- Primary school = preschool, kindergarten, or elementary/primary education, typically up to around ages 11–12.
- The key requirement is that the PERSONA is a child who is currently attending primary/elementary school (or an equivalent early school level).
- Exclude:
  - Middle school or secondary school students.
  - High school students.
  - University/college students.
  - Teachers, parents, or other adults talking about primary school children.
  - Adults describing their past in primary school.
- If the text does not give enough information to decide, label it as "uncertain".

Decision rules:
- If the persona clearly describes a child currently attending primary/elementary school (or preschool/kindergarten), label "yes".
- If the persona clearly belongs to a different educational level (secondary, high school, university, adult), label "no".
- If the age or educational level is unclear or ambiguous, label "uncertain".

Output format (JSON only):
{
  "label": "yes" | "no" | "uncertain",
  "confidence": <integer 0-100>,
  "reason": "<short explanation>"
}

Return ONLY this JSON object, with no additional text.
"""

JUDGE_SYSTEM_PROMPT_SECONDARY = """
You are a strict classifier.

Task: Decide whether the persona describes a secondary school student (middle school / lower secondary).

Definitions:
- Secondary school (for this task) = middle school, junior high, or lower secondary education that comes after primary/elementary school and before high school.
- Typical age range is around 11–14, but age alone is not sufficient; the persona must be clearly positioned in this middle / lower secondary stage.
- The key requirement is that the PERSONA is currently attending a secondary school level that is after primary but not yet high school.

Exclude:
- Primary / elementary / preschool children.
- High school students.
- University / college students.
- Teachers, parents, or other adults talking about secondary school children.
- Adults describing their past in secondary school.

If the text does not give enough information to decide, label it as "uncertain".

Decision rules:
- If the persona clearly describes a current middle school / junior high / lower secondary school student, label "yes".
- If the persona clearly belongs to a different educational level (primary, high school, university, adult), label "no".
- If the age or educational level is unclear or ambiguous, label "uncertain".

Output format (JSON only):
{
  "label": "yes" | "no" | "uncertain",
  "confidence": <integer 0-100>,
  "reason": "<short explanation>"
}

Return ONLY this JSON object, with no additional text.
"""

JUDGE_SYSTEM_PROMPT_HIGHSCHOOL = """
You are a strict classifier.

Task: Decide whether the persona describes a high school student.

Definitions:
- High school = upper secondary education before university/college, typically around ages 14–18.
- The key requirement is that the PERSONA is currently attending high school (or an equivalent upper secondary level), such as "high school", "upper secondary", "gymnasium" (in some systems), etc.

Exclude:
- Primary / elementary / middle school / lower secondary students.
- University / college students.
- Teachers, parents, or other adults talking about high school students.
- Adults describing their past in high school.
- Vocational or professional training that is clearly post-secondary (unless it is explicitly described as a high school program).

If the text does not give enough information to decide, label it as "uncertain".

Decision rules:
- If the persona clearly describes a current high school / upper secondary student, label "yes".
- If the persona clearly belongs to a different educational level (primary, middle school, university, adult), label "no".
- If the age or educational level is unclear or ambiguous, label "uncertain".

Output format (JSON only):
{
  "label": "yes" | "no" | "uncertain",
  "confidence": <integer 0-100>,
  "reason": "<short explanation>"
}

Return ONLY this JSON object, with no additional text.
"""

JUDGE_SYSTEM_PROMPT_UNI = """
You are a strict classifier.

Task: Decide whether the persona describes a university freshman (first-year university or college student).

Definitions:
- University freshman = a person who is currently in their first year of a university, college, or equivalent higher-education program (including community college, polytechnic, etc.).
- They may be described as "freshman", "first-year student", "first year at university/college", or similar.
- The key requirement is that the PERSONA is currently in their first year of a higher-education degree program.

Exclude:
- High school students.
- Middle school or primary/elementary students.
- University students who are not in their first year (e.g., second-year, third-year, senior, graduate student, master's student, PhD student).
- Teachers, professors, or other staff.
- Adults only mentioning that they attended university in the past (not currently enrolled).

If the text does not give enough information to decide, label it as "uncertain".

Decision rules:
- If the persona clearly describes a current first-year university/college student, label "yes".
- If the persona clearly belongs to a different educational level (high school, later-year university, graduate school, adult not in school), label "no".
- If the educational level or year of study is unclear or ambiguous, label "uncertain".

Output format (JSON only):
{
  "label": "yes" | "no" | "uncertain",
  "confidence": <integer 0-100>,
  "reason": "<short explanation>"
}

Return ONLY this JSON object, with no additional text.
"""

# -------------------------------------------------------------------
# Keyword lists for quick filtering
# -------------------------------------------------------------------
PRIMARY_EXCLUDE_KEYWORDS = [
    "teacher", "professor", "lecturer", "instructor",
    "parent", "mother", "father", "mom", "dad",
    "doctoral", "phd", "postdoc"
]

SECONDARY_EXCLUDE_KEYWORDS = [
    "professor", "lecturer", "postdoc", "phd",
    "kindergarten teacher", "preschool teacher"
]

HIGH_EXCLUDE_KEYWORDS = [
    "professor", "lecturer", "postdoc", "phd",
    "kindergarten teacher", "preschool teacher"
]

UNI_EXCLUDE_KEYWORDS = [
    "high school teacher", "middle school teacher",
    "elementary teacher", "primary teacher",
    "principal", "headmaster"
]

# These are used by is_relevant()
YOUNG_EDU_KEYWORDS = [
    # School Levels
    "primary school",
    "elementary school",
    "kindergarten",
    "preschool",
    "nursery school",
    "grade school",
    "middle school",
    "junior high",
    # Grades / Years (Specific formats)
    "1st grade",
    "2nd grade",
    "3rd grade",
    "4th grade",
    "5th grade",
    "grade 1",
    "grade 2",
    "grade 3",
    "grade 4",
    "grade 5",
    "year 1",
    "year 2",
    "year 3",
    "year 4",
    "year 5",
    # Roles & Activities
    "pupil",
    "young student",
    "young learner",
    "schoolboy",
    "schoolgirl",
    "kindergartener",
    "preschooler",
    "pre-schooler",
    "kid",
    "child",
    "learning to read",
    "learning to write",
    "alphabet",
    "reading",
    "storytime",
    "story time",
]

SECONDARY_EDU_KEYWORDS = [
    "secondary school",
    "middle school",
    "preparatory school",
    "college prep",
    "academy",
    "gymnasium",
    "lyceum",
    "comprehensive school",
    # Grades / Years
    "6th grade",
    "7th grade",
    "8th grade",
    "grade 6",
    "grade 7",
    "grade 8",
    "year 6",
    "year 7",
    "year 8",
    # Roles & Identity
    "teen student",
    "teenage student",
    "middle schooler",
    "adolescent student",
    "student council",
    "varsity team",
]

HIGH_SCHOOL_KEYWORDS = [
    # Institutions
    "high school",
    "senior high",
    "college prep",
    "preparatory school",
    "boarding school",
    "academy",
    # Grades / Years (US & International)
    "9th grade",
    "10th grade",
    "11th grade",
    "12th grade",
    "grade 9",
    "grade 10",
    "grade 11",
    "grade 12",
    "year 9",
    "year 10",
    "year 11",
    "year 12",
    "freshman high",
    "sophomore high",
    "junior high",
    "senior high",
    "high school freshman",
    "high school sophomore",
    "high school junior",
    "high school senior",
    # Roles & Identity
    "high school student",
    "high schooler",
    "teen student",
    "teenage student",
    "student council",
    "varsity",
    "yearbook club",
    "debate team",
]

# 1. UNIVERSITY KEYWORDS (18 + - including 18)
# =========================================================
UNIVERSITY_KEYWORDS = [
    # Institutions
    "university",
    "college",
    "campus",
    "medical school",
    "law school",
    "business school",
    "community college",
    "liberal arts college",
    "polytechnic",
    # Levels / Degrees
    "undergrad",
    "undergraduate",
    "bachelor's",
    # Years / Status
    "freshman",
    "sophomore",
    "junior",
    "senior",
    "first year",
    "second year",
    "third year",
    "final year",
    "major in",
    "minoring in",
    "studying for a degree",
    "thesis",
    "capstone",
    # Identity
    "college student",
    "university student",
    "med student",
    "law student",
    "engineering student",
    "art student",
]

# -------------------------------------------------------------------
# Simple exclusion helpers
# -------------------------------------------------------------------
def should_exclude_primary(text: str) -> bool:
    t = text.lower()
    return any(kw in t for kw in PRIMARY_EXCLUDE_KEYWORDS)


def should_exclude_secondary(text: str) -> bool:
    t = text.lower()
    return any(kw in t for kw in SECONDARY_EXCLUDE_KEYWORDS)


def should_exclude_highschool(text: str) -> bool:
    t = text.lower()
    return any(kw in t for kw in HIGH_EXCLUDE_KEYWORDS)


def should_exclude_university(text: str) -> bool:
    t = text.lower()
    return any(kw in t for kw in UNI_EXCLUDE_KEYWORDS)


# -------------------------------------------------------------------
# V2 loaders (LLM-centric)
# -------------------------------------------------------------------
def loading_filtering_young_learners(min_confidence: int = 80) -> pd.DataFrame:
    """
    V2: Minimal rule-based filtering (domain + simple keyword excludes).
    LLM decides if persona is a PRIMARY / young learner.

    Returns df with:
      idx, source_config, persona_text, age,
      llm_label, llm_confidence, llm_reason
    """
    print("V2: Loading and Filtering for PRESCHOOL / PRIMARY (LLM-centric)...")
    all_rows = []

    # Step 1: domain match
    for config_name in PERSONAHUB_CONFIGS:
        try:
            ds = load_dataset(
                "proj-persona/PersonaHub", name=config_name, split="train"
            )
            for i in range(len(ds)):
                rec = ds[i]
                if is_relevant(rec, "young"):
                    all_rows.append(
                        {
                            "idx": i,
                            "persona_text": preferred_persona_text(rec),
                            "source_config": config_name,
                        }
                    )
        except Exception:
            pass

    df = pd.DataFrame(all_rows)
    if df.empty:
        print("V2: No young learner candidates found.")
        return df

    # Step 2: normalize text + extract age for info
    df["persona_text"] = df["persona_text"].astype(str)
    df["age"] = df["persona_text"].apply(extract_age)

    # Step 3: minimalist keyword exclude
    df["is_excluded"] = df["persona_text"].apply(should_exclude_primary)
    df = df[~df["is_excluded"]].copy()

    print(f"V2: Candidates after keyword exclude: {len(df)}")

    if df.empty:
        return df

    # Step 4: LLM judge
    print("V2: Running LLM judge for PRIMARY classification...")
    llm_labels, llm_conf, llm_reasons = [], [], []

    for text in df["persona_text"]:
        result = judge_persona(text, JUDGE_SYSTEM_PROMPT_PRIMARY)
        llm_labels.append(result.get("label"))
        llm_conf.append(result.get("confidence"))
        llm_reasons.append(result.get("reason"))

    df["llm_label"] = llm_labels
    df["llm_confidence"] = llm_conf
    df["llm_reason"] = llm_reasons

    # Step 5: keep only LLM-approved
    df_final = df[
        (df["llm_label"] == "yes")
        & (df["llm_confidence"].astype(float) >= min_confidence)
    ].copy()

    print(f"V2: Final PRIMARY personas (LLM-approved): {len(df_final)}")
    return df_final


def loading_filtering_secondary_students(min_confidence: int = 80) -> pd.DataFrame:
    """
    V2: Minimal rule-based filtering for SECONDARY (middle/high) students.
    Age is extracted but not used to gate; LLM decides category.

    Returns df with persona_text, age, LLM fields.
    """
    print("V2: Loading and Filtering for SECONDARY SCHOOL STUDENTS...")
    all_rows = []

    # Step 1: domain match
    for config_name in PERSONAHUB_CONFIGS:
        try:
            ds = load_dataset(
                "proj-persona/PersonaHub", name=config_name, split="train"
            )
            for i in range(len(ds)):
                rec = ds[i]
                if is_relevant(rec, "secondary"):
                    all_rows.append(
                        {
                            "idx": i,
                            "persona_text": preferred_persona_text(rec),
                            "source_config": config_name,
                        }
                    )
        except Exception:
            pass

    df = pd.DataFrame(all_rows)
    if df.empty:
        print("V2: No secondary candidates found.")
        return df

    print(f"V2: Initial secondary domain matches: {len(df)}")

    # Step 2: normalize + age for info
    df["persona_text"] = df["persona_text"].astype(str)
    df["age"] = df["persona_text"].apply(extract_age)

    # Step 3: minimalist keyword exclude
    df["is_excluded"] = df["persona_text"].apply(should_exclude_secondary)
    df = df[~df["is_excluded"]].copy()

    print(f"V2: Secondary candidates after keyword exclude: {len(df)}")

    if df.empty:
        return df

    # Step 4: LLM judge
    print("V2: Running LLM judge for SECONDARY classification...")
    llm_labels, llm_conf, llm_reasons = [], [], []

    for text in df["persona_text"]:
        result = judge_persona(text, JUDGE_SYSTEM_PROMPT_SECONDARY)
        llm_labels.append(result.get("label"))
        llm_conf.append(result.get("confidence"))
        llm_reasons.append(result.get("reason"))

    df["llm_label"] = llm_labels
    df["llm_confidence"] = llm_conf
    df["llm_reason"] = llm_reasons

    # Step 5: keep LLM-approved
    df_final = df[
        (df["llm_label"] == "yes")
        & (df["llm_confidence"].astype(float) >= min_confidence)
    ].copy()

    print(f"V2: Final SECONDARY personas (LLM-approved): {len(df_final)}")

    return df_final


def loading_filtering_university_students(min_confidence: int = 80) -> pd.DataFrame:
    """
    V2: Minimal rule-based filtering for UNIVERSITY FRESHMEN.
    Only domain + keyword excludes; LLM decides if persona is a first-year uni student.

    Returns df with persona_text, age, LLM fields.
    """
    print("V2: Loading and Filtering for UNIVERSITY STUDENTS...")
    all_rows = []

    # Step 1: domain match
    for config_name in PERSONAHUB_CONFIGS:
        try:
            ds = load_dataset(
                "proj-persona/PersonaHub", name=config_name, split="train"
            )
            for i in range(len(ds)):
                rec = ds[i]
                if is_relevant(rec, "university"):
                    all_rows.append(
                        {
                            "idx": i,
                            "persona_text": preferred_persona_text(rec),
                            "source_config": config_name,
                        }
                    )
        except Exception:
            pass

    df = pd.DataFrame(all_rows)
    if df.empty:
        print("V2: No university candidates found.")
        return df

    print(f"V2: Initial university domain matches: {len(df)}")

    # Step 2: normalize + age for info
    df["persona_text"] = df["persona_text"].astype(str)
    df["age"] = df["persona_text"].apply(extract_age)

    # Step 3: minimalist keyword exclude
    df["is_excluded"] = df["persona_text"].apply(should_exclude_university)
    df = df[~df["is_excluded"]].copy()

    print(f"V2: University candidates after keyword exclude: {len(df)}")

    if df.empty:
        return df

    # Step 4: LLM judge
    print("V2: Running LLM judge for UNIVERSITY FRESHMAN classification...")
    llm_labels, llm_conf, llm_reasons = [], [], []

    for text in df["persona_text"]:
        result = judge_persona(text, JUDGE_SYSTEM_PROMPT_UNI)
        llm_labels.append(result.get("label"))
        llm_conf.append(result.get("confidence"))
        llm_reasons.append(result.get("reason"))

    df["llm_label"] = llm_labels
    df["llm_confidence"] = llm_conf
    df["llm_reason"] = llm_reasons

    # Step 5: keep LLM-approved
    df_final = df[
        (df["llm_label"] == "yes")
        & (df["llm_confidence"].astype(float) >= min_confidence)
    ].copy()

    print(f"V2: Final UNIVERSITY FRESHMEN personas (LLM-approved): {len(df_final)}")

    return df_final


def loading_filtering_highschool_students(min_confidence: int = 80) -> pd.DataFrame:
    """
    V2: Minimal rule-based filtering for HIGH SCHOOL STUDENTS.

    - Uses domain filter (is_relevant(..., "highschool"))
    - Applies a very light keyword-based exclusion
    - Does NOT filter by age (only extracts it for analysis)
    - Asks the LLM (JUDGE_SYSTEM_PROMPT_HIGHSCHOOL) to decide if the persona
      is a high school student.
    
    Returns a dataframe with:
        idx, source_config, persona_text, age,
        is_excluded, llm_label, llm_confidence, llm_reason
    """
    print("V2: Loading and Filtering for HIGH SCHOOL STUDENTS...")
    all_rows = []

    # -------- STEP 1: DOMAIN FILTER --------
    for config_name in PERSONAHUB_CONFIGS:
        try:
            ds = load_dataset(
                "proj-persona/PersonaHub", name=config_name, split="train"
            )
            for i in range(len(ds)):
                rec = ds[i]
                # you may also want to treat 'secondary' as high school depending on configs
                if is_relevant(rec, "highschool"):
                    all_rows.append(
                        {
                            "idx": i,
                            "persona_text": preferred_persona_text(rec),
                            "source_config": config_name,
                        }
                    )
        except Exception:
            pass

    df = pd.DataFrame(all_rows)
    if df.empty:
        print("V2: No high school candidates found.")
        return df

    print(f"V2: Initial high school domain matches: {len(df)}")

    # -------- STEP 2: NORMALIZE TEXT + EXTRACT AGE (for info only) --------
    df["persona_text"] = df["persona_text"].astype(str)
    df["age"] = df["persona_text"].apply(extract_age)

    # -------- STEP 3: MINIMAL KEYWORD-BASED EXCLUDE --------
    df["is_excluded"] = df["persona_text"].apply(should_exclude_highschool)
    df = df[~df["is_excluded"]].copy()

    print(f"V2: High school candidates after keyword exclude: {len(df)}")

    if df.empty:
        return df

    # -------- STEP 4: LLM JUDGE FOR HIGH SCHOOL --------
    print("V2: Running LLM judge for HIGH SCHOOL classification...")
    llm_labels, llm_conf, llm_reasons = [], [], []

    for text in df["persona_text"]:
        result = judge_persona(text, JUDGE_SYSTEM_PROMPT_HIGHSCHOOL)
        llm_labels.append(result.get("label"))
        llm_conf.append(result.get("confidence"))
        llm_reasons.append(result.get("reason"))

    df["llm_label"] = llm_labels
    df["llm_confidence"] = llm_conf
    df["llm_reason"] = llm_reasons

    # -------- STEP 5: KEEP LLM-APPROVED PERSONAS --------
    df_final = df[
        (df["llm_label"] == "yes")
        & (df["llm_confidence"].astype(float) >= min_confidence)
    ].copy()

    print(f"V2: Final HIGH SCHOOL personas (LLM-approved): {len(df_final)}")

    return df_final


# -------------------------------------------------------------------
# Age extraction
# -------------------------------------------------------------------
AGE_REGEX = re.compile(
    r"(\d{1,2})\s*-*\s*(?:year[s]?\s*[-]?\s*old|y/o|yr[s]?)",
    re.IGNORECASE,
)


# -------------------------------------------------------------------
# Global LLM + sampling_params will be initialized in init_llm()
# -------------------------------------------------------------------
llm = None
sampling_params = None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Filter PersonaHub for educational personas using gpt-oss."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="openai/gpt-oss-120b",
        help="Model name or path for vLLM (default: openai/gpt-oss-120b).",
    )
    parser.add_argument(
        "--file",
        type=str,
        default="high-school.csv",
        help=(
            "Output filename. Category is inferred from its name.\n"
            "Must contain one of: primary, secondary, highschool, university."
        ),
    )
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=4,
        help="Tensor parallel size for vLLM (default: 4).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for LLM judging (default: 32).",
    )
    parser.add_argument(
        "--min_confidence",
        type=int,
        default=80,
        help="Min confidence (%) to accept 'yes' judgments (default: 80).",
    )
    return parser.parse_args()


# for LLaMA-style models – add tokenizer because we should use a chat template
def init_llm(model_name: str, tensor_parallel_size: int):
    global llm, sampling_params, tokenizer

    print(f"Loading model: {model_name}")
    print(f"Tensor parallel size: {tensor_parallel_size}\n")

    llm = LLM(
        model=model_name,
        tensor_parallel_size=tensor_parallel_size,
    )

    sampling_params = SamplingParams(
        max_tokens=64,
        temperature=0.0,
        top_p=1.0,
        top_k=-1,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)


# -------------------------------------------------------------------
# Utility functions
# -------------------------------------------------------------------
def text_fields(record):
    fields = []
    for k, v in record.items():
        if isinstance(v, str):
            fields.append(v)
        elif isinstance(v, list):
            fields.extend([str(x) for x in v if isinstance(x, (str, int, float))])
        elif isinstance(v, dict):
            fields.extend([f"{kk}: {vv}" for kk, vv in v.items()])
    return fields


def is_relevant(rec, keywords):
    """
    Filter: Checks if the persona mentions education terms per group.
    """
    blob = " ".join(text_fields(rec)).lower()
    if keywords == "young":
        return any(k in blob for k in YOUNG_EDU_KEYWORDS)
    elif keywords == "secondary":
        return any(k in blob for k in SECONDARY_EDU_KEYWORDS)
    elif keywords in ("high_school", "highschool"):
        return any(k in blob for k in HIGH_SCHOOL_KEYWORDS)
    elif keywords == "university":
        return any(k in blob for k in UNIVERSITY_KEYWORDS)
    return False


def preferred_persona_text(rec):
    for cand in [
        "persona_text",
        "persona",
        "description",
        "text",
        "profile",
        "traits",
        "input persona",
    ]:
        if cand in rec and isinstance(rec[cand], str) and rec[cand].strip():
            return rec[cand].strip()
    return " ".join(text_fields(rec))[:1500]


def extract_age(text):
    if not isinstance(text, str):
        return None

    text_low = text.lower()

    match = AGE_REGEX.search(text_low)
    if match:
        try:
            return int(match.group(1))
        except ValueError:
            return None

    # Heuristic proxies
    if "preschool" in text_low:
        return 3
    if "kindergarten" in text_low:
        return 5
    if "primary school" in text_low:
        return 8
    if "elementary" in text_low:
        return 8
    if "middle school" in text_low:
        return 12
    if "high school" in text_low:
        return 16
    if "university" in text_low and "student" in text_low:
        return 19
    if "first grade" in text_low or "1st grade" in text_low:
        return 6
    if "second grade" in text_low or "2nd grade" in text_low:
        return 7
    if "third grade" in text_low or "3rd grade" in text_low:
        return 8
    if "fourth grade" in text_low or "4th grade" in text_low:
        return 9
    if "fifth grade" in text_low or "5th grade" in text_low:
        return 10
    if "sixth grade" in text_low or "6th grade" in text_low:
        return 11
    if "seventh grade" in text_low or "7th grade" in text_low:
        return 12
    if "eighth grade" in text_low or "8th grade" in text_low:
        return 13
    if (
        "ninth grade" in text_low
        or "9th grade" in text_low
        or "high school freshman" in text_low
    ):
        return 14
    if "sophomore" in text_low and "high" in text_low:
        return 15
    if "junior" in text_low and "high" in text_low:
        return 16
    if "senior" in text_low and "high" in text_low:
        return 17
    if "freshman" in text_low and "university" in text_low:
        return 18
    if "freshman" in text_low and "college" in text_low:
        return 18
    if "first year" in text_low and "college" in text_low:
        return 18
    if "first year" in text_low and "undergraduate" in text_low:
        return 18
    if "first year" in text_low and "undergrad" in text_low:
        return 18
    if "senior" in text_low and "university" in text_low:
        return 22

    return None


# -------------------------------------------------------------------
# LLM judging helpers
# -------------------------------------------------------------------
def judge_persona(persona: str, system_prompt: str) -> Dict:
    """
    Classify a SINGLE persona using the provided system_prompt.
    Returns a dict: {label, confidence, reason}.
    """
    if llm is None or sampling_params is None or tokenizer is None:
        raise RuntimeError("LLM/tokenizer not initialized. Call init_llm() first.")

    messages = [
        {"role": "system", "content": system_prompt.strip()},
        {
            "role": "user",
            "content": f"Persona:\n{persona}\n\nAnswer ONLY in JSON.",
        },
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    outputs = llm.generate([prompt], sampling_params)
    text = outputs[0].outputs[0].text.strip()

    try:
        start = text.find("{")
        end = text.rfind("}") + 1
        data = json.loads(text[start:end])
    except Exception:
        data = {
            "label": "uncertain",
            "confidence": 0,
            "reason": f"Could not parse JSON: {text[:120]}",
        }
    return data


def judge_batch_120b(personas: List[str]) -> List[Dict]:
    """
    (Unused in current pipeline, but left as utility)
    Use the global llm + sampling_params to classify a batch of personas
    with some default system prompt (e.g., JUDGE_SYSTEM_PROMPT_HIGHSCHOOL).
    """
    if llm is None or sampling_params is None:
        raise RuntimeError("LLM not initialized. Call init_llm() first.")

    # This assumes you have a global JUDGE_SYSTEM_PROMPT defined;
    # or you can adapt it similarly to judge_persona with a parameter.
    messages_list = [
        [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT_HIGHSCHOOL.strip()},
            {
                "role": "user",
                "content": f"Persona:\n{p}\n\nAnswer ONLY in JSON.",
            },
        ]
        for p in personas
    ]

    prompts = [
        tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )
        for msgs in messages_list
    ]

    outputs = llm.generate(prompts, sampling_params)

    results = []
    # Debug print first few
    for out in outputs[:10]:
        text = out.outputs[0].text.strip()
        print("RAW LLM OUTPUT EXAMPLE:")
        print(text)

    for out in outputs:
        text = out.outputs[0].text.strip()
        try:
            start = text.find("{")
            end = text.rfind("}") + 1
            data = json.loads(text[start:end])
        except Exception:
            data = {
                "label": "uncertain",
                "confidence": 0,
                "reason": f"Could not parse JSON: {text[:120]}",
            }
        results.append(data)

    return results


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------
def main():
    args = parse_args()
    out_file = args.file
    min_conf = args.min_confidence

    # Initialize LLM once
    init_llm(args.model, args.tensor_parallel_size)

    # Extract category from filename
    filename = os.path.basename(out_file).lower()

    # Determine category
    if "primary" in filename:
        print("Detected CATEGORY = PRIMARY from filename.")
        df = loading_filtering_young_learners(min_confidence=min_conf)

    elif "secondary" in filename:
        print("Detected CATEGORY = SECONDARY from filename.")
        df = loading_filtering_secondary_students(min_confidence=min_conf)

    elif (
        "highschool" in filename
        or "high_school" in filename
        or "high-school" in filename
    ):
        print("Detected CATEGORY = HIGH SCHOOL from filename.")
        df = loading_filtering_highschool_students(min_confidence=min_conf)

    elif "university" in filename or "uni" in filename:
        print("Detected CATEGORY = UNIVERSITY FRESHMEN from filename.")
        df = loading_filtering_university_students(min_confidence=min_conf)

    else:
        raise ValueError(
            "Filename must contain one of: primary, secondary, highschool, university"
        )

    # Save dataframe
    if df is not None and not df.empty:
        out_path = os.path.join(OUTPUT_FOLDER, f"{out_file}_llm_judge.csv")
        df.to_csv(out_file, index=False)
        print(f"\nSaved {len(df)} personas to: {out_file}")
    else:
        print("\nNo personas found for this category; nothing saved.")


if __name__ == "__main__":
    main()
