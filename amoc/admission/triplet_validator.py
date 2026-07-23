from typing import Dict, List, Optional, Tuple, Any
import logging
from difflib import SequenceMatcher
import numpy as np

from amoc.extraction.linguistic import LinguisticProcessing
from amoc.admission.text_normalizer import TextNormalizer
from amoc.llm.vllm_client import VLLMClient
from amoc.core.edge import _maybe_embed
from amoc.admission.triplet_deduplicator import TripletDeduplicator


class TripletValidator:
    COMMON_COPULAS = frozenset(
        {"is", "am", "are", "was", "were", "be", "been", "being"}
    )
    
    VAGUE_PROTOTYPES = [
        "related to", "associated with", "connected to", 
        "involved in", "concerned with", "pertaining to",
        "refers to", "regarding", "with respect to",
        "has", "have", "having", "belongs to", "part of"
    ]
    
    NEGATION_PHRASES = [
        "not related", "no connection", "not available", "not applicable",
        "not connected", "no relation", "no link", "not involved",
        "not associated", "without connection", "without relation",
        "unconnected", "unrelated", "disconnected", "disassociated",
        "nonapplicable", "nonexistent", "unavailable", "uninvolved",
        "not", "no", "never", "neither", "nor", "without"
    ]

    FORCE_ALLOW_CONCEPTS = frozenset({"horse", "sword", "fire"})

    def __init__(
        self,
        linguistic_ops: LinguisticProcessing,
        extract_deterministic_fn,
        text_normalizer: TextNormalizer,
        client: VLLMClient,
        persona: str = "",
        similarity_threshold: float = 0.8,
        spacy_nlp=None,
        vague_threshold: float = 0.75,
        reject_vague: bool = True,
        reject_negations: bool = True,
    ):
        self.linguistic_ops = linguistic_ops
        self.extract_deterministic_fn = extract_deterministic_fn
        self.text_normalizer = text_normalizer
        self.client = client
        self.persona = persona
        self.similarity_threshold = similarity_threshold
        self.vague_threshold = vague_threshold
        self.reject_vague = reject_vague
        self.reject_negations = reject_negations
        self._embedding_cache = {}
        self.spacy_nlp = spacy_nlp or (
            linguistic_ops._spacy_nlp if linguistic_ops else None
        )
        self.deduplicator = TripletDeduplicator(self.spacy_nlp)
        
        self._vague_prototype_embeddings = []
        for proto in self.VAGUE_PROTOTYPES:
            emb = _maybe_embed(proto)
            if emb is not None:
                self._vague_prototype_embeddings.append((proto, emb))

    def deduplicate_triplets(
        self, triplets: List[Tuple[str, str, str]]
    ) -> List[Tuple[str, str, str]]:
        return self.deduplicator.deduplicate(triplets)

    def prioritize_hub(
        self,
        triplets: List[Tuple[str, str, str]],
        explicit_nodes: List[str],
    ) -> List[Tuple[str, str, str]]:
        if not triplets or not explicit_nodes:
            return triplets

        specific_nodes = [n for n in explicit_nodes]

        if not specific_nodes:
            return triplets

        hub_candidate = specific_nodes[0]

        hub_as_subject = [t for t in triplets if t[0] == hub_candidate]
        other_triples = [t for t in triplets if t[0] != hub_candidate]

        return hub_as_subject + other_triples

    def normalize_llm_triplets(
        self, new_relationships: List[Any]
    ) -> List[Tuple[str, str, str]]:
        normalized = []
        for rel in new_relationships or []:
            triple = self.normalize_single_triplet(rel)
            if triple:
                normalized.append(triple)
        return normalized

    def normalize_single_triplet(
        self, relationship: Any
    ) -> Optional[Tuple[str, str, str]]:
        if not relationship:
            return None

        if isinstance(relationship, dict):
            subj = relationship.get("subject") or relationship.get("head")
            rel = relationship.get("relation") or relationship.get("predicate")
            obj = relationship.get("object") or relationship.get("tail")
            if subj and rel and obj:
                return (str(subj), str(rel), str(obj))
            return None

        if isinstance(relationship, (list, tuple)):
            if len(relationship) == 3:
                return tuple(relationship)
            if len(relationship) == 4:
                subj, rel, _, obj = relationship
                return (subj, rel, obj)

        return None

    def validate_triplet_early(
        self, subj: str, rel: str, obj: str
    ) -> Optional[Dict]:
        if not subj or not rel or not obj:
            return {
                "valid": False,
                "reason": "Empty subject, relation, or object",
                "corrected_triple": None,
                "action": "reject_empty"
            }
        
        if self.reject_negations and self.is_negation_relation(rel):
            return {
                "valid": False,
                "reason": f"Negation relation '{rel}' rejected - adds no semantic value",
                "corrected_triple": None,
                "action": "reject_negation"
            }
        
        if self.reject_vague and self.is_vague_relation(rel):
            return {
                "valid": False,
                "reason": f"Vague relation '{rel}' rejected - be more specific",
                "corrected_triple": None,
                "action": "reject_vague"
            }
        
        if subj.lower().strip() == obj.lower().strip():
            return {
                "valid": False,
                "reason": f"Self-loop: '{subj}' relates to itself",
                "corrected_triple": None,
                "action": "reject_self_loop"
            }
        
        if len(rel) < 2 or len(subj) < 1 or len(obj) < 1:
            return {
                "valid": False,
                "reason": "Too short subject, relation, or object",
                "corrected_triple": None,
                "action": "reject_too_short"
            }
        
        return None

    def is_vague_relation(self, label: str) -> bool:
        if not label:
            return True
        
        normalized = label.lower().strip()
        
        exact_vague = ["relates to", "related to", "associated with", 
                      "connected to", "involves", "concerns", "pertains to",
                      "refers to", "regarding", "has", "have", "belongs to"]
        if normalized in exact_vague:
            logging.debug(f"Vague relation (exact): '{label}'")
            return True
        
        vague_prefixes = ["relate", "associate", "connect", "involve", 
                         "concern", "pertain", "refer", "regard"]
        for prefix in vague_prefixes:
            if normalized.startswith(prefix) or prefix in normalized:
                logging.debug(f"Vague relation (prefix match): '{label}'")
                return True
        
        label_emb = self.get_label_embedding(label)
        if label_emb is not None:
            for proto, proto_emb in self._vague_prototype_embeddings:
                similarity = float(np.dot(label_emb, proto_emb))
                if similarity > self.vague_threshold:
                    logging.debug(
                        f"Vague relation (embedding): '{label}' similar to '{proto}' (sim={similarity:.3f})"
                    )
                    return True
        
        return False

    def get_label_embedding(self, label: str):
        if label not in self._embedding_cache:
            self._embedding_cache[label] = _maybe_embed(label)
        return self._embedding_cache[label]

    def is_negation_relation(self, label: str) -> bool:
        if not label:
            return False

        normalized = label.lower().replace("_", " ").strip()
        
        for phrase in self.NEGATION_PHRASES:
            if phrase in normalized:
                logging.debug(f"Negation detected: '{label}' contains '{phrase}'")
                return True

        negation_words = {"not", "no", "never", "neither", "nor", "without"}
        words = normalized.split()
        if any(word in negation_words for word in words):
            return True

        if self.spacy_nlp:
            doc = self.spacy_nlp(normalized)
            for token in doc:
                if token.dep_ == "neg":
                    return True
                if token.lower_ in {"not", "n't", "no", "never", "neither", "nor", "without"}:
                    return True

        return False

    def is_valid_relation_label(self, label: str) -> bool:
        if self.is_negation_relation(label):
            return False
        if self.is_vague_relation(label):
            return False
        return True

    def validate_with_llm(
        self, subj: str, rel: str, obj: str, sentence: str, story_context: str = ""
    ) -> Dict:
        early_result = self.validate_triplet_early(subj, rel, obj)
        if early_result is not None:
            return early_result
        
        llm_validation = self.client.validate_triplet(
            sentence=sentence,
            subject=subj,
            relation=rel,
            object=obj,
            persona=self.persona,
        )

        if not llm_validation.get("valid", False):
            logging.info(
                f"llm rejected triple ({subj},{rel},{obj}): {llm_validation.get('reason', '')}"
            )

        return llm_validation

    def labels_are_similar(self, label1: str, label2: str) -> bool:
        if not label1 or not label2:
            return False

        if label1.strip().lower() == label2.strip().lower():
            return True

        if (
            SequenceMatcher(None, label1.lower(), label2.lower()).ratio()
            >= self.similarity_threshold
        ):
            return True

        emb1 = self.get_label_embedding(label1)
        emb2 = self.get_label_embedding(label2)
        if emb1 is not None and emb2 is not None:
            cos = float(np.dot(emb1, emb2))
            if cos >= self.similarity_threshold:
                return True

        return False

    def normalize_endpoints(
        self, subj: str, obj: str
    ) -> Tuple[Optional[str], Optional[str]]:
        subj_norm = self.text_normalizer.extract_canonical_node_lemma(
            subj, is_subject=True
        )
        obj_norm = self.text_normalizer.extract_canonical_node_lemma(
            obj, is_subject=False
        )
        return subj_norm, obj_norm

    def clean_and_validate_relation(self, rel: str) -> Optional[str]:
        edge_label = self.text_normalizer.normalize_edge_label(
            rel.replace("(edge)", "").strip()
        )
        if not self.text_normalizer.is_valid_relation_label(edge_label):
            return None
        if not self.is_valid_relation_label(edge_label):
            return None
        return edge_label

    def validate_relation_is_verb(
        self, triplet: Tuple[str, str, str]
    ) -> Tuple[bool, Optional[str], Optional[Tuple[str, str, str]]]:
        subj, relation, obj = triplet

        if not relation or not isinstance(relation, str):
            return False, "empty or invalid relation", None

        relation_clean = relation.strip().lower()
        if not relation_clean:
            return False, "empty relation after cleaning", None

        if self.spacy_nlp is None:
            return True, None, None

        if relation_clean in self.COMMON_COPULAS:
            return True, None, None

        relation_for_parse = relation_clean.replace("_", " ")

        rel_doc = self.spacy_nlp(relation_for_parse)
        if not rel_doc or len(rel_doc) == 0:
            return False, f"could not parse relation '{relation}'", None

        has_verb = False

        for token in rel_doc:
            if token.pos_ in {"VERB", "AUX"}:
                has_verb = True
                break

            morph_str = str(token.morph)
            verb_features = ["Tense=", "VerbForm=", "Mood=", "Voice="]
            if any(f in morph_str for f in verb_features):
                has_verb = True
                break

            if token.dep_ in {"ROOT", "aux", "auxpass", "xcomp", "ccomp"}:
                has_verb = True
                break

            if len(rel_doc) == 1 and token.is_alpha:
                if hasattr(token, "prob") and token.prob < -3:
                    has_verb = True
                    break

        if not has_verb:
            words = relation_for_parse.split()
            if len(words) >= 1 and words[0] in self.COMMON_COPULAS:
                has_verb = True

        if not has_verb:
            pos_tags = [f"{t.text}({t.pos_})" for t in rel_doc]
            return (
                False,
                f"relation '{relation}' has no verb. pos: {', '.join(pos_tags)}",
                None,
            )

        return True, None, None

    def extract_verb_info(self, relation: str, rel_doc) -> Tuple[Optional[str], bool]:
        if not rel_doc:
            return None, False

        relation_lower = relation.lower().strip()

        words = relation_lower.split()
        if len(words) == 1 and words[0] in self.COMMON_COPULAS:
            return words[0], True

        for token in rel_doc:
            if token.pos_ in {"VERB", "AUX"}:
                return token.lemma_.lower(), True

        for token in rel_doc:
            morph_str = str(token.morph)
            if any(f in morph_str for f in ["Tense=", "VerbForm=", "Mood=", "Voice="]):
                return token.lemma_.lower(), True

        if len(words) >= 1 and words[0] in self.COMMON_COPULAS:
            return words[0], True

        return rel_doc[0].lemma_.lower(), False

    def check_has_adjective(
        self, rel_lemma: str, obj_pos: str, obj: str
    ) -> Optional[Dict]:
        if rel_lemma == "have" and obj_pos == "ADJ":
            return {
                "valid": False,
                "reason": f"'has' cannot take adjective '{obj}' as object — 'has' requires a noun",
                "corrected_triple": None,
                "action": "reject",
            }
        return None

    def is_event_or_action(self, doc):
        if not doc or len(doc) == 0:
            return False
        token = doc[0]
        if token.pos_ == "NOUN" and token.morph:
            morph_str = str(token.morph)
            if any(
                suffix in token.text.lower()
                for suffix in ["tion", "ing", "ment", "al", "ence", "ance", "ure"]
            ):
                return True
        if token.pos_ == "VERB":
            return True
        return False

    def check_circular_is(
        self, rel_lemma: str, obj_pos: str, subj: str, obj: str, subj_doc, obj_doc
    ) -> Optional[Dict]:
        if rel_lemma != "be" or obj_pos not in {"NOUN", "PROPN"}:
            return None

        obj_lower = obj.lower().strip()
        subj_lower = subj.lower().strip()

        if obj_lower == subj_lower:
            return {
                "valid": False,
                "reason": f"circular relation: '{subj} is {obj}' says nothing",
                "corrected_triple": None,
                "action": "reject",
            }

        subj_is_event = self.is_event_or_action(subj_doc)
        obj_is_event = self.is_event_or_action(obj_doc)

        if subj_is_event and obj_is_event:
            return {
                "valid": False,
                "reason": f"'{subj} is {obj}' is circular — both are events/actions",
                "corrected_triple": None,
                "action": "reject",
            }

        return None

    def is_copular_construction(self, relation: str, rel_doc=None) -> bool:
        COPULAR_LEMMAS = frozenset({"be", "become", "seem", "appear", "remain"})
        if rel_doc is None:
            rel_doc = self.spacy_nlp(relation) if relation else None

        if not rel_doc or len(rel_doc) == 0:
            return False

        relation_lower = relation.lower().strip()

        words = relation_lower.split()
        if len(words) == 1 and words[0] in self.COMMON_COPULAS:
            return True

        if len(words) >= 1 and words[0] in self.COMMON_COPULAS:
            return True

        root = None
        for token in rel_doc:
            if token.dep_ == "ROOT":
                root = token
                break

        if root:
            if root.lemma_.lower() in COPULAR_LEMMAS:
                return True

            for child in root.children:
                if child.dep_ in {"attr", "acomp", "pred"}:
                    return True

        has_subj = any(tok.dep_ in {"nsubj", "nsubjpass"} for tok in rel_doc)
        has_copula = any(tok.lemma_.lower() in {"be", "become"} for tok in rel_doc)
        has_complement = any(tok.dep_ in {"attr", "acomp", "pred"} for tok in rel_doc)

        if has_subj and has_copula and has_complement:
            return True

        return False

    def check_adjective_subject(
        self, subj: str, subj_pos: str, rel_has_verb: bool, relation: str, rel_doc
    ) -> Optional[Dict]:
        # Property/adjective nodes only ever describe concepts; they may be
        # the object of a relation but never the subject, regardless of what
        # kind of relation follows (verb, copula, or otherwise): "faithful
        # involves death" is just as invalid as "faithful is death".
        if subj_pos == "ADJ":
            return {
                "valid": False,
                "reason": f"adjective '{subj}' cannot be the subject of relation '{relation}'",
                "corrected_triple": None,
                "action": "reject",
            }
        return None

    def check_adjective_object(
        self,
        subj_pos: str,
        obj_pos: str,
        rel_has_verb: bool,
        relation: str,
        rel_doc,
        obj: str,
    ) -> Optional[Dict]:
        if subj_pos in {"NOUN", "PROPN", "PRON"} and obj_pos == "ADJ" and rel_has_verb:
            if self.is_copular_construction(relation, rel_doc):
                return None
            return {
                "valid": False,
                "reason": f"action verb '{relation}' cannot take adjective '{obj}' as object — needs a noun",
                "corrected_triple": None,
                "action": "reject",
            }
        return None

    def handle_missing_verb(
        self, rel_doc, obj_doc, subj: str, relation: str, obj: str
    ) -> Optional[Dict]:
        if obj_doc and any(t.pos_ in {"VERB", "AUX"} for t in obj_doc):
            return {
                "valid": False,
                "reason": f"relation '{relation}' has no verb, but object '{obj}' does",
                "corrected_triple": (subj, obj, relation),
                "action": "swap",
            }

        if rel_doc and len(rel_doc) == 1 and rel_doc[0].pos_ == "ADJ":
            return {
                "valid": False,
                "reason": f"relation '{relation}' is an adjective, not a verb",
                "corrected_triple": (subj, "is", relation),
                "action": "add_copula",
            }

        return None

    def validate_triplet_relation(
        self, triplet: Tuple[str, str, str]
    ) -> Dict[str, Any]:
        subj, relation, obj = triplet

        # FORCE‑ALLOW: if subject or object is in the whitelist
        if (subj.lower() in self.FORCE_ALLOW_CONCEPTS or 
            obj.lower() in self.FORCE_ALLOW_CONCEPTS):
            # Still reject self‑loops
            if subj.lower().strip() == obj.lower().strip():
                return {
                    "valid": False,
                    "reason": f"Self-loop on forced token '{subj}' – rejected",
                    "corrected_triple": None,
                    "action": "reject_self_loop"
                }
            # Still reject negations
            if self.is_negation_relation(relation):
                return {
                    "valid": False,
                    "reason": f"Negation relation on forced token – rejected",
                    "corrected_triple": None,
                    "action": "reject_negation"
                }
            # Accept all other forced‑token triplets
            return {
                "valid": True,
                "reason": f"Forced acceptance because token in {self.FORCE_ALLOW_CONCEPTS}",
                "corrected_triple": None,
                "action": "accept_forced"
            }

        # Original validation logic for all other tokens
        if self.is_negation_relation(relation):
            return {
                "valid": False,
                "reason": f"negation relation '{relation}' adds no semantic value",
                "corrected_triple": None,
                "action": "reject_negation",
            }

        if self.is_vague_relation(relation):
            return {
                "valid": False,
                "reason": f"vague relation '{relation}' – be more specific",
                "corrected_triple": None,
                "action": "reject",
            }

        if self.spacy_nlp is None:
            return {
                "valid": True,
                "reason": None,
                "corrected_triple": None,
                "action": "accept",
            }

        subj_doc = self.spacy_nlp(subj) if subj else None
        relation_for_parse = (
            relation.strip().lower().replace("_", " ") if relation else ""
        )
        rel_doc = self.spacy_nlp(relation_for_parse) if relation_for_parse else None
        obj_doc = self.spacy_nlp(obj) if obj else None

        subj_pos = subj_doc[0].pos_ if subj_doc and len(subj_doc) > 0 else None
        obj_pos = obj_doc[0].pos_ if obj_doc and len(obj_doc) > 0 else None

        rel_lemma, rel_has_verb = self.extract_verb_info(relation, rel_doc)

        if subj_pos == "ADJ" and obj_pos == "ADJ":
            return {
                "valid": False,
                "reason": f"both subject '{subj}' and object '{obj}' are adjectives",
                "corrected_triple": None,
                "action": "reject",
            }

        result = self.check_has_adjective(rel_lemma, obj_pos, obj)
        if result:
            return result

        result = self.check_circular_is(
            rel_lemma, obj_pos, subj, obj, subj_doc, obj_doc
        )
        if result:
            return result

        result = self.check_adjective_subject(subj, subj_pos, rel_has_verb, relation, rel_doc)
        if result:
            return result

        result = self.check_adjective_object(
            subj_pos, obj_pos, rel_has_verb, relation, rel_doc, obj
        )
        if result:
            return result

        if rel_has_verb:
            return {
                "valid": True,
                "reason": None,
                "corrected_triple": None,
                "action": "accept",
            }

        result = self.handle_missing_verb(rel_doc, obj_doc, subj, relation, obj)
        if result:
            return result

        return {
            "valid": False,
            "reason": f"relation '{relation}' has no verb",
            "corrected_triple": None,
            "action": "reject",
        }