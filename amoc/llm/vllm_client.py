import logging
from typing import List, Dict

from vllm import LLM, SamplingParams

from amoc.llm.parsing import (
    parse_for_dict,
    extract_list_from_string,
)

from amoc.prompts.amoc_prompts import (
    NEW_RELATIONSHIPS_PROMPT,
    NEW_RELATIONSHIPS_FOR_FIRST_SENTENCE_PROMPT,
    INFER_OBJECTS_AND_PROPERTIES_PROMPT,
    GENERATE_NEW_INFERRED_RELATIONSHIPS_PROMPT,
    INFER_OBJECTS_AND_PROPERTIES_FIRST_SENTENCE_PROMPT,
    GENERATE_NEW_INFERRED_RELATIONSHIPS_FIRST_SENTENCE_PROMPT,
    SELECT_RELEVANT_EDGES_PROMPT,
    REPLACE_PRONOUNS_PROMPT,
)


class VLLMClient:
    def __init__(
        self,
        model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct",
        tp_size: int = 1,
        debug: bool = False,
    ):
        self.debug = debug
        self.model_name = model_name
        self.tp_size = tp_size
        if debug:
            logging.getLogger().setLevel(logging.DEBUG)
        else:
            logging.getLogger().setLevel(logging.INFO)

        logging.info(f"Initializing vLLM with model: {model_name}, tp_size={tp_size}")
        self.llm = LLM(
            model=model_name,
            tensor_parallel_size=tp_size,
            trust_remote_code=True,
            gpu_memory_utilization=0.80,
            max_model_len=5000,
        )
        self.sampling_params = SamplingParams(
            temperature=0.8, top_p=0.95, max_tokens=1024
        )

        import torch

        num_gpus = torch.cuda.device_count()
        if tp_size > num_gpus:
            raise ValueError(
                f"Requested tensor parallel size {tp_size}, but only {num_gpus} GPUs are available."
            )

    def generate(self, messages: List[Dict[str, str]]) -> str:
        if self.llm is None:
            logging.error("VLLM not initialized.")
            return "[]"

        # Manual prompt template construction
        prompt = ""
        for m in messages:
            role = m.get("role", "user")
            content = (m.get("content") or "").strip()
            prompt += (
                f"<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>"
            )

        # Force Start to 'final' channel (The "Anti-Loop" Fix)
        prompt += "<|start_header_id|>assistant<|end_header_id|>\n\nfinal\n"

        try:
            outputs = self.llm.generate([prompt], self.sampling_params, use_tqdm=False)
            raw_text = outputs[0].outputs[0].text

            # Keep output as-is; downstream parsing helpers extract the first
            # [...] or {...} block. Mutating here can easily corrupt otherwise
            # parseable responses (e.g., prefixing '[' before prose containing a list).
            if "final" in raw_text:
                return raw_text.split("final")[-1].strip()
            return raw_text.strip()

        except Exception as e:
            logging.exception(f"VLLM runtime error: {e}")
            return "[]"

    def call_vllm(self, prompt: str, persona: str) -> str:
        # Inject persona into system prompt
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a knowledge graph builder and reasoning agent.\n"
                    "You must reason about the following persona and age, and extract factual relationships about them (and related concepts) only when they are supported by the text.\n\n"
                    f"Persona description:\n{persona}\n\n"
                    "Do not invent new attributes or relationships that are not supported by the persona description itself."
                ),
            },
            {"role": "user", "content": prompt},
        ]
        return self.generate(messages)

    # --- Wrappers for AMoC Logic ---
    # Now accepting 'persona' arg passed down from AMoC class

    def get_new_relationships(
        self, nodes_from_text, nodes_from_graph, edges_from_graph, text, persona
    ):
        prompt = NEW_RELATIONSHIPS_PROMPT.format(
            nodes_from_text=nodes_from_text,
            nodes_from_graph=nodes_from_graph,
            edges_from_graph=edges_from_graph,
            text=text,
        )
        response = self.call_vllm(prompt, persona)
        return extract_list_from_string(response)

    def get_new_relationships_first_sentence(self, nodes_from_text, text, persona):
        prompt = NEW_RELATIONSHIPS_FOR_FIRST_SENTENCE_PROMPT.format(
            nodes_from_text=nodes_from_text, text=text
        )
        response = self.call_vllm(prompt, persona)
        return extract_list_from_string(response)

    def infer_objects_and_properties(
        self, nodes_from_text, nodes_from_graph, edges_from_graph, text, persona
    ):
        prompt = INFER_OBJECTS_AND_PROPERTIES_PROMPT.format(
            nodes_from_text=nodes_from_text,
            nodes_from_graph=nodes_from_graph,
            edges_from_graph=edges_from_graph,
            text=text,
        )
        response = self.call_vllm(prompt, persona)
        return parse_for_dict(response)

    def generate_new_inferred_relationships(
        self,
        nodes_from_text,
        nodes_from_graph,
        edges_from_graph,
        concepts,
        properties,
        text,
        persona,
    ):
        prompt = GENERATE_NEW_INFERRED_RELATIONSHIPS_PROMPT.format(
            nodes_from_text=nodes_from_text,
            nodes_from_graph=nodes_from_graph,
            edges_from_graph=edges_from_graph,
            concepts=concepts,
            properties=properties,
            text=text,
        )
        response = self.call_vllm(prompt, persona)
        return parse_for_dict(response)

    def infer_objects_and_properties_first_sentence(
        self, nodes_from_text, text, persona
    ):
        prompt = INFER_OBJECTS_AND_PROPERTIES_FIRST_SENTENCE_PROMPT.format(
            nodes_from_text=nodes_from_text, text=text
        )
        response = self.call_vllm(prompt, persona)
        return parse_for_dict(response)

    def generate_new_inferred_relationships_first_sentence(
        self, nodes_from_text, concepts, properties, text, persona
    ):
        prompt = GENERATE_NEW_INFERRED_RELATIONSHIPS_FIRST_SENTENCE_PROMPT.format(
            nodes_from_text=nodes_from_text,
            concepts=concepts,
            properties=properties,
            text=text,
        )
        response = self.call_vllm(prompt, persona)
        return parse_for_dict(response)

    def get_relevant_edges(self, edges_from_graph, text, persona):
        prompt = SELECT_RELEVANT_EDGES_PROMPT.format(edges=edges_from_graph, text=text)
        response = self.call_vllm(prompt, persona)
        return extract_list_from_string(response)

    def resolve_pronouns(self, text, persona):
        prompt = REPLACE_PRONOUNS_PROMPT + text
        return self.call_vllm(prompt, persona)
