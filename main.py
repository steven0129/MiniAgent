import json
import re
from collections import deque
from typing import Dict, List, Tuple

import llama_cpp
import networkx as nx
from huggingface_hub import hf_hub_download


class KnowledgeGraph:
    """
    Thin wrapper around NetworkX for storing and traversing triplets.
    Each edge represents a relationship: (subject) --[relation]--> (object)
    """

    def __init__(self):
        self.graph = nx.DiGraph()

    def add_triplet(self, subject: str, relation: str, obj: str):
        """Add or update a triplet in the graph."""
        subject = subject.lower().strip()
        relation = relation.lower().strip()
        obj = obj.lower().strip()

        if not subject or not relation or not obj:
            return

        self._apply_simple_updates(subject, relation, obj)

        if not self.graph.has_node(subject):
            self.graph.add_node(subject)
        if not self.graph.has_node(obj):
            self.graph.add_node(obj)

        self.graph.add_edge(subject, obj, relation=relation)

    def _apply_simple_updates(self, subject: str, relation: str, obj: str):
        """Handle a few common personal-memory updates so stale facts do not win."""
        romantic_relations = {"girlfriend", "boyfriend", "partner", "dating"}

        if subject == "user" and relation in romantic_relations:
            for _, old_obj, data in list(self.graph.out_edges(subject, data=True)):
                if data.get("relation") in romantic_relations and old_obj != obj:
                    self.graph.remove_edge(subject, old_obj)

        if subject == "user" and relation in {"broke_up_with", "ex_girlfriend", "ex_boyfriend"}:
            for _, old_obj, data in list(self.graph.out_edges(subject, data=True)):
                if old_obj == obj and data.get("relation") in romantic_relations:
                    self.graph.remove_edge(subject, old_obj)

    def get_related_triplets(
        self,
        entities: List[str],
        relation_hints: List[str] | None = None,
        max_depth: int = 2,
    ) -> List[Tuple[str, str, str]]:
        """
        Starting from seed entities, traverse up to max_depth hops.
        Returns list of (subject, relation, object) tuples.
        """
        triplets: List[Tuple[str, str, str]] = []
        visited_edges = set()

        entities = [e.lower().strip() for e in entities if e and e.strip()]
        relation_hints = [
            r.lower().strip()
            for r in (relation_hints or [])
            if r and r.strip()
        ]

        current_frontier = [e for e in entities if self.graph.has_node(e)]

        for _ in range(max_depth):
            next_frontier = []

            for node in current_frontier:
                for _, neighbor, data in self.graph.out_edges(node, data=True):
                    edge_key = (node, data["relation"], neighbor)
                    if edge_key not in visited_edges:
                        visited_edges.add(edge_key)
                        triplets.append(edge_key)
                        next_frontier.append(neighbor)

                for neighbor, _, data in self.graph.in_edges(node, data=True):
                    edge_key = (neighbor, data["relation"], node)
                    if edge_key not in visited_edges:
                        visited_edges.add(edge_key)
                        triplets.append(edge_key)
                        next_frontier.append(neighbor)

            current_frontier = next_frontier
            if not current_frontier:
                break

        if relation_hints and triplets:
            triplets.sort(
                key=lambda triplet: self._relevance_score(triplet, relation_hints),
                reverse=True,
            )

        return triplets

    @staticmethod
    def _relevance_score(
        triplet: Tuple[str, str, str],
        relation_hints: List[str],
    ) -> int:
        _, relation, _ = triplet

        if relation in relation_hints:
            return 2
        if any(hint in relation or relation in hint for hint in relation_hints):
            return 1
        return 0

    def triplets_to_context(self, triplets: List[Tuple[str, str, str]]) -> str:
        """Serialize triplets into a human-readable context block."""
        if not triplets:
            return ""

        lines = ["Known facts:"]
        for subject, relation, obj in triplets:
            lines.append(f" - {subject} --[{relation}]--> {obj}")
        return "\n".join(lines)

    def print_graph(self):
        """Print all triplets in the graph."""
        print("\n=== Knowledge Graph Contents ===")
        if self.graph.number_of_edges() == 0:
            print(" (empty)")
        else:
            for subject, obj, data in self.graph.edges(data=True):
                print(f" ({subject}) --[{data['relation']}]--> ({obj})")
        print("================================\n")

    def stats(self) -> Dict[str, int]:
        return {
            "nodes": self.graph.number_of_nodes(),
            "edges": self.graph.number_of_edges(),
        }


class LLMExtractor:
    """Handles entity and triplet extraction via LLM calls."""

    QUERY_EXTRACTION_PROMPT = """
Analyze this query and extract:
1. Entities: people, places, things mentioned. Use "user" for I/me/my.
2. Relations: relationship types being asked about.

Return ONLY JSON. No explanation.

Examples:
Text: "Who is my manager?"
Output: {{"entities": ["user"], "relations": ["manager", "reports_to", "supervised_by"]}}

Text: "Where does Sarah live?"
Output: {{"entities": ["Sarah"], "relations": ["lives_in", "resides_in", "located_in"]}}

Text: "Does Tom have any colleagues at Microsoft?"
Output: {{"entities": ["Tom", "Microsoft"], "relations": ["colleague", "works_with", "works_at"]}}

Text: "What is my dog's name?"
Output: {{"entities": ["user"], "relations": ["owns", "pet", "dog"]}}

Text: "What's the weather today?"
Output: {{"entities": [], "relations": []}}

Now extract from:
Text: "{text}"
Output:
"""

    TRIPLET_EXTRACTION_PROMPT = """
Extract factual relationships from this text as triplets.
Each triplet has: subject, relation, object.
Return ONLY a JSON array of objects. No explanation.

CRITICAL RULES:
1. ONLY extract facts the user is STATING, not asking about.
2. IGNORE questions entirely - they extract NOTHING.
3. Extract facts about:
   - The user themselves. Use "user" for I/me/my.
   - People the user mentions: friends, family, colleagues, pets, etc.
   - Relationships between people the user knows.
4. IGNORE general world knowledge: history, science, geography, trivia, etc.
5. Use simple relation names: works_at, sister_of, lives_in, owns, dating, girlfriend, broke_up_with.

Examples:
Text: "My brother is called Marcus"
Output: [{{"subject": "user", "relation": "brother", "object": "Marcus"}}]

Text: "I live in Tokyo and my manager is David"
Output: [{{"subject": "user", "relation": "lives_in", "object": "Tokyo"}}, {{"subject": "user", "relation": "manager", "object": "David"}}]

Text: "Emma works at Netflix"
Output: [{{"subject": "Emma", "relation": "works_at", "object": "Netflix"}}]

Text: "Alice is Bob's sister."
Output: [{{"subject": "Alice", "relation": "sister_of", "object": "Bob"}}]

Text: "What year was the Eiffel Tower built?"
Output: []

Text: "The speed of light is 299,792 km/s"
Output: []

Text: "Thanks for the help!"
Output: []

Now extract from:
Text: "{text}"
Output:
"""

    def __init__(self, llm: llama_cpp.Llama):
        self.llm = llm
        self.generation_params = {
            "temperature": 0.1,
            "top_p": 0.9,
            "max_tokens": 256,
            "stop": ["\n\n", "<|im_end|>", "<|endoftext|>"],
        }

    def extract_entities_and_relations(self, text: str) -> Dict[str, List[str]]:
        """Extract entities and relation hints from text."""
        prompt = self.QUERY_EXTRACTION_PROMPT.format(text=text)
        response = self.llm.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            **self.generation_params,
        )["choices"][0]["message"]["content"]

        result = self._loads_json(response, default={})
        if not isinstance(result, dict):
            return {"entities": [], "relations": []}

        return {
            "entities": self._string_list(result.get("entities", [])),
            "relations": self._string_list(result.get("relations", [])),
        }

    def extract_triplets(self, text: str) -> List[Dict[str, str]]:
        """Extract (subject, relation, object) triplets from text."""
        prompt = self.TRIPLET_EXTRACTION_PROMPT.format(text=text)
        response = self.llm.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            **self.generation_params,
        )["choices"][0]["message"]["content"]

        triplets = self._loads_json(response, default=[])
        if not isinstance(triplets, list):
            return []

        clean_triplets = []
        for triplet in triplets:
            if not isinstance(triplet, dict):
                continue

            subject = str(triplet.get("subject", "")).strip()
            relation = str(triplet.get("relation", "")).strip()
            obj = str(triplet.get("object", "")).strip()

            if subject and relation and obj:
                clean_triplets.append({
                    "subject": subject,
                    "relation": relation,
                    "object": obj,
                })

        return clean_triplets

    @staticmethod
    def _string_list(value) -> List[str]:
        if not isinstance(value, list):
            return []
        return [str(item).strip() for item in value if str(item).strip()]

    @staticmethod
    def _loads_json(text: str, default):
        """Parse strict JSON first, then try to recover the first JSON object/array."""
        stripped = text.strip()

        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            pass

        match = re.search(r"(\{.*\}|\[.*\])", stripped, flags=re.DOTALL)
        if not match:
            return default

        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            return default


class LLMWithKGMemory:
    """
    Chat wrapper using Knowledge Graph for memory.

    Flow:
    1. Extract entities and relations from user query
    2. Traverse graph to get relevant context
    3. Generate response with context
    4. Extract triplets from user message
    5. Update graph
    """

    SYSTEM_PROMPT = """
You are an AI assistant with access to personal relationship facts.
You may be provided with "Known facts" representing relationships you've learned.

Instructions:
1. If the user ASKS a question, use the known facts to answer. If the answer isn't in your facts, say you don't have that information.
2. If the user TELLS you new information, acknowledge it naturally. This information will be remembered.
3. When facts are provided, trust them as ground truth.
4. Be concise and conversational.
5. Do not mention "knowledge graph", "triplets", or "memory system" to the user.
""".strip()

    CONTEXT_WINDOW = 4096
    MAX_GENERATION_TOKENS = 512
    SAFETY_MARGIN = 32
    TRAVERSAL_DEPTH = 2

    def __init__(self, model_path: str):
        self.prompt_budget = (
            self.CONTEXT_WINDOW
            - self.MAX_GENERATION_TOKENS
            - self.SAFETY_MARGIN
        )
        self.generation_params = {
            "temperature": 0.7,
            "top_p": 0.9,
            "max_tokens": self.MAX_GENERATION_TOKENS,
            "stop": ["<|im_end|>", "<|endoftext|>"],
        }

        self.llm = llama_cpp.Llama(
            model_path=model_path,
            n_gpu_layers=-1,
            n_ctx=self.CONTEXT_WINDOW,
            verbose=False,
        )

        self.kg = KnowledgeGraph()
        self.extractor = LLMExtractor(self.llm)

    def answer(self, user_text: str) -> str:
        query_info = self.extractor.extract_entities_and_relations(user_text)
        entities = query_info["entities"]
        relation_hints = query_info["relations"]

        triplets = self.kg.get_related_triplets(
            entities,
            relation_hints=relation_hints,
            max_depth=self.TRAVERSAL_DEPTH,
        )
        context = self.kg.triplets_to_context(triplets)

        messages = self._build_prompt(user_text, context)
        reply = self.llm.create_chat_completion(
            messages=messages,
            **self.generation_params,
        )["choices"][0]["message"]["content"]

        new_triplets = self.extractor.extract_triplets(user_text)
        for triplet in new_triplets:
            self.kg.add_triplet(
                triplet["subject"],
                triplet["relation"],
                triplet["object"],
            )

        return reply

    def _build_prompt(self, user_text: str, context: str) -> List[Dict[str, str]]:
        system_content = self.SYSTEM_PROMPT
        if context:
            system_content += f"\n\n{context}"

        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_text},
        ]

    def print_graph(self):
        """Print all triplets in the graph."""
        self.kg.print_graph()

    def debug_retrieval(self, query: str):
        """Show what would be retrieved for a query without generating a response."""
        print(f"\n--- Debug: {query!r} ---")
        query_info = self.extractor.extract_entities_and_relations(query)
        print(f"Extracted entities: {query_info['entities']}")
        print(f"Relation hints: {query_info['relations']}")

        triplets = self.kg.get_related_triplets(
            query_info["entities"],
            relation_hints=query_info["relations"],
            max_depth=self.TRAVERSAL_DEPTH,
        )
        print(f"Retrieved triplets: {triplets}")
        print("----------------------------\n")

    def stats(self) -> Dict[str, int]:
        kg_stats = self.kg.stats()
        return {
            **kg_stats,
            "prompt_budget": self.prompt_budget,
            "context_window": self.CONTEXT_WINDOW,
            "max_generation_tokens": self.MAX_GENERATION_TOKENS,
            "traversal_depth": self.TRAVERSAL_DEPTH,
        }


class LLMWithSlidingWindow:
    def __init__(self, model_path, system_prompt: str):
        self.context_window = 4096
        self.max_generation_tokens = 1024
        self.safety_margin = 32
        self.k_turns = 10
        self.system_prompt = system_prompt
        self.prompt_budget = self.context_window - self.max_generation_tokens - self.safety_margin
        self.history = deque(maxlen=self.k_turns * 2)
        self.gen_params = {
            "max_tokens": self.max_generation_tokens,
            "temperature": 1,
            "top_p": 0.9,
            "stop": ["<|im_end|>", "<|endoftext|>"],
        }

        self.llm = llama_cpp.Llama(
            model_path=model_path,
            n_gpu_layers=-1,
            n_ctx=self.context_window,
            verbose=False,
        )

    def _tok(self, txt: str) -> int:
        """Return the exact token count as the model sees it."""
        return len(self.llm.tokenize(txt.encode(), special=True))

    def _msg(self, role: str, txt: str, pinned: bool = False) -> dict:
        return {
            "role": role,
            "content": txt,
            "n_tokens": self._tok(txt),
            "pinned": pinned,
        }

    def _to_chat_messages(self, messages: list[dict]) -> list[dict]:
        return [
            {"role": msg["role"], "content": msg["content"]}
            for msg in messages
        ]

    def _build_window(self, current: dict) -> list[dict]:
        """
        Assemble [system] + slice(history) + [current user] <= prompt_budget.
        Returns messages ready for llama_cpp.create_chat_completion().
        """
        system_msg = self._msg("system", self.system_prompt, pinned=True)
        used = system_msg["n_tokens"] + current["n_tokens"]
        window = [system_msg]

        for msg in reversed(list(self.history)[:-1]):
            if used + msg["n_tokens"] <= self.prompt_budget:
                window.insert(1, msg)
                used += msg["n_tokens"]
            elif msg.get("pinned", False):
                continue
            else:
                break

        window.append(current)
        return self._to_chat_messages(window)

    def answer(self, user_text: str) -> str:
        user_msg = self._msg("user", user_text)
        self.history.append(user_msg)

        prompt_window = self._build_window(user_msg)

        reply = self.llm.create_chat_completion(
            messages=prompt_window,
            **self.gen_params,
        )["choices"][0]["message"]["content"]

        self.history.append(self._msg("assistant", reply))
        return reply

    def stats(self) -> dict:
        history_tokens = sum(msg["n_tokens"] for msg in self.history)
        return {
            "history_messages": len(self.history),
            "history_tokens": history_tokens,
            "prompt_budget": self.prompt_budget,
            "context_window": self.context_window,
            "max_generation_tokens": self.max_generation_tokens,
            "safety_margin": self.safety_margin,
        }


if __name__ == "__main__":
    repo_id = "Qwen/Qwen2.5-7B-Instruct-GGUF"

    model_path = hf_hub_download(
        repo_id=repo_id,
        filename="qwen2.5-7b-instruct-q5_k_m-00001-of-00002.gguf",
        cache_dir="./models"
    )

    hf_hub_download(
        repo_id=repo_id,
        filename="qwen2.5-7b-instruct-q5_k_m-00002-of-00002.gguf",
        cache_dir="./models"
    )

    agent = LLMWithKGMemory(model_path)

    print("Knowledge Graph Memory chat is ready.")
    print("Commands: /graph, /debug <query>, exit, quit\n")

    while True:
        user_input = input("User: ").strip()

        if not user_input:
            continue

        if user_input.lower() in ["exit", "quit"]:
            break

        if user_input == "/graph":
            agent.print_graph()
            continue

        if user_input.startswith("/debug "):
            query = user_input.removeprefix("/debug ").strip()
            if query:
                agent.debug_retrieval(query)
            else:
                print("Usage: /debug <query>")
            continue

        response = agent.answer(user_input)
        print(f"Assistant: {response}")
        print(f"Stats: {agent.stats()}")
        print("\n")