from collections import deque

import llama_cpp
from huggingface_hub import hf_hub_download

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
            n_gpu_layers=-1,                         # Set to -1 to offload everything to the GPU
            n_ctx=self.context_window,  # Context window size
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

        # Newest -> oldest, skip the latest user turn because it is `current`.
        for msg in reversed(list(self.history)[:-1]):
            if used + msg["n_tokens"] <= self.prompt_budget:
                window.insert(1, msg)    # Keep timeline order
                used += msg["n_tokens"]
            elif msg.get("pinned", False):
                continue
            else:
                break                    # Window is full

        window.append(current)
        return self._to_chat_messages(window)

    def answer(self, user_text: str) -> str:
        # 1. Log incoming turn
        user_msg = self._msg("user", user_text)
        self.history.append(user_msg)

        # 2. Craft prompt window
        prompt_window = self._build_window(user_msg)

        # 3. Ask the model
        reply = self.llm.create_chat_completion(
            messages=prompt_window,
            **self.gen_params,
        )["choices"][0]["message"]["content"]

        # 4. Store assistant turn
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

    agent = LLMWithSlidingWindow(model_path, "Answer in two concise sentences.")

    while True:
        user_input = input("User: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        response = agent.answer(user_input)
        print(f"Assistant: {response}")
        print(f"Stats: {agent.stats()}")
        print("\n")