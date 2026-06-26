import llama_cpp


llm = llama_cpp.Llama(
    model_path="./models/qwen2.5-7b-instruct-q5_k_m-00001-of-00002.gguf",
    n_gpu_layers=-1,        # Set to -1 to offload everything to the GPU
    n_ctx=1024,             # Context window size
    verbose=False,
)

def chat_completion(prompt: str) -> str:
    system_prompt = "Answer in two concise sentences."
    generation_params = {
    "temperature": 0.7,
    "top_p": 0.9,
    "stop": ["<|im_end|>", "<|endoftext|>"],
    "max_tokens": 2048,
    }

    answer = llm.create_chat_completion(
    messages=[
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": prompt}
    ],
    **generation_params
    )

    return answer["choices"][0]["message"]["content"]
 

if __name__ == "__main__":
    print(chat_completion(prompt="What is coffee?"))
    print(chat_completion(prompt="Is it good for your health?"))