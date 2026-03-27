from llama_cpp import Llama

print("Loading moondream2...")

llm = Llama(
    model_path="models/moondream2/moondream2-text-model-f16_ct-vicuna.gguf",
    n_ctx=2048,
    verbose=False,
)

print("Model loaded successfully.")

output = llm(
    "Q: What is the capital of France?\nA:",
    max_tokens=50,
)

print(output["choices"][0]["text"])