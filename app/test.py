from vllm import LLM


def main():
    llm = LLM(
        model="facebook/opt-125m",
        enforce_eager=True,
        dtype="float16",
        max_model_len=128,
        gpu_memory_utilization=0.2,
    )

    print("loaded")


if __name__ == "__main__":
    main()