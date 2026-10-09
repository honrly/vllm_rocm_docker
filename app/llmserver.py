from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import time


def main():
    model_name = "tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.3"

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    llm = LLM(
        model=model_name,
        tensor_parallel_size=1,
        enforce_eager=True,
        dtype="float16",
    )

    sampling_params = SamplingParams(
        temperature=0.6,
        top_p=0.9,
        max_tokens=512,
        stop="<|eot_id|>",
    )

    messages = [
        {
            "role": "system",
            "content": "あなたは誠実で優秀な日本人の介護職員です。",
        },
        {
            "role": "user",
            "content": "こんにちは、今日の天気はどうですか？",
        },
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    
    start_time = time.time()

    outputs = llm.generate(prompt, sampling_params)
    
    end_time = time.time()

    print(outputs[0].outputs[0].text)
    print(f"時間: {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    main()