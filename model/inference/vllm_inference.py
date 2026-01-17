import json
import base64
from openai import OpenAI

# vLLM server configuration
VLLM_BASE_URL = "http://localhost:8000/v1"
MODEL_NAME = "opencua-7b"  # Should match --served-model-name in vllm serve

TEST_CASE_DIR = "./grounding_examples"
test_cases = [
    f'{TEST_CASE_DIR}/test0.json',
    f'{TEST_CASE_DIR}/test1.json',
    f'{TEST_CASE_DIR}/test2.json',
    f'{TEST_CASE_DIR}/test3.json',
    f'{TEST_CASE_DIR}/test4.json',
]

SYSTEM_PROMPT = (
    "You are a GUI agent. You are given a task and a screenshot of the screen. "
    "You need to perform a series of pyautogui actions to complete the task."
)


def encode_image(path: str) -> str:
    """Encode image to base64 string."""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def get_test_messages(case_file: str):
    """Load test case and create chat messages."""
    with open(case_file) as f:
        info = json.load(f)
        img_name = info['image'].split('/')[-1]
        img_path = f'{TEST_CASE_DIR}/{img_name}'
        user_prompt = info['instruction']

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{encode_image(img_path)}"
                    }
                },
                {"type": "text", "text": user_prompt},
            ],
        },
    ]
    return messages, img_path


def main():
    # Initialize OpenAI client pointing to vLLM server
    client = OpenAI(
        base_url=VLLM_BASE_URL,
        api_key="EMPTY",  # vLLM doesn't require a real API key
    )

    print(f"Connecting to vLLM server at {VLLM_BASE_URL}")
    print(f"Using model: {MODEL_NAME}")
    print()

    # Run test cases
    for tc in test_cases:
        messages, img_path = get_test_messages(tc)

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            max_tokens=512,
            temperature=0,
        )

        output_text = response.choices[0].message.content

        print("=" * 100)
        print(f"Test case: {tc}")
        print(f"Output: {output_text}")
        print("=" * 100)
        print()


if __name__ == "__main__":
    main()
