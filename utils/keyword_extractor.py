import openai
from utils.image_search import load_api_key

# Make sure to set your OpenAI API key
openai.api_key = load_api_key(label="GPT")  # or "OPENAI" if that's your label

def extract_keywords(prompt: str, max_keywords: int = 5):
    system_prompt = (
        "You are a helpful assistant for scientific visualization. "
        f"Given a user prompt, extract up to {max_keywords} clean, distinct keywords useful for image search. "
        "Focus on named entities, scientific concepts, image style, and objects. "
        "Return ONLY a comma-separated list of the keywords (no explanations, no extra words, no numbers)."
    )

    user_input = f"Prompt: {prompt}"

    try:
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input},
            ],
            max_tokens=40,
            temperature=0.3
        )
        # The response content (should be a string: "keyword1, keyword2, ...")
        raw = response.choices[0].message.content.strip()
        keywords = [kw.strip().strip('"') for kw in raw.replace(",", "\n").split("\n") if kw.strip()]
        return keywords
    except Exception as e:
        print("[extract_keywords] OpenAI API error:", e)
        return []
