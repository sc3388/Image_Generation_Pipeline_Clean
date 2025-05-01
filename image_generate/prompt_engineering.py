import openai
from utils.image_search import load_api_key

# Set up your OpenAI API key (you can modify the path/label if needed)
openai.api_key = load_api_key(label="GPT")

def enrich_prompt_llama(prompt):
    system_message = (
        "You are a prompt engineer for an AI image generation tool focused on clean, cartoon-style illustrations. "
        "Rewrite the user's prompt to improve visual clarity and composition, while keeping the scene simple and presentation-friendly. "
        "Use flat color language, avoid detailed names or character lore, and limit visual complexity. "
        "The output should ONLY be the prompt, with NO explanation, and MUST be under 77 tokens."
        "The output should specifically emphasize do not include text caption in the generated image"
    )
    try:
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ],
            max_tokens=80,
            temperature=0.7
        )
        enriched = response.choices[0].message.content.strip()
        return enriched if enriched else prompt
    except Exception as e:
        print("Error enriching prompt with GPT-4o:", e)
        return prompt
