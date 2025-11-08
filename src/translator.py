import os
from ollama import Client

# Initialize Ollama client
OLLAMA_URL = os.getenv("OLLAMA_HOST", "http://localhost:11434")
MODEL_NAME = os.getenv("OLLAMA_MODEL", "mistral:7b")

client = Client(host=OLLAMA_URL)

TRANSLATION_CONTEXT = """\
You are a professional translator. Translate the input text to English. 
If the text is already in English, return it as is. Only provide the translation, nothing else.

Example:
INPUT: Bonjour, je m'appelle Bob
OUTPUT: Hello, my name is Bob
"""

CLASSIFICATION_CONTEXT = """\
You are a language classifier. Detect the language of the input text and reply only with the English name of that language.

Example:
INPUT: Bonjour, je m'appelle Bob
OUTPUT: French
"""

def get_language(post: str) -> str:
    prompt = f"{CLASSIFICATION_CONTEXT}\n\nINPUT: {post}\nOUTPUT:"
    response = client.chat(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}]
    )
    return response["message"]["content"].strip()

def get_translation(post: str) -> str:
    prompt = f"{TRANSLATION_CONTEXT}\n\nINPUT: {post}\nOUTPUT:"
    response = client.chat(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}]
    )
    return response["message"]["content"].strip()

def query_llm_robust(post: str) -> tuple[bool, str]:
    """Robust LLM query with full validation and fallback behavior."""
    if post is None:
        return (False, "")
    if not isinstance(post, str):
        try:
            post = str(post)
        except Exception:
            return (False, "")
    if not post.strip():
        return (False, post)

    try:
        alpha_ratio = sum(1 for c in post if c.isalpha()) / len(post)
        if alpha_ratio < 0.3:
            return (False, post)
    except Exception:
        return (False, post)

    # Language detection
    try:
        language_response = get_language(post)
        if not isinstance(language_response, str) or not language_response.strip():
            raise ValueError("Invalid language response")

        language = language_response.strip().lower()
        if any(p in language for p in ["error", "invalid", "sorry", "help", "request"]):
            raise ValueError(f"Error-like language response: {language}")

        is_english = language == "english"
    except Exception as e:
        print(f"Language detection failed: {e}")
        return (False, post)

    if is_english:
        return (True, post)

    # Translation
    try:
        translation_response = get_translation(post)
        if not isinstance(translation_response, str) or not translation_response.strip():
            raise ValueError("Invalid translation response")

        translation = translation_response.strip()
        if len(translation) > len(post) * 5:
            raise ValueError("Translation too long")
        if not any(c.isalpha() for c in translation):
            raise ValueError("Translation contains no letters")

        return (False, translation)
    except Exception as e:
        print(f"Translation failed: {e}")
        return (False, post)

def translate_content(content: str) -> tuple[bool, str]:
    try:
        result = query_llm_robust(content)

        # Validate structure and types
        if (
            not isinstance(result, tuple)
            or len(result) != 2
            or not isinstance(result[0], bool)
            or not isinstance(result[1], str)
        ):
            # Gracefully recover from malformed responses
            return (False, content)

        return result

    except Exception as e:
        print(f"Error during translation: {e}")
        return (False, content)