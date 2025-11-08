from unittest.mock import patch
from src import translator
from src.translator import translate_content


def test_chinese():
    """Simulate translating Chinese text successfully via mocked LLM."""
    with patch.object(translator, "query_llm_robust", return_value=(False, "This is a Chinese message")):
        is_english, translated_content = translate_content("这是一条中文消息")
        assert is_english is False
        assert translated_content == "This is a Chinese message"


def test_llm_normal_response():
    """Simulate a normal, well-formed LLM response."""
    with patch.object(translator, "query_llm_robust", return_value=(True, "Hello world")):
        is_english, translated_content = translate_content("Hello world")
        assert is_english is True
        assert translated_content == "Hello world"


def test_llm_gibberish_response():
    """Simulate an invalid/malformed LLM response and ensure safe fallback."""
    # Case 1: returns something not a tuple
    with patch.object(translator, "query_llm_robust", return_value="nonsense"):
        is_english, translated_content = translate_content("Hola")
        assert is_english is False
        assert translated_content == "Hola"

    # Case 2: tuple with invalid types
    with patch.object(translator, "query_llm_robust", return_value=("bad", 123)):
        is_english, translated_content = translate_content("Bonjour")
        assert is_english is False
        assert translated_content == "Bonjour"
