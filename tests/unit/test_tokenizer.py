"""Isolated tests for core.tokenizer module."""

import pytest
from core.tokenizer import (
    _WhitespaceTokenizer,
    load_tokenizer,
    tokenize_text,
    decode_tokens,
    pad_and_tokenize,
    Tokenizer,
)


def _get_tokenizer():
    """Helper to get tokenizer, skip test if HuggingFace unavailable."""
    try:
        tok = load_tokenizer()
        # Test if it actually works (network available)
        tok("test")
        return tok
    except (OSError, Exception):
        pytest.skip("HuggingFace tokenizer unavailable (no network or model not cached)")


class TestWhitespaceTokenizer:
    def test_single_text(self):
        tok = _WhitespaceTokenizer()
        result = tok("hello world")
        assert "input_ids" in result
        assert len(result["input_ids"]) == 1
        assert len(result["input_ids"][0]) == 2

    def test_batch_text(self):
        tok = _WhitespaceTokenizer()
        result = tok(["a b", "c d e"])
        assert len(result["input_ids"]) == 2

    def test_padding_max_length(self):
        tok = _WhitespaceTokenizer()
        result = tok("a", padding=True, max_length=5)
        assert len(result["input_ids"][0]) == 5

    def test_truncation(self):
        tok = _WhitespaceTokenizer()
        result = tok("a b c d e", truncation=True, max_length=3)
        assert len(result["input_ids"][0]) == 3

    def test_longest_padding(self):
        tok = _WhitespaceTokenizer()
        result = tok(["a", "a b c"], padding="longest")
        assert len(result["input_ids"][0]) == len(result["input_ids"][1])

    def test_unknown_tokens(self):
        tok = _WhitespaceTokenizer(vocab={"<pad>": 0, "<unk>": 1, "hello": 2})
        result = tok("hello unknown")
        assert result["input_ids"][0][0] == 2
        assert result["input_ids"][0][1] == 1  # <unk>

    def test_decode(self):
        tok = _WhitespaceTokenizer()
        text = tok.decode([1, 2, 3])
        assert text == "1 2 3"

    def test_batch_decode(self):
        tok = _WhitespaceTokenizer()
        texts = tok.batch_decode([[1, 2], [3]])
        assert len(texts) == 2

    def test_attention_mask(self):
        tok = _WhitespaceTokenizer()
        result = tok("a b", padding=True, max_length=5)
        mask = result["attention_mask"][0]
        assert mask[:2] == [1, 1]
        assert mask[2:] == [0, 0, 0]

    def test_token_type_ids(self):
        tok = _WhitespaceTokenizer()
        result = tok("a b", return_token_type_ids=True)
        assert "token_type_ids" in result
        assert all(v == 0 for v in result["token_type_ids"][0])


class TestLoadTokenizer:
    def test_returns_tokenizer(self):
        tok = _get_tokenizer()
        assert callable(tok)


class TestTokenizeText:
    def test_single_text(self):
        tok = _get_tokenizer()
        result = tokenize_text("hello world", tok)
        assert isinstance(result, list)

    def test_batch_text(self):
        tok = _get_tokenizer()
        result = tokenize_text(["hello", "world"], tok, padding="longest")
        assert "input_ids" in result


class TestDecodeTokens:
    def test_single_sequence(self):
        tok = _get_tokenizer()
        text = decode_tokens([1, 2], tok)
        assert isinstance(text, str)

    def test_batch(self):
        tok = _get_tokenizer()
        texts = decode_tokens([[1, 2], [3]], tok)
        assert isinstance(texts, list)
        assert len(texts) == 2

    def test_dict_input(self):
        tok = _get_tokenizer()
        result = decode_tokens({"input_ids": [[1, 2], [3]]}, tok)
        assert isinstance(result, list)

    def test_dict_missing_key_raises(self):
        tok = _get_tokenizer()
        with pytest.raises(ValueError):
            decode_tokens({"bad_key": [1]}, tok)


class TestPadAndTokenize:
    def test_basic(self):
        tok = _get_tokenizer()
        result = pad_and_tokenize(["hello world", "hi"], tok, max_length=10)
        assert "input_ids" in result
        assert len(result["input_ids"]) == 2


class TestTokenizerWrapper:
    def test_call(self):
        tok = _get_tokenizer()  # Verify network first
        t = Tokenizer()
        result = t("hello")
        assert "input_ids" in result

    def test_decode(self):
        tok = _get_tokenizer()  # Verify network first
        t = Tokenizer()
        text = t.decode([1, 2])
        assert isinstance(text, str)

    def test_batch_decode(self):
        tok = _get_tokenizer()  # Verify network first
        t = Tokenizer()
        texts = t.batch_decode([[1], [2]])
        assert len(texts) == 2
