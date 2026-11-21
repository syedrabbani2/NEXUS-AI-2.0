
from unittest.mock import MagicMock

from modules import embeddings


def test_create_embeddings_uses_correct_model(monkeypatch):
    mock_embeddings = MagicMock()

    monkeypatch.setattr(
        embeddings,
        "HuggingFaceEmbeddings",
        mock_embeddings,
    )

    result = embeddings.create_embeddings()

    mock_embeddings.assert_called_once_with(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    assert result is mock_embeddings.return_value


def test_create_embeddings_returns_model_instance(monkeypatch):
    expected_instance = object()

    monkeypatch.setattr(
        embeddings,
        "HuggingFaceEmbeddings",
        lambda **kwargs: expected_instance,
    )

    result = embeddings.create_embeddings()

    assert result is expected_instance
