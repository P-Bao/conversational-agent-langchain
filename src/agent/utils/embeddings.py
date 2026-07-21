"""Embedding model utilities."""

from langchain_core.embeddings import Embeddings

from agent.utils.config import Config


def get_embedding_model(cfg: Config) -> Embeddings:
    """Return an embeddings client for the configured provider."""
    provider = cfg.embedding_provider
    model_name = cfg.embedding_model

    match provider:
        case "cohere":
            from langchain_cohere import CohereEmbeddings  # noqa: PLC0415

            return CohereEmbeddings(model=model_name, cohere_api_key=cfg.cohere_api_key or None)

        case "google":
            try:
                from langchain_google_genai import GoogleGenerativeAIEmbeddings  # noqa: PLC0415
            except ImportError as exc:
                msg = "langchain-google-genai is required for Google embeddings."
                raise ImportError(msg) from exc

            return GoogleGenerativeAIEmbeddings(
                model=cfg.embedding_model,
                google_api_key=cfg.gemini_api_key or None,
                output_dimensionality=cfg.embedding_size,
            )

        case "openai":
            from langchain_openai import OpenAIEmbeddings  # noqa: PLC0415

            return OpenAIEmbeddings(
                model=model_name,
                api_key=cfg.openai_api_key or None,
            )

        case "openai-compatible" | "custom":
            if not cfg.embedding_base_url:
                msg = "embedding_base_url is required for 'openai-compatible' provider."
                raise ValueError(msg)

            if "generativelanguage.googleapis" in cfg.embedding_base_url:
                try:
                    from langchain_google_genai import GoogleGenerativeAIEmbeddings  # noqa: PLC0415
                except ImportError as exc:
                    msg = "langchain-google-genai is required for Google embeddings."
                    raise ImportError(msg) from exc

                return GoogleGenerativeAIEmbeddings(
                    model=model_name,
                    google_api_key=cfg.embedding_api_key or None,
                )

            from langchain_openai import OpenAIEmbeddings  # noqa: PLC0415
            return OpenAIEmbeddings(
                model=model_name,
                api_key=cfg.embedding_api_key or None,
                base_url=cfg.embedding_base_url,
            )

        case _:
            msg = "No suitable embedding Model configured!"
            raise KeyError(msg)
