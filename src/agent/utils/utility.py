"""Utility module."""

import uuid
from collections.abc import Sequence
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from loguru import logger

from agent.data_model.internal_model import RetrievalResults

# add new languages to detect here


def combine_text_from_list(input_list: list) -> str:
    """Combines all strings in a list to one string."""
    logger.info(f"List: {input_list}")
    for text in input_list:
        if not isinstance(text, str):
            msg = "Input list must contain only strings"
            raise TypeError(msg)
    return "\n".join(input_list)


def create_tmp_folder() -> str:
    """Creates a temporary folder for files to store.

    Returns
    -------
        str: The directory name.

    """
    # Create a temporary folder to save the files
    tmp_dir = Path.cwd() / f"tmp_{uuid.uuid4()}"
    try:
        tmp_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created new folder {tmp_dir}.")
    except ValueError as e:
        logger.error(f"Failed to create directory {tmp_dir}. Error: {e}")
        raise
    return str(tmp_dir)


def format_docs_for_citations(docs: Sequence[Document]) -> str:
    """Format the documents for citations.

    Args:
    ----
        docs (Sequence[Document]): Langchain documents from a vectordatabase.

    Returns:
    -------
        str: Combined documents in a format suitable for citations.

    """
    formatted_docs = []
    for i, doc in enumerate(docs):
        doc_string = f"<doc id='{i}'>{doc.page_content}</doc>"
        formatted_docs.append(doc_string)
    return "\n".join(formatted_docs)
