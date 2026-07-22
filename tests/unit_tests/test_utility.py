"""Tests for the utility functions."""
import pytest
from langchain_core.documents import Document
from agent.utils.utility import (
    combine_text_from_list,
    create_tmp_folder,
    format_docs_for_citations,
)


def test_combine_text_from_list():
    """Test that combine_text_from_list returns the correct string."""
    input_list = ["a", "b", "c"]
    result = combine_text_from_list(input_list)
    assert result == "a\nb\nc"


def test_combine_text_from_list_with_non_string():
    """Test that combine_text_from_list raises a TypeError if the list contains a non-string."""
    input_list = ["a", "b", 1]
    with pytest.raises(TypeError):
        combine_text_from_list(input_list)


def test_create_tmp_folder():
    """Test that create_tmp_folder returns a valid directory name."""
    import os
    tmp_dir = create_tmp_folder()
    assert os.path.isdir(tmp_dir)
    os.rmdir(tmp_dir)


def test_format_docs_for_citations():
    """Test that format_docs_for_citations returns the correct string."""
    docs = [
        Document(page_content="This is a test document.", metadata={"source": "test"}),
        Document(page_content="This is another test document.", metadata={"source": "test"}),
    ]
    result = format_docs_for_citations(docs)
    assert result == "<doc id='0'>This is a test document.</doc>\n<doc id='1'>This is another test document.</doc>"
