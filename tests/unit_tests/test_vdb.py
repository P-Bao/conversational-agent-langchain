import pytest
from unittest.mock import MagicMock, patch
from agent.utils.vdb import initialize_vector_db, init_vdb, initialize_all_vector_dbs
from agent.utils.config import Config


@patch("agent.utils.vdb.load_vec_db_conn")
def test_initialize_vector_db_exists(mock_load_conn):
    mock_client = MagicMock()
    mock_client.collection_exists.return_value = True
    mock_load_conn.return_value = mock_client

    initialize_vector_db("test_coll", 1024)

    mock_client.collection_exists.assert_called_with(collection_name="test_coll")
    mock_client.create_collection.assert_not_called()


@patch("agent.utils.vdb.load_vec_db_conn")
def test_initialize_vector_db_not_exists(mock_load_conn):
    mock_client = MagicMock()
    mock_client.collection_exists.return_value = False
    mock_load_conn.return_value = mock_client

    initialize_vector_db("test_coll", 1024)

    mock_client.collection_exists.assert_called_with(collection_name="test_coll")
    mock_client.create_collection.assert_called_once()


@patch("agent.utils.vdb.QdrantVectorStore")
@patch("agent.utils.vdb.sparse_embeddings")
def test_init_vdb(mock_sparse, mock_vstore):
    mock_embedding = MagicMock()

    init_vdb("test_coll", mock_embedding)

    mock_vstore.assert_called_once()
    _, kwargs = mock_vstore.call_args
    assert kwargs["collection_name"] == "test_coll"
    assert kwargs["embedding"] == mock_embedding
    assert kwargs["sparse_vector_name"] == "bge-m3-sparse"


@patch("agent.utils.vdb.initialize_vector_db")
def test_initialize_all_vector_dbs(mock_init_vdb):
    config = Config()
    initialize_all_vector_dbs(config)
    mock_init_vdb.assert_called_once()
