"""Grading node for the graph."""

from typing import Literal

from langchain_core.language_models import LanguageModelLike
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableConfig

from agent.backend.prompts import GRADER_TEMPLATE
from agent.backend.state import AgentState, Grade


def grade_documents(
    state: AgentState,
    config: RunnableConfig,  # noqa: ARG001
    *,
    llm: LanguageModelLike,
) -> Literal["response_synthesizer", "rewrite_query"]:
    """Grade the retrieved documents holistically.

    Returns either:
        - "response_synthesizer" if the documents are relevant (or retry limit reached)
        - "rewrite_query" if the documents are not relevant and we should retry with a rewritten query
    """
    from langchain_core.output_parsers import PydanticOutputParser

    model = llm.with_config(stream=False)
    parser = PydanticOutputParser(pydantic_object=Grade)

    prompt = PromptTemplate(
        template=GRADER_TEMPLATE + "\n\n{format_instructions}",
        input_variables=["documents", "question"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    chain = prompt | model | parser

    docs_text = "\n\n".join([f"Document {i + 1}:\n{doc.page_content}" for i, doc in enumerate(state["documents"])])

    try:
        grade: Grade = chain.invoke({"documents": docs_text, "question": state["query"]})
    except Exception as e:
        import logging

        logging.warning(f"Failed to parse grade, defaulting to relevant: {e}")
        grade = Grade(is_relevant=True)

    if grade.is_relevant or state.get("retry_count", 0) >= 2:
        return "response_synthesizer"

    return "rewrite_query"
