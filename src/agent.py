from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.agent import (
    CustomSimpleAgentWorker,
    Task,
    AgentChatResponse,
)
from llama_index.core.bridge.pydantic import Field, BaseModel
from typing import Dict, Any, List, Tuple, Optional
from llama_index.core.tools import BaseTool, QueryEngineTool
from llama_index.core.query_engine import RouterQueryEngine
from prompt_const import DEFAULT_PROMPT_STR, get_chat_prompt_template
from llama_index.core.selectors import PydanticSingleSelector
from llama_index.core.program import LLMTextCompletionProgram
from llama_index.core.schema import NodeWithScore

# Retrievers
from llama_index.core.retrievers import (
    BaseRetriever,
    VectorIndexRetriever,
    KeywordTableSimpleRetriever,
)
from llama_index.core import QueryBundle

class ResponseEval(BaseModel):
    """Evaluation of whether the response has an error."""

    has_error: bool = Field(..., description="Whether the response has an error.")
    new_question: str = Field(..., description="The suggested new question.")
    explanation: str = Field(
        ...,
        description=("The explanation for the error as well as for the new question." "Can include the direct stack trace as well."),
    )

class CustomRetriever(BaseRetriever):
    """Custom retriever that performs both semantic search and hybrid search."""

    def __init__(
        self,
        vector_retriever: VectorIndexRetriever,
        keyword_retriever: KeywordTableSimpleRetriever,
        mode: str = "AND",
    ) -> None:
        """Init params."""

        self._vector_retriever = vector_retriever
        self._keyword_retriever = keyword_retriever
        if mode not in ("AND", "OR"):
            raise ValueError("Invalid mode.")
        self._mode = mode
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve nodes given query."""

        vector_nodes = self._vector_retriever.retrieve(query_bundle)
        keyword_nodes = self._keyword_retriever.retrieve(query_bundle)

        vector_ids = {n.node.node_id for n in vector_nodes}
        keyword_ids = {n.node.node_id for n in keyword_nodes}

        combined_dict = {n.node.node_id: n for n in vector_nodes}
        combined_dict.update({n.node.node_id: n for n in keyword_nodes})

        if self._mode == "AND":
            retrieve_ids = vector_ids.intersection(keyword_ids)
        else:
            retrieve_ids = vector_ids.union(keyword_ids)

        retrieve_nodes = [combined_dict[rid] for rid in retrieve_ids]
        return retrieve_nodes
    
class RetryAgentWorker(CustomSimpleAgentWorker):
    """Agent worker that adds a retry layer on top of a router.

    Continues iterating until there's no errors / task is done.

    """

    prompt_str: str = Field(default=DEFAULT_PROMPT_STR)
    max_iterations: int = Field(default=10)

    _router_query_engine: RouterQueryEngine = PrivateAttr()

    def __init__(self, tools: List[BaseTool], **kwargs: Any) -> None:
        """Init params."""
        # validate that all tools are query engine tools
        for tool in tools:
            if not isinstance(tool, QueryEngineTool):
                raise ValueError(f"Tool {tool.metadata.name} is not a query engine tool.")
        self._router_query_engine = RouterQueryEngine(
            selector=PydanticSingleSelector.from_defaults(),
            query_engine_tools=tools,
            verbose=kwargs.get("verbose", False),
        )
        super().__init__(
            tools=tools,
            **kwargs,
        )

    def _initialize_state(self, task: Task, **kwargs: Any) -> Dict[str, Any]:
        """Initialize state."""
        return {"count": 0, "current_reasoning": []}

    def _run_step(self, state: Dict[str, Any], task: Task, input: Optional[str] = None) -> Tuple[AgentChatResponse, bool]:
        """Run step.

        Returns:
            Tuple of (agent_response, is_done)

        """
        if "new_input" not in state:
            new_input = task.input
        else:
            new_input = state["new_input"]

        # first run router query engine
        response = self._router_query_engine.query(new_input)

        # append to current reasoning
        state["current_reasoning"].extend([("user", new_input), ("assistant", str(response))])

        # Then, check for errors
        # dynamically create pydantic program for structured output extraction based on template
        chat_prompt_tmpl = get_chat_prompt_template(self.prompt_str, state["current_reasoning"])
        llm_program = LLMTextCompletionProgram.from_defaults(
            # output_parser=PydanticOutputParser(output_cls=ResponseEval),
            prompt=chat_prompt_tmpl,
            llm=self.llm,
        )
        # run program, look at the result
        response_eval = llm_program(query_str=new_input, response_str=str(response))
        if not response_eval.has_error:
            is_done = True
        else:
            is_done = False
        state["new_input"] = response_eval.new_question

        if self.verbose:
            print(f"> Question: {new_input}")
            print(f"> Response: {response}")
            print(f"> Response eval: {response_eval.dict()}")

        # return response
        return AgentChatResponse(response=str(response)), is_done

    def _finalize_task(self, state: Dict[str, Any], **kwargs) -> None:
        """Finalize task."""
        # nothing to finalize here
        # this is usually if you want to modify any sort of
        # internal state beyond what is set in `_initialize_state`
        pass
