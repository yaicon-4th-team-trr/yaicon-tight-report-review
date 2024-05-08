import torch
import numpy as np
from os import path as osp
from enum import Enum
from typing import List
from argparse import ArgumentParser
from transformers import AutoTokenizer
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, StorageContext, PromptTemplate, get_response_synthesizer
from prompt_const import SYSTEM as SYSTEM_PROMPT
from search_via_google import search
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor

HF_TOKEN = "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" # seil kang
class SimilarityMode(str, Enum):
    """Modes for similarity/distance."""

    DEFAULT = "cosine"
    DOT_PRODUCT = "dot_product"
    EUCLIDEAN = "euclidean"


def setup_args():
    parser = ArgumentParser() 
    parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--embed-model", type=str, default="BAAI/bge-m3")
    parser.add_argument("--temp", type=float, default=0.6)
    parser.add_argument("--topp", type=float, default=0.9)
    parser.add_argument("-k", "--keywords", nargs="+", help="keyword list", required=True)

    return parser.parse_args()

def load_external_documents(input_dir=None, input_keywords=None) -> dict:

    external_docs = {}
    for key in input_keywords:
        search_result_docs = search(key) # editable
        external_docs[key] = SimpleDirectoryReader(input_dir=f"data/{key}").load_data()
        for doc in external_docs[key]:
            doc.metadata = {"source": "web",
                            "name": "web search results",
                            "description": "Web search results for the paper-reviews"
                            }
    return external_docs

def similarity(
    embedding1: List[float],
    embedding2: List[float],
    mode: SimilarityMode = SimilarityMode.DEFAULT,
) -> float:

    """Get embedding similarity."""
    if mode == SimilarityMode.EUCLIDEAN:
        # Using -euclidean distance as similarity to achieve same ranking order
        return -float(np.linalg.norm(np.array(embedding1) - np.array(embedding2)))
    elif mode == SimilarityMode.DOT_PRODUCT:
        return np.dot(embedding1, embedding2)
    else:
        product = np.dot(embedding1, embedding2)
        norm = np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
        return product / norm


def evaluate_similarity(embed_model, ex_docs_web, ex_docs_paper, int_docs_user_reports):

    score = []
    for doc_web in ex_docs_web:
        user_report = "\n".join([user_report.text for user_report in int_docs_user_reports])
        reference = doc_web.text

        embed_user_report = embed_model.get_text_embedding(user_report)
        embed_reference = embed_model.get_text_embedding(reference)

        result = similarity(embed_user_report, embed_reference)
        score.append(round(result, 3))
    return np.mean(score)

def main(args):

    # set LLM, Embedding model

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        token=HF_TOKEN,
    )

    stopping_ids = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>"),
    ]

    llm = HuggingFaceLLM(
        model_name=args.model,
        model_kwargs={
            "token": HF_TOKEN,
            "torch_dtype": torch.bfloat16,  # comment this line and uncomment below to use 4bit
            # "quantization_config": quantization_config
        },
        generate_kwargs={
            "do_sample": True,
            "temperature": args.temp,
            "top_p": args.topp,
        },
        tokenizer_name=args.model,
        tokenizer_kwargs={"token": HF_TOKEN},
        stopping_ids=stopping_ids,
    )

    embed_model = HuggingFaceEmbedding(model_name=args.embed_model) # for llama-index
    # embed_model = OllamaEmbedding(model_name=args.embed_model) # for llama-index

    Settings.embed_model = embed_model
    Settings.llm = llm

    # Load documents # TODO: more efficient way to load documents
    external_documents_web = load_external_documents(osp.join("data"), args.keywords)

    external_documents_paper = {}
    for key in args.keywords:
        external_documents_paper[key] = SimpleDirectoryReader(osp.join("papers", key)).load_data()
        for doc in external_documents_paper[key]:
            doc.metadata = {"source": "paper",
                            "name": "paper",
                            "description": "papers of the paper-reviews"
                            }

    internal_documents_user_reports = {}
    for key in args.keywords:
        internal_documents_user_reports[key]=SimpleDirectoryReader(osp.join("reports", key)).load_data()
        for doc in internal_documents_user_reports[key]:
            doc.metadata = {"source": "user",
                            "name": "user report",
                            "description": "user reports of the paper-reviews"
                            }
    # for key in args.keywords:
    #     evaluate_similarity(
    #         embed_model=embed_model,
    #         ex_docs_web=external_documents_web[key],
    #         ex_docs_paper=external_documents_paper[key],
    #         int_docs_user_reports=internal_documents_user_reports[key],
    #     )

    # Index external documents
    callback_manager = llm.callback_manager
    for i, key in enumerate(args.keywords):
        total_documents = external_documents_web[key] + external_documents_paper[key] + internal_documents_user_reports[key]
        storage_context = StorageContext.from_defaults()

        vector_indices = VectorStoreIndex.from_documents(total_documents, storage_context=storage_context)
        storage_context.persist(persist_dir=f"./storage/{key}/paper")

        retriever = VectorIndexRetriever(
            index=vector_indices,
            similarity_top_k =5,
        )

        response_synthesizer = get_response_synthesizer()

        query_engine = RetrieverQueryEngine(
            retriever=retriever,
            response_synthesizer=response_synthesizer,
            node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.5)],
        )

        # ====== Customise prompt template ======
        qa_prompt_tmpl_str = (
            "Context information is below.\n"
            "---------------------\n"
            f"{SYSTEM_PROMPT}\n"
            "---------------------\n"
            "Query: {query_str}\n"
            "Answer: "
        )
        qa_prompt_tmpl = PromptTemplate(qa_prompt_tmpl_str)

        query_engine.update_prompts({"response_synthesizer:text_qa_template": qa_prompt_tmpl})

        # Generate the response
        response = query_engine.query(
            "Explain how similar the user report is to the original content of the paper, and compare it with the results of a web search.",
        )

        print(response)

if __name__ == "__main__":
    args = setup_args()
    main(args)
