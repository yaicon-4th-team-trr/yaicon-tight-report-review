import torch
import transformers
import numpy as np
from os import path as osp
from enum import Enum
from typing import List
from argparse import ArgumentParser
from transformers import AutoTokenizer
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, DocumentSummaryIndex
from prompt_const import SYSTEMS as SYSTEM_PROMPTS
from search_via_google import search
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.query_engine import RetrieverQueryEngine

HF_TOKEN = "" # your huggingface token
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

def setup_query_engine(index_a, index_b):
    return RetrieverQueryEngine(indices={"A": index_a, "B": index_b})

def retrieve_context(query_engine, document, index_name):
    return query_engine.query(f"related documents for {document.content}", index=index_name)

def query_documents(query_engine, query):
    response = query_engine.query(query)
    return response

def get_documents_content(documents):
    content = "\n\n".join([doc.get_text() for doc in documents])
    return content

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
        max_new_tokens=1024,
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
        stopping_ids=stopping_ids
    )
    # pipeline = transformers.pipeline("text-generation",
    #                                  model=args.model,
    #                                  model_kwargs={"torch_dtype": torch.bfloat16,},
    #                                  max_length=100000,
    #                                  device_map="auto")
    
    embed_model = HuggingFaceEmbedding(model_name=args.embed_model) # for llama-index
    # embed_model = OllamaEmbedding(model_name=args.embed_model) # for llama-index

    Settings.embed_model = embed_model
    Settings.llm = llm

    # Load documents # TODO: more efficient way to load documents
    external_documents_web = load_external_documents(osp.join("data"), args.keywords)

    external_documents_paper = {}
    internal_documents_user_reports = {}
    for key in args.keywords:
        external_documents_paper[key] = SimpleDirectoryReader(osp.join("papers", key)).load_data()
        internal_documents_user_reports[key]=SimpleDirectoryReader(osp.join("reports", key)).load_data()

    # Index external documents
    for i, key in enumerate(args.keywords):
        for i, SYSTEM_PROMPT in enumerate(SYSTEM_PROMPTS):
            user_report_content = DocumentSummaryIndex.from_documents(internal_documents_user_reports[key], llm=llm)
            external_web_report_content = DocumentSummaryIndex.from_documents(external_documents_web[key], llm=llm)
            paper_content = DocumentSummaryIndex.from_documents(external_documents_paper[key], llm=llm)

            user_report_content = get_documents_content(internal_documents_user_reports[key])
            external_web_report_content = get_documents_content(external_documents_web[key])
            paper_content = get_documents_content(external_documents_paper[key])

            if 'web' in SYSTEM_PROMPT:
                response = pipeline(SYSTEM_PROMPT.format(student=user_report_content, web=external_web_report_content))
            else:
                response = pipeline(SYSTEM_PROMPT.format(student=user_report_content, paper=paper_content))

            print(response)
            with open(f"output_{i}.txt", "a") as f:
                f.write(response)

if __name__ == "__main__":
    args = setup_args()
    main(args)
