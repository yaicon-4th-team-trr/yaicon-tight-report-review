import torch
from os import path as osp
from argparse import ArgumentParser
from transformers import AutoTokenizer
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core.llms import ChatMessage
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, StorageContext
from prompt_const import SYSTEM as SYSTEM_PROMPT
from search_via_google import search
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from agent import RetryAgentWorker
from llama_index.core.tools import QueryEngineTool


HF_TOKEN = "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" # seil kang

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

    return external_docs

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

    Settings.embed_model = embed_model
    Settings.llm = llm

    # Load documents # TODO: more efficient way to load documents
    external_documents = load_external_documents(osp.join("data"), args.keywords)  
    internal_documents_paper = SimpleDirectoryReader(osp.join("papers")).load_data()
    internal_documents_user_reports = {}
    for key in args.keywords:
        internal_documents_user_reports['key']=SimpleDirectoryReader(osp.join("reports", key)).load_data()

    # Index documents
    vector_tools = {}
    callback_manager = llm.callback_manager
    agents = {}
    index_set = []
    for i, (key, doc) in enumerate(zip(args.keywords, external_documents)):
        external_documents_by_keyword = external_documents[key]
        storage_context = StorageContext.from_defaults()
        for i, doc_by_keyword in enumerate(external_documents_by_keyword):    
            vector_index = VectorStoreIndex.from_documents([doc_by_keyword], storage_context=storage_context)
            index_set.append(vector_index)
            storage_context.persist(persist_dir=f"./storage/{key}/{i}")

    for i, (key, doc) in enumerate(zip(args.keywords, external_documents)):
        vector_query_engine = vector_index.as_query_engine()  # you can set similarity_top_k here
        vector_tool = QueryEngineTool.from_defaults(query_engine=vector_query_engine)
        vector_tools[key] = vector_tool

        # build custom agent
        query_engine_tools = vector_tools
        agent_worker = RetryAgentWorker.from_tools(
            query_engine_tools[key],
            llm=llm,
            verbose=True,
            callback_manager=callback_manager,
        )

        agents[key] = agent_worker.as_agent(callback_manager=callback_manager)

        response = agents[key].chat("ViT는 무엇인가?")
        print(str(response))


if __name__ == "__main__":
    args = setup_args()
    main(args)
