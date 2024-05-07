from llama_index.tools.google import GoogleSearchToolSpec
from llama_index.readers.web import FireCrawlWebReader
from llama_index.readers.web import SimpleWebPageReader
from llama_index.core import SummaryIndex
import os 
KEY = "AIzaSyAdFNkEC7wpXJ74uNkXgQhV6gZcMB-LCBU" # seil kang
CX = "c716bd339a70e4907" # seil kang

def search(input_keyword):

    search_api = GoogleSearchToolSpec(key=KEY, engine=CX)

    result = search_api.google_search(f"{input_keyword} 논문리뷰")[0] # TODO: enable to control more search terms.
    result_dict=eval(result.get_content())
    urls = [item['link'] for item in result_dict['items']]
    docs = [] # by paper
    docs_txt = []
    for i, url in enumerate(urls):
        # document = firecrawl(url)
        document = SimpleWebPageReader(html_to_text=True).load_data([url])
        for doc in document:
            doc.metadata = {"num": i, "url": url}
        docs.append(document)
        docs_txt.append(document[0])
        os.makedirs(f"data/{input_keyword}", exist_ok=True)
        with open(f"data/{input_keyword}/output-{i}.txt", "w") as file:
            file.write(document[0].get_text())
    
    return docs
    # for doc in docs:
    #     # set Logging to DEBUG for more detailed outputs
    #     index = SummaryIndex.from_documents(doc)
    #     query_engine = index.as_query_engine()
    #     response = query_engine.query("ViT는 무엇인가?")
    #     print(response)

def firecrawl(url):
    # using firecrawl to crawl a website
    firecrawl_reader = FireCrawlWebReader(
        api_key="fc-cdd80d4511e949869b3cb12f6846eecb",  # Replace with your actual API key from https://www.firecrawl.dev/
        mode="scrape",  # Choose between "crawl" and "scrape" for single page scraping
        params={"additional": "parameters"},  # Optional additional parameters
    )

    # Load documents from a single page URL
    documents = firecrawl_reader.load_data(url=url)
    return documents 
