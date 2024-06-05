from llama_index.tools.google import GoogleSearchToolSpec
from llama_index.readers.web import FireCrawlWebReader
from llama_index.readers.web import SimpleWebPageReader
from llama_index.core import SummaryIndex
import os 
import shutil

KEY = "" # your google api key
CX = "" # yout google search engine id

def search(input_keyword):

    search_api = GoogleSearchToolSpec(key=KEY, engine=CX)
    num_search = 5
    result = search_api.google_search(f"{input_keyword} '설명'")[0] # TODO: enable to control more search terms.
    result_dict=eval(result.get_content())
    items = result_dict["items"][:num_search]

    urls = [item["link"] for item in items]
    docs = [] # by paper
    docs_txt = []
    if os.path.exists(f"data/{input_keyword}"):
        shutil.rmtree(f"data/{input_keyword}")
        
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
