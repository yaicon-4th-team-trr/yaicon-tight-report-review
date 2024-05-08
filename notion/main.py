import os

from dotenv import dotenv_values
from notion_client import Client
from pprint import pprint
import logging
from json import dumps
from tqdm import tqdm

"""
["type"] : image, table_of_contents, paragraph, heading_1
["has_children"] : True / False
["(type name)"]["rich_text"][0]["plain_text"]
"""

# 로그 파일 설정
logging.basicConfig(
    filename='app.log',   # 로그 파일명
    filemode='w',         # 파일 모드 ('a'는 추가, 'w'는 쓰기)
    level=logging.ERROR,  # 로깅 레벨
    format='%(asctime)s - %(levelname)s - %(message)s'  # 로그 메시지 포맷
)

def save_content_to_file(page_content, filename="page_content.txt"):
    """페이지 내용을 텍스트 파일로 저장하는 함수"""
    try:
        with open(filename, "w", encoding="utf-8") as file:
            # 파일에 내용을 작성합니다.
            file.write(str(page_content))
        # print(f"Content saved to {filename}")
    except Exception as e:
        print(f"Failed to save content to file: {e}")


def pages_from_database(notion, database_id):
    query = {
        "database_id": database_id,
        "page_size": 100,
    }

    result = []
    has_more = True
    while has_more:
        response = notion.databases.query(**query)
        result.extend(response['results'])
        has_more = response['has_more']
        if has_more:
            query['start_cursor'] = response['next_cursor']

    return result


def content_from_page(notion, page_id):
    """
    page_id -> string of content

    @param notion:
    @param page_id: page id
    @return: string of page content
    """
    # 페이지 객체를 가져옵니다.
    page = notion.pages.retrieve(page_id=page_id)
    # 페이지 내용을 파싱하여 텍스트 형식으로 변환합니다.

    blocks = notion.blocks.children.list(block_id=page_id).get("results", [])

    res = ""
    for block in blocks:
        res += text_from_block(notion, block)

    return res


def text_from_block(notion, block):
    """
    block object -> plain text string

    @param notion:
    @param block: block object of Notion API
    @return: String object of concatenated 'plain_text'
    """
    ret = ""
    block_type = block['type']
    if "rich_text" not in block[block_type]:
        log_message = f"no rich_text found. object type : {block_type}\n"
        log_message += dumps(block)
        logging.error(log_message)
        # print(type(block))
    else:
        for text in block[block_type]["rich_text"]:
            ret += text["plain_text"] + '\n'

    if block["has_children"]:
        blocks = notion.blocks.children.list(block_id=block["id"]).get("results", [])
        for block in blocks:
            ret += text_from_block(notion, block)

    return ret

def page_ids_from_pages(pages):
    return [page['id'] for page in pages]


if __name__ == "__main__":
    config = dotenv_values(".env")
    notion_secret = config.get('NOTION_TOKEN')
    notion_agent = Client(auth=notion_secret)

    folder_path = "reports"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    DB_id = "d6b1595a-d5e6-41c7-b48a-353730fea35d"
    try:
        pages = pages_from_database(notion_agent, DB_id)
        ids = page_ids_from_pages(pages)    # page ids

        for i in tqdm(ids, desc="Processing"):
            content = content_from_page(notion_agent, i)
            save_content_to_file(content, filename=os.path.join(folder_path, f'{i}.txt'))

    except Exception as e:
        raise e
        # print(f"An error occurred: {e}")
