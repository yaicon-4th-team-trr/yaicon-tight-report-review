from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core import ChatPromptTemplate, PromptTemplate
from typing import Dict, Any, List, Tuple, Optional

SYSTEM = "You are a specialized AI agent for reviewing reports, focusing on evaluating user reports based on a specific paper." 
"When reviewing reports, you should verify whether the report 'borrows' external materials extensively and accurately writes down the contents of the paper."
"If it borrows many external materials and accurately writes down the contents of the paper, then consider the report to be uncreative."
"Evaluate user reports based on the following items and provide an answer with the results."
"Follow step by step to evaluate the user report."
"1. Check the user report."
"2. Check the original paper."
"3. Check the external materials. (web search)"
"4. Compare the user report with the original paper."
"5. Compare the user report with the external web search materials."
"6. Evaluate the user report based on the comparison."
"7. If you can identify the sources that replicated the user report, then please provide a detailed analysis of how they match up with the content of the user report."

DEFAULT_PROMPT_STR = """
Given previous question/response pairs, please determine if an error has occurred in the response, and suggest \
    a modified question that will not trigger the error.

Examples of modified questions:
- The question itself is modified to elicit a non-erroneous response
- The question is augmented with context that will help the downstream system better answer the question.
- The question is augmented with examples of negative responses, or other negative questions.

An error means that either an exception has triggered, or the response is completely irrelevant to the question.

Please return the evaluation of the response in the following JSON format.
"""


def get_chat_prompt_template(system_prompt: str, current_reasoning: Tuple[str, str]) -> ChatPromptTemplate:
    system_msg = ChatMessage(role=MessageRole.SYSTEM, content=system_prompt)
    messages = [system_msg]
    for raw_msg in current_reasoning:
        if raw_msg[0] == "user":
            messages.append(ChatMessage(role=MessageRole.USER, content=raw_msg[1]))
        else:
            messages.append(ChatMessage(role=MessageRole.ASSISTANT, content=raw_msg[1]))
    return ChatPromptTemplate(message_templates=messages)
