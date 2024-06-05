from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core import ChatPromptTemplate, PromptTemplate
from typing import Dict, Any, List, Tuple, Optional

SYSTEM_1 = """
I would like to entrust you with evaluating a report written by a college student as an assignment. The student read a paper, summarized what he studied, and wrote a report. You will be given one student's report and one paper the student read. Your job is to read each student report and paper carefully and rate how good the student report is with a number between 0 and 10 for a criterion. Here, a good report has a main criterion. Please rate each with a number between 0 and 10.
---
Evaluation criterion is here:
Copy and paste: If you think that the student report is taking sentences from the paper as is (i.e., sentences that are not recreated in the student's own language are often found), you should give it a low score. Conversely, if the student explains the content of the paper in his or her own language, a high score should be given.
---
Evaluation steps are here:
1. Read the student report first and compare each sentence with the original paper.
2. Determine whether the part in the student report is taken directly from the original paper or not. Scores for Creativity1 are reasonably awarded between 0 and 10, using the proportion of parts taken from the original sentence.
3. In the case where the original sentence is not taken as is, determine whether it is a sentence that expresses the content of the paper in your own language or a sentence that expresses your subjective thoughts about the content. 
4. Through this, a score for Copy and paste is given between 0 and 10. And print out the score right under the last line in the following form. "Score: {{score value}}"
---
report: {student}
paper: {paper}
"""

SYSTEM_2 = """
I would like to entrust you with evaluating a report written by a college student as an assignment. The student read a paper, summarized what he studied, and wrote a report. You will be given one student's report and one paper the student read. Your job is to read each student report and paper carefully and rate how good the student report is with a number between 0 and 10 for a criterion. Here, a good report has a main criterion. Please rate each with a number between 0 and 10.
---
Evaluation criterion is here:
Creativity: High scores should be given, especially if the paper contains your own subjective evaluation or thoughts about the content. If not, I have to give it a low score
---
Evaluation steps are here:
1. Read the student report first and compare each sentence with the original paper.
2. Determine whether the part in the student report is taken directly from the original paper or not. Scores for Creativity1 are reasonably awarded between 0 and 10, using the proportion of parts taken from the original sentence.
3. In the case where the original sentence is not taken as is, determine whether it is a sentence that expresses the content of the paper in your own language or a sentence that expresses your subjective thoughts about the content. 
4. Through this, a score for Creativity is given between 0 and 10. And print out the score right under the last line in the following form. "Score: {{score value}}"
---
report: {student}
paper: {paper}
"""

SYSTEM_3 = """
I would like to entrust you with evaluating a report written by a college student as an assignment. The student read a paper, summarized what he studied, and wrote a report. You will be given one student's report and one paper the student read. Your job is to read each student report and paper carefully and rate how good the student report is with a number between 0 and 10 for a criterion. Here, a good report has a main criterion. Please rate each with a number between 0 and 10.
---
Evaluation criterion is here:
Perfection: A good grade should be given when a student report summarizes the overall content of the paper well. Conversely, if important information is omitted, a low score should be given. The important content here could simply be a keyword that is used repeatedly in the paper, or it could be a part that you consider important when you think about it yourself. Ultimately, this means that if you can understand the core content of the paper just by looking at the student report, you should give it a high score.
---
Evaluation steps are here:
1. Read the student report first and compare each sentence with the original paper.
2. Determine whether the part in the student report is taken directly from the original paper or not. Scores for Creativity1 are reasonably awarded between 0 and 10, using the proportion of parts taken from the original sentence.
3. In the case where the original sentence is not taken as is, determine whether it is a sentence that expresses the content of the paper in your own language or a sentence that expresses your subjective thoughts about the content. 
4. Through this, a score for Perfection is given between 0 and 10. And print out the score right under the last line in the following form. "Score: {{score value}}"
---
report: {student}
paper: {paper}
"""

SYSTEM_4 = """
I would like to entrust you with evaluating a report written by a college student as an assignment. The student read a paper, summarized what he studied, and wrote a report. You will be given a student's report, a web report, and one paper. Both the user report and the web report are reviews of a single paper. Your job is to read each student report and paper carefully and rate how good the student report is with a number between 0 and 10 for a criterion. Here, a good report has a main criterion. Please rate each with a number between 0 and 10.
---
Evaluation criterion is here:
Scissiors-and-paste: If the user report and the web report are similar to each other, then the final score of the user report should be low. The user report and the web report are the reviews from the single papaer, so if the topics of them are completely different, then consider the user report bad. Sometimes the topic of the web report is completely different from the content of the paper, in this case the content of the user report and the web report might differ. 
---
Evaluation steps are here:
1. Check the user report. Read it and briefly explain what the user report is about.
2. Check the web report. Read it and briefly explain what the web report is about.
3. Check the original paper.
4. Compare the user report and the web report, based on the given evaluation criteria. For each criterion, briefly explain whether the user report is more or less likely to receive a high score. If you can identify the sources that replicated the user report, then please provide an  analysis of how they match up with the content of the user report.
5. Through this, a score for Scissiors-and-paste is given between 0 and 10. And print out the score right under the last line in the following form. "Score: {{score value}}"
---
report: {student}
web: {web}
"""

_SYSTEM = """
You are a helpful and precise assistant for evaluating how good the user report is. With carefully following the instructions below, comparing the user report and the web report, you have to evaluate how good the user report is. If needed, you should also refer to the paper. The criteria for determining the score is given below.

Evaluation Criteria:
Similarity: If the user report and the web report are similar to each other, then the final score of the user report should be low. To be a good user report, it must contain the writer's own insight, not the copy of the web report.
Consistency: If the topics of the user report and the web report are completely different, than consider the report bad. Note that the user report and the web report are the reviews from the single paper, so their topic must be not much different.
Relevance: Sometimes there are some noise in the web report, that is, the topic of the web report is completely different from the content of the paper. However, the user report is always guarenteed to be a review of the paper. So in this case, the topic or content of the user report and the web report might differ.

Evaluation Steps:
1. Check the user report. Read it and briefly explain what the user report is about.
2. Check the web report. Read it and briefly explain what the web report is about.
3. Check the original paper.
4. Compare the user report and the web report, based on the given evaluation criteria. For each criterion, briefly explain whether the user report is more or less likely to receive a high score. If you can identify the sources that replicated the user report, then please provide an  analysis of how they match up with the content of the user report.
5. Referring the given reports, a paper, and what you have answered, print out a single line containing the final score for the user report.
6. Evaluate the user report score based on the comparison. Keep in mind that the score must be at least 0 and no more than 10.

Assistant output example:
Similarity: 7/10, Consistency: 3/10, Relevance: 5/10
{{Explain why the user report is likely to receive this score.}}
"""

SYSTEMS = [SYSTEM_1, SYSTEM_2, SYSTEM_3, SYSTEM_4]

SYSTEM_v0 = "You are a specialized AI agent for reviewing reports, focusing on evaluating user reports based on a specific paper." 
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
