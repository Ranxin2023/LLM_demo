from typing import TypedDict
from chains.binary_questions import BINARY_QUESTION_CHAIN
from chains.escalation_check import ESCALATION_CHECK_CHAIN
from chains.notice_extraction import NOTICE_PARSER_CHAIN, NoticeEmailExtract
from langgraph.graph import END, START, StateGraph
from pydantic import EmailStr
from utils.graph_utils import create_legal_ticket, send_escalation_email
from utils.logging_config import LOGGER
def answer_follow_up_question_node(state: GraphState) -> GraphState:
    """Answer follow-up questions about the notice using
    BINARY_QUESTION_CHAIN"""
    if state["current_follow_up"]:
        question = state["current_follow_up"] + " " + state["notice_message"]
        answer = BINARY_QUESTION_CHAIN.invoke({"question": question})
        if state.get("follow_ups"):
            state["follow_ups"][state["current_follow_up"]] = answer
        else:
            state["follow_ups"] = {state["current_follow_up"]: answer}
    return state
def route_follow_up_edge(state: GraphState) -> str:
    """Determine whether a follow-up question is required"""
    if state.get("current_follow_up"):
        return "answer_follow_up_question"
    return END

def build_graph():

    workflow.add_node("parse_notice_message", parse_notice_message_node)
    workflow.add_node("check_escalation_status", check_escalation_status_node)
    workflow.add_node("send_escalation_email", send_escalation_email_node)
    workflow.add_node("create_legal_ticket", create_legal_ticket_node)
    workflow.add_node("answer_follow_up_question", answer_follow_up_question_node)

    workflow.add_edge(START, "parse_notice_message")
    workflow.add_edge("parse_notice_message", "check_escalation_status")
    workflow.add_conditional_edges(
        "check_escalation_status",
        route_escalation_status_edge,
        {
            "send_escalation_email": "send_escalation_email",
            "create_legal_ticket": "create_legal_ticket",
        },
    )
    workflow.add_conditional_edges(
        "create_legal_ticket",
        route_follow_up_edge,
        {
            "answer_follow_up_question": "answer_follow_up_question",
            END: END,
        },
    )

    workflow.add_edge("send_escalation_email", "create_legal_ticket")
    workflow.add_edge("answer_follow_up_question", "create_legal_ticket")

    NOTICE_EXTRACTION_GRAPH = workflow.compile()