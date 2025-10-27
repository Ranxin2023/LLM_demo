# from demo_code.LLM_Evaluation import evaluate_LLM
# from demo_code.reduce_computational_cost import five_method
# from demo_code.output_control import output_control_demo
# from demo_code.pre_trained_objectives import pre_trained_demo
# from demo_code.context_window_demo import context_window_demo
# # from demo_code.fine_tune_demo import fine_tune_demo
# from demo_code.oov_demo import oov_demo
# from demo_code.Agent_demo.simple_version import simple_version
# from demo_code.LangChain.agent_demo import agent_demo_redirect
# from demo_code.LangChain.langchain_agent2 import langchain2_redirect
# from demo_code.LLMConceptsDemo.mitigate_bias import mitigate_bias_output
from demo_code.FineTuning.catastrophic_forgetting import catastrophic_forgetting_redirect
from demo_code.RAG.AgenticRAGDemo import run_agentic_RAG
from demo_code.LLMConceptsDemo.CoTDemo import CoT_redirect_output
from demo_code.KnowledgeDistillation.KnowledgeDistillation import knowledge_distillation_redirect
def main():
    # evaluate_LLM()
    # five_method()
    # output_control_demo()
    # pre_trained_demo()
    # context_window_demo()
    # fine_tune_demo()
    # oov_demo()
    # simple_version()
    # agent_demo_redirect()
    # langchain2_redirect()
    # mitigate_bias_output()
    # catastrophic_forgetting_redirect()
    # run_agentic_RAG()
    # CoT_redirect_output()
    knowledge_distillation_redirect()
if __name__=='__main__':
    main()