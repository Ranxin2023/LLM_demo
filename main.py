from demo_code.LLM_Evaluation import evaluate_LLM
from demo_code.reduce_computational_cost import five_method
from demo_code.output_control import output_control_demo
from demo_code.pre_trained_objectives import pre_trained_demo
from demo_code.context_window_demo import context_window_demo
# from demo_code.fine_tune_demo import fine_tune_demo
from demo_code.oov_demo import oov_demo
from demo_code.Agent_demo.simple_version import simple_version
def main():
    # evaluate_LLM()
    # five_method()
    # output_control_demo()
    # pre_trained_demo()
    # context_window_demo()
    # fine_tune_demo()
    # oov_demo()
    simple_version()

if __name__=='__main__':
    main()