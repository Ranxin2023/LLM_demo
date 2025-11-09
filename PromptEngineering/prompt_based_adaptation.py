from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from contextlib import redirect_stdout
from dotenv import load_dotenv
load_dotenv()
# Initialize model
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)

# -------------------------------
#  Domain-specific prompt templates
# -------------------------------
domain_prompts = {
    "Finance": ChatPromptTemplate.from_messages([
        ("system", "You are a senior financial analyst specializing in global markets. "
                   "Always provide factual and data-driven insights in a concise tone."),
        ("human", "Analyze the current market trend of renewable energy stocks and suggest key investment drivers.")
    ]),

    "Medical": ChatPromptTemplate.from_messages([
        ("system", "You are a medical research assistant with expertise in clinical studies and biotechnology. "
                   "Your tone should be formal, precise, and supported by medical reasoning."),
        ("human", "Summarize recent developments in Alzheimer’s research and their implications for treatment.")
    ]),

    "Cybersecurity": ChatPromptTemplate.from_messages([
        ("system", "You are a cybersecurity consultant helping organizations detect and prevent data breaches. "
                   "Provide structured, actionable insights in bullet points."),
        ("human", "Identify key vulnerabilities in cloud-based systems and recommend best security practices.")
    ])
}

# -------------------------------
# Generate domain-specific outputs
# -------------------------------
def domain_specific_prompt():
    for domain, prompt in domain_prompts.items():
        print(f"\n🌐 {domain} Domain Response:")
        response = llm.invoke(prompt.format_messages())
        print(response.content)

def redirect_adaption_output():
    with open("./output_results/PromptEngineering/prompt_based_adaptation.txt", "w", encoding="utf-8") as f:
        with redirect_stdout(f):
            domain_specific_prompt()