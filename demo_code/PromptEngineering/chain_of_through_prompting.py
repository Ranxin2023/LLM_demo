from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
load_dotenv()
# Initialize model
llm = ChatOpenAI(model="gpt-4o-mini")

# Define a Chain-of-Thought style prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful reasoning assistant. Always explain your reasoning step-by-step."),
    ("human", "If a car travels 60 miles in 1.5 hours, what is its average speed in miles per hour?")
])

# Invoke the model
response = (prompt | llm).invoke({})

print(response.content)
