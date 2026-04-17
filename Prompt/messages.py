from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage,HumanMessage,AIMessage

load_dotenv()

model=ChatOpenAI()

messages=[
    SystemMessage(content='You are a helpful assistant'),
    HumanMessage(content='Tell me indian cricket'),

]

result=model.invoke(messages)

messages.append(AIMessage(content=result.content))

print(messages)