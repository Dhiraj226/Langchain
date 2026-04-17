from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

em=OpenAIEmbeddings(model='text-embedding-3-large',dimensions=32)

documents=['Modi is the prime minister of india',
           'Who is the prime minister of india?',
           'The Prime minister of india is Modi',
           'I am modi'
]


result=em.embed_documents(documents)

print(result)

