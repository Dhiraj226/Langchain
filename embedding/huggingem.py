from langchain_huggingface import HuggingFaceEmbeddings

em=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

text='Modi is the prime minister of india'

result=em.embed_query(text)

print(result)

