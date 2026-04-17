from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

em=OpenAIEmbeddings(model='text-embedding-3-large',dimensions=300)

documents = [
    "Virat Kohli is known for his aggressive batting and consistency in international cricket.",
    "Rohit Sharma is famous for his elegant stroke play and multiple double centuries in ODIs.",
    "MS Dhoni is one of the greatest captains and is known for his calm finishing ability.",
    "Sachin Tendulkar is called the God of Cricket and holds numerous batting records.",
    "Jasprit Bumrah is a world-class fast bowler known for his unique bowling action and deadly yorkers."
]

query='tell me about virat kohli'

doc=em.embed_documents(documents)
qu=em.embed_query(query)

print(cosine_similarity([qu],doc))
