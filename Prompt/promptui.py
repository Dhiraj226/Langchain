from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
load_dotenv()

model=ChatOpenAI(model='gpt-4')

model1=HuggingFaceEndpoint(
    repo_id='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
    task='text-generation'
                    
                    )
model2=ChatHuggingFace(llm=model1)

from langchain_core.prompts import PromptTemplate

import streamlit as st

st.header('Research Tool')



paper_input=st.selectbox('Select ResearchcPaper Name',
                         ['Support-Vector Networks ','Attention Is All You Need','Adam: A Method for Stochastic Optimization (Kingma & Ba, 2014) - Popular optimize',
                         'Dropout: A Simple Way to Prevent Neural Networks from Overfitting (Srivastava et al., 2014)'])

style_input=st.selectbox('Select Explaination',
                         ['Beginner-Friendly','Technical Oriented','Mathematical'])

length_input=st.selectbox('select length',['Short(1-2 Paragraph)','Medium(3-5 Paragraph)','long(more than 5 Paragraph)'])

temp=PromptTemplate(
    template='''
PLease summarize the {paper_input} with specification:
explanation style: {style_input}
explaination length : {length_input}

Ensure that summary must be clear and accurate aligned with provided length and style
''',
input_variables=['paper_input','style_input','length_input']
)

te=temp.invoke({
    'paper_input':paper_input,
    'style_input':style_input,
    'length_input':length_input
})

if st.button('summarize'):
    result=model.invoke(te)
    st.write(result.content)

