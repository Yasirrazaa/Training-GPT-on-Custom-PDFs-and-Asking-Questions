from dotenv import load_dotenv
import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback



load_dotenv()

st.set_page_config(page_title='LangChain', page_icon='🔗', layout='centered')
st.header('PDF GPT')
st.write(os.getenv('OPENAI_API_KEY'))
pdf=st.file_uploader('Upload a PDF file', type=['pdf'])
if pdf is not None:
    pdf_read=PdfReader(pdf)
    text=""
    for page in pdf_read.pages:
        text=text+page.extract_text()
    text_splitter=CharacterTextSplitter(separator="\n", chunk_size=1000,chunk_overlap=200,length_function=len)
    chunks=text_splitter.split_text(text)
    embeddings=OpenAIEmbeddings()
    knowledge_base=FAISS.from_texts(chunks,embeddings)
    user_query=st.text_input('Ask a question about your PDF')
    if user_query:
        docs=knowledge_base.similarity_search(user_query, k=1)
        llm=OpenAI()
        chain=load_qa_chain(llm,chain_type="stuff")
        with get_openai_callback() as cb:
            response=chain.run(input_documents=docs, question=user_query)
            print(cb)
        st.write(response)




