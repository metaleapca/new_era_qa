import os

# Document Loaders
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import Docx2txtLoader
from langchain.document_loaders import TextLoader

# 1. Add documents
base_dir = './docs'
documents = []
for file in os.listdir(base_dir): 
    # Full path
    file_path = os.path.join(base_dir, file)
    if file.endswith('.pdf'):
        loader = PyPDFLoader(file_path)
        documents.extend(loader.load())
    elif file.endswith('.docx'): 
        loader = Docx2txtLoader(file_path)
        documents.extend(loader.load())
    elif file.endswith('.txt'):
        loader = TextLoader(file_path)
        documents.extend(loader.load())

# 2. Split documents into chunks
from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=10)
chunked_documents = text_splitter.split_documents(documents)

# 3. Store chunks in vector store
from langchain.vectorstores import Qdrant
from langchain.embeddings import OpenAIEmbeddings
from qdrant_client import QdrantClient

vectorstore = Qdrant.from_documents(
    documents=chunked_documents, 
    embedding=OpenAIEmbeddings(), 
    location=":memory:",
    collection_name="my_documents",)

# 4. RetrievalQA Chain
import logging
from langchain.chat_models import ChatOpenAI
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.chains import RetrievalQA

# Setting up basic logging configuration
logging.basicConfig()
# Configuring logging level for the MultiQueryRetriever
logging.getLogger('langchain.retrievers.multi_query').setLevel(logging.INFO)

# Initializing the Large Language Model (LLM) using OpenAI GPT-3.5 with specific parameters
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

# Creating a MultiQueryRetriever instance that uses both the vector store retriever and the LLM
retriever_from_llm = MultiQueryRetriever.from_llm(retriever=vectorstore.as_retriever(), llm=llm)

# Setting up the RetrievalQA chain, combining the LLM and the MultiQueryRetriever
qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever_from_llm)


# 5. Q&A Output system
from flask import Flask, request, render_template
app = Flask(__name__) # Flask APP

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':

        # User input question
        question = request.form.get('question')        
        
        # RetrievalQA Chain
        result = qa_chain({"query": question})
        
        # Reinder template
        return render_template('index.html', result=result)
    
    return render_template('index.html')

if __name__ == "__main__":
    app.run(host='0.0.0.0',debug=True,port=8080)
