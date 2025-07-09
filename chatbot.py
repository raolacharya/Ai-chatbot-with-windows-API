import os
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

# Load and split documents
loader = TextLoader("company_faq.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs = text_splitter.split_documents(documents)

# Create vector DB
embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(docs, embeddings)

# Setup chatbot chain
qa = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model_name="gpt-3.5-turbo"),
    retriever=db.as_retriever()
)

# Ask questions in a loop
print("Ask me anything based on your company data (type 'exit' to quit):")
while True:
    question = input("You: ")
    if question.lower() == "exit":
        break
    answer = qa.run(question)
    print("Bot:", answer)
