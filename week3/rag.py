import os
import glob
from dotenv import load_dotenv

from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
import numpy as np
# from sklearn.manifold import TSNE
# import plotly.graph_objects as go
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

from langchain_core.callbacks import StdOutCallbackHandler # helps with printing langchain logs



# Retrieval-augmented generation (RAG) is a technique that enhances the accuracy and 
# reliability of generative AI models by incorporating information from external knowledge sources. 

load_dotenv(override=True)
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY', 'your-key-if-not-using-env')


class RAG:
    def __init__(self, aiModel, vectorDbName):
        self.aiModel = aiModel
        self.vectorDbName = vectorDbName
        
        folders = glob.glob("knowledge-base/*")
        text_loader_kwargs = {'encoding': 'utf-8'}

        documents = []
        for folder in folders:
            doc_type = os.path.basename(folder)
            loader = DirectoryLoader(folder, glob="**/*.md", loader_cls=TextLoader, loader_kwargs=text_loader_kwargs)
            folder_docs = loader.load()
            for doc in folder_docs:
                doc.metadata["doc_type"] = doc_type
                documents.append(doc)
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)

        print('-------------------------------')
        print(    len(chunks))
        doc_types = set(chunk.metadata['doc_type'] for chunk in chunks)
        print(f"Document types found: {', '.join(doc_types)}")
        print('-------------------------------')

        embeddings = OpenAIEmbeddings()
        if os.path.exists(vectorDbName):
            Chroma(persist_directory=vectorDbName, embedding_function=embeddings).delete_collection()

        vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=vectorDbName)
        print(f"Vectorstore created with {vectorstore._collection.count()} documents")

        # collection = vectorstore._collection
        # sample_embedding = collection.get(limit=1, include=["embeddings"])["embeddings"][0]
        # dimensions = len(sample_embedding)
        # print(f"The vectors have {dimensions:,} dimensions")

        llm = ChatOpenAI(temperature=0.7, model_name=aiModel)
        memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)

        # the retriever is an abstraction over the VectorStore that will be used during RAG; k is how many chunks to use
        retriever = vectorstore.as_retriever(search_kwargs={"k":25})
        
        # this line will log chunks to the console, helps to debug issues with llm search
        # conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, memory=memory, callbacks=[StdOutCallbackHandler()])
        self.conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, memory=memory)
        
    def ask(self, question):
        print('-------------------------------')
        print('Question: ', question)
        print('Answer: ', self.conversation_chain.invoke({"question": question})["answer"])


def main():
    
    ai = RAG("gpt-4o-mini", "vector_db")

    ai.ask("Can you describe EchmaTech Solutions in a few sentences")
    ai.ask("Can you pull up HR notes for David Thompson?")
    ai.ask("Is there any empoloyee who is working remotely from South Korea?")
    ai.ask("Who received the AcmeTech Brawo award in 2024?")
    ai.ask("Who are the people got highest performance review ratings?")


if __name__ == "__main__":
    main()