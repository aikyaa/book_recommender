import pandas as pd
import numpy as np
import gradio as gr

from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from langchain_chroma import Chroma


load_dotenv()

embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
    )

books=pd.read_csv("books_with_emotion.csv")
books["large_thumbnail"]=books["thumbnail"] + "&fife=800"
books["large_thumbnail"]=np.where(
    books["large_thumbnail"].isna(),
    "No_Cover.jpg",
    books["large_thumbnail"],
)

raw_documents=TextLoader("tagged_description.txt").load()
text_splitter=CharacterTextSplitter(chunk_size=0, chunk_overlap=0, separator="\n")
documents=text_splitter.split_documents(raw_documents)
db_books = Chroma.from_documents(documents, embeddings, persist_directory="./chroma_db")