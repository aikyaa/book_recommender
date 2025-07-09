import pandas as pd
import numpy as np
import gradio as gr


from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

model_name = "sentence-transformers/all-mpnet-base-v2"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
hf = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

books=pd.read_csv("books_with_emotions.csv")
books["large_thumbnail"]=books["thumbnail"] + "&fife=800"
books["large_thumbnail"]=np.where(
    books["large_thumbnail"].isna(),
    "No_Cover.jpg",
    books["large_thumbnail"],
)

raw_documents=TextLoader("tagged_description.txt").load()
text_splitter=CharacterTextSplitter(chunk_size=0, chunk_overlap=0, separator="\n")
documents=text_splitter.split_documents(raw_documents)
db_books = Chroma.from_documents(documents, hf)

def retrieve_recommendations(
        query: str,
        category: str = None,
        tone:str=None,
        initial_top_k: int=50,
        final_top_k: int=16,
) ->pd.DataFrame:
    
    recs = db_books.similarity_search(query, k=initial_top_k)
    recommended_books = [int(rec.page_content.split()[0].strip().strip('"')) for rec in recs]
    book_recs = books[books["isbn13"].isin(recommended_books)].head(initial_top_k)
    #book_recs.to_csv("recs.csv", index=False)

    #dropdown
    
    if category!="All":
        book_recs=book_recs[book_recs["simple_categories"]==category].head(final_top_k)
    else :
        book_recs= book_recs.head(final_top_k)

    #discarded disgust, neutral(essentially all)

    if tone=="Happy":
        book_recs.sort_values(by="joy", ascending=False, inplace=True)
    elif tone=="Surprising":
        book_recs.sort_values(by="surprise", ascending=False, inplace=True)
    elif tone=="Angry":
        book_recs.sort_values(by="anger", ascending=False, inplace=True)
    elif tone=="Suspenseful":
        book_recs.sort_values(by="fear", ascending=False, inplace=True)
    elif tone=="Sad":
        book_recs.sort_values(by="sadness", ascending=False, inplace=True)

    return book_recs

def recommend_books(
        query: str,
        category:str,
        tone:str
):
    recommendations = retrieve_recommendations(query, category, tone)
    results=[]

    for _, row in recommendations.iterrows():
        description = row["description"]
        truncated_desc_split = description.split()
        truncated_description = " ".join(truncated_desc_split[:30]) + "..."

        authors_split = row["authors"].split(";")
        if len(authors_split) == 2:
            authors_str = f"{authors_split[0]} and {authors_split[1]}"
        elif len(authors_split) > 2:
            authors_str = f"{', '.join(authors_split[:-1])}, and {authors_split[-1]}"
        else:
            authors_str = row["authors"]

        caption = f"{row['title']} by {authors_str}: {truncated_description}"
        results.append((row["large_thumbnail"], caption))
    return results

categories = ["All"] + sorted(books["simple_categories"].unique())
tones = ["All"] + ["Happy", "Surprising", "Suspenseful", "Sad", "Angry"]

with gr.Blocks(theme = gr.themes.Soft()) as dashboard :
    gr.Markdown("Book Recommender")
    
    with gr.Column():
        with gr.Row() :
            user_query = gr.Textbox(label = "Please enter a description of a book:",
                                    placeholder ="e.g , A story about forgiveness"
                                    )
            category_dropdown = gr.Dropdown(choices = categories, label= "Select a category:", value="All")
            tone_dropdown = gr.Dropdown(choices = tones, label= "Select a tone:", value="All")
        submit_button = gr.Button("Get reccommendations")
    
    gr.Markdown("## Recommendations")
    output = gr.Gallery(label="Recommended books", columns=5, rows=2)

    submit_button.click(fn=recommend_books, inputs=[user_query, category_dropdown, tone_dropdown], outputs= output)

if __name__ == "__main__" :
        dashboard.launch()