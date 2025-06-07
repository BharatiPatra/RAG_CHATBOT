
# import requests
# from bs4 import BeautifulSoup
# from urllib.parse import urljoin
# from langchain_openai import OpenAIEmbeddings
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.docstore.document import Document
# import os
# from dotenv import load_dotenv
# from pinecone import Pinecone, ServerlessSpec
# from langchain_pinecone import PineconeVectorStore
# # Step 1: Load env variables
# load_dotenv()

# PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
# PINECONE_ENV = os.getenv("PINECONE_ENVIRONMENT")
# INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

# # Initialize Pinecone
# print("Initializing Pinecone client...")
# pc = Pinecone(api_key=PINECONE_API_KEY)
# print("Pinecone client initialized successfully.")


# # Step 3: Create index if not exists
# embedding_model = OpenAIEmbeddings(model="text-embedding-3-large")
# EMBEDDING_DIM = 3072  # for text-embedding-3-large

# if INDEX_NAME not in pc.list_indexes():
#     pc.create_index(
#         name=INDEX_NAME,
#         dimension=EMBEDDING_DIM,
#         metric="cosine",
#         spec=ServerlessSpec(
#             cloud='aws', # Specify your desired cloud provider
#             region=PINECONE_ENV  # Specify the region
#         )
#     )

# # Step 4: Connect to Pinecone index
# index = pc.Index(INDEX_NAME)

# # Base URL to crawl
# base_url = "https://chaidocs.vercel.app/youtube/getting-started/"

# # Step 1: Get all unique internal links
# response = requests.get(base_url)
# response.raise_for_status()
# soup = BeautifulSoup(response.text, 'html.parser')
# anchor_tags = soup.find_all('a', href=True)
# unique_links = list(set([urljoin(base_url, tag['href']) for tag in anchor_tags]))
# print(f"Found {len(unique_links)} unique links.")

# # Step 2: Crawl each page and extract text and code
# docs = []
# for link in unique_links:
#     try:
#         print(f"\nProcessing: {link}")
#         page_response = requests.get(link)
#         page_response.raise_for_status()
#         page_soup = BeautifulSoup(page_response.text, 'html.parser')

#         # Extract text content
#         content_parts = []

#         # Extract title
#         title = page_soup.title.string if page_soup.title else "No Title"
#         content_parts.append(f"# {title}")

#         # Paragraphs
#         paragraphs = page_soup.find_all('p')
#         for para in paragraphs:
#             content_parts.append(para.get_text(strip=True))

#         # Code snippets in <pre><code> or <div> blocks with classes like "code" or "highlight"
#         code_blocks = page_soup.find_all(['h1', 'code', 'div'], class_=['code', 'highlight'])
#         for code in code_blocks:
#             content_parts.append("```python\n" + code.get_text(strip=True) + "\n```")

#         # Join and create Document
#         page_text = "\n\n".join(content_parts)
#         docs.append(Document(page_content=page_text, metadata={"source": link}))
#       # Print first 200 characters of the content

#     except requests.exceptions.RequestException as e:
#         print(f"Failed to process {link}: {e}")

# # Step 4: Split the content into smaller chunks
# text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size=1000,
#     chunk_overlap=400,
#     length_function=len
# )
# split_docs = text_splitter.split_documents(docs)

# # Step 5: Store embeddings using LangChain’s Pinecone wrapper in batches

# # First, initialize the vector store without any documents
# vectorstore = PineconeVectorStore(
#     index_name=INDEX_NAME,
#     embedding=embedding_model
# )

# # Define the size of each batch
# batch_size = 100 

# # Loop through the split_docs in batches
# print(f"Adding {len(split_docs)} documents to Pinecone in batches of {batch_size}...")
# for i in range(0, len(split_docs), batch_size):
#     # Find the end of the batch
#     i_end = min(i + batch_size, len(split_docs))
#     # Get the batch of documents
#     batch = split_docs[i:i_end]
#     # Add the batch to Pinecone
#     vectorstore.add_documents(batch)
#     print(f"Added batch {i//batch_size + 1}/{(len(split_docs) + batch_size - 1)//batch_size}")


# print("✅ Successfully stored embeddings in Pinecone!")










# # if query:
# #     with st.spinner("Searching..."):
# #         results = search_docs(query)

# #     st.success(f"Top {len(results)} results for: **{query}**")

# #     for i, doc in enumerate(results, start=1):
# #         st.markdown(f"### Result {i}")
# #         st.markdown(f"**Source**: {doc.metadata.get('source', 'Unknown')}")
# #         st.code(doc.page_content[:2000], language="markdown")  # Trimmed for UI display
# #         st.markdown("---")
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore


def load_env_vars():
    load_dotenv()
    return {
        "api_key": os.getenv("PINECONE_API_KEY"),
        "env": os.getenv("PINECONE_ENVIRONMENT"),
        "index_name": os.getenv("PINECONE_INDEX_NAME")
    }


def init_pinecone(api_key):
    print("Initializing Pinecone client...")
    pc = Pinecone(api_key=api_key)
    print("Pinecone client initialized successfully.")
    return pc


def create_index_if_needed(pc, index_name, env, dim=3072):
    if index_name not in pc.list_indexes():
        pc.create_index(
            name=index_name,
            dimension=dim,
            metric="cosine",
            spec=ServerlessSpec(cloud='aws', region=env)
        )


def get_internal_links(base_url):
    response = requests.get(base_url)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, 'html.parser')
    anchor_tags = soup.find_all('a', href=True)
    return list(set([urljoin(base_url, tag['href']) for tag in anchor_tags]))


def crawl_and_extract_docs(links):
    docs = []
    for link in links:
        try:
            print(f"\nProcessing: {link}")
            page_response = requests.get(link)
            page_response.raise_for_status()
            page_soup = BeautifulSoup(page_response.text, 'html.parser')

            content_parts = []
            title = page_soup.title.string if page_soup.title else "No Title"
            content_parts.append(f"# {title}")

            for para in page_soup.find_all('p'):
                content_parts.append(para.get_text(strip=True))

            code_blocks = page_soup.find_all(['h1', 'code', 'div'], class_=['code', 'highlight'])
            for code in code_blocks:
                content_parts.append("```python\n" + code.get_text(strip=True) + "\n```")

            page_text = "\n\n".join(content_parts)
            docs.append(Document(page_content=page_text, metadata={"source": link}))

        except requests.exceptions.RequestException as e:
            print(f"Failed to process {link}: {e}")
    return docs


def split_documents(docs, chunk_size=1000, chunk_overlap=400):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    return splitter.split_documents(docs)


def store_documents_in_pinecone(index_name, embedding_model, docs, batch_size=100):
    vectorstore = PineconeVectorStore(index_name=index_name, embedding=embedding_model)
    print(f"Adding {len(docs)} documents to Pinecone in batches of {batch_size}...")
    for i in range(0, len(docs), batch_size):
        batch = docs[i:i + batch_size]
        vectorstore.add_documents(batch)
        print(f"Added batch {i // batch_size + 1}/{(len(docs) + batch_size - 1) // batch_size}")
    print("✅ Successfully stored embeddings in Pinecone!")


def main():
    env_vars = load_env_vars()
    pc = init_pinecone(env_vars["api_key"])
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-large")

    create_index_if_needed(pc, env_vars["index_name"], env_vars["env"])

    index = pc.Index(env_vars["index_name"])

    base_url = "https://chaidocs.vercel.app/youtube/getting-started/"
    links = get_internal_links(base_url)
    print(f"Found {len(links)} unique links.")

    docs = crawl_and_extract_docs(links)
    split_docs = split_documents(docs)

    store_documents_in_pinecone(env_vars["index_name"], embedding_model, split_docs)


if __name__ == "__main__":
    main()
