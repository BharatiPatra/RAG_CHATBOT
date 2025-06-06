
# import asyncio
# from crawl4ai import *

# async def main():
#     async with AsyncWebCrawler() as crawler:
#         result = await crawler.arun(
#             url="https://chaidocs.vercel.app/youtube/getting-started/",
#         )
#         print(result.markdown)

# if __name__ == "__main__":
#     asyncio.run(main())
# import requests
# from bs4 import BeautifulSoup
# from urllib.parse import urljoin

# # URL to fetch
# url = "https://chaidocs.vercel.app/youtube/getting-started/"

# # Send a GET request to the URL
# response = requests.get(url)
# response.raise_for_status()  # Raise an error for bad status codes

# # Parse the HTML content
# soup = BeautifulSoup(response.text, 'html.parser')

# # Find all anchor tags with href attributes
# links = soup.find_all('a', href=True)

# # Extract and print the absolute URLs
# for link in links:
#     absolute_url = urljoin(url, link['href'])
#     print(absolute_url)
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
# Step 1: Load env variables
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENVIRONMENT")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

# Initialize Pinecone
print("Initializing Pinecone client...")
pc = Pinecone(api_key=PINECONE_API_KEY)
print("Pinecone client initialized successfully.")


# Step 3: Create index if not exists
embedding_model = OpenAIEmbeddings(model="text-embedding-3-large")
EMBEDDING_DIM = 3072  # for text-embedding-3-large

if INDEX_NAME not in pc.list_indexes():
    pc.create_index(
        name=INDEX_NAME,
        dimension=EMBEDDING_DIM,
        metric="cosine",
        spec=ServerlessSpec(
            cloud='aws', # Specify your desired cloud provider
            region=PINECONE_ENV  # Specify the region
        )
    )

# Step 4: Connect to Pinecone index
index = pc.Index(INDEX_NAME)

# Base URL to crawl
base_url = "https://chaidocs.vercel.app/youtube/getting-started/"

# Step 1: Get all unique internal links
response = requests.get(base_url)
response.raise_for_status()
soup = BeautifulSoup(response.text, 'html.parser')
anchor_tags = soup.find_all('a', href=True)
unique_links = list(set([urljoin(base_url, tag['href']) for tag in anchor_tags]))
print(f"Found {len(unique_links)} unique links.")

# Step 2: Crawl each page and extract text and code
docs = []
for link in unique_links:
    try:
        print(f"\nProcessing: {link}")
        page_response = requests.get(link)
        page_response.raise_for_status()
        page_soup = BeautifulSoup(page_response.text, 'html.parser')

        # Extract text content
        content_parts = []

        # Extract title
        title = page_soup.title.string if page_soup.title else "No Title"
        content_parts.append(f"# {title}")

        # Paragraphs
        paragraphs = page_soup.find_all('p')
        for para in paragraphs:
            content_parts.append(para.get_text(strip=True))

        # Code snippets in <pre><code> or <div> blocks with classes like "code" or "highlight"
        code_blocks = page_soup.find_all(['h1', 'code', 'div'], class_=['code', 'highlight'])
        for code in code_blocks:
            content_parts.append("```python\n" + code.get_text(strip=True) + "\n```")

        # Join and create Document
        page_text = "\n\n".join(content_parts)
        docs.append(Document(page_content=page_text, metadata={"source": link}))
      # Print first 200 characters of the content

    except requests.exceptions.RequestException as e:
        print(f"Failed to process {link}: {e}")

# Step 4: Split the content into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=400,
    length_function=len
)
split_docs = text_splitter.split_documents(docs)

# Step 5: Store embeddings using LangChain’s Pinecone wrapper in batches

# First, initialize the vector store without any documents
vectorstore = PineconeVectorStore(
    index_name=INDEX_NAME,
    embedding=embedding_model
)

# Define the size of each batch
batch_size = 100 

# Loop through the split_docs in batches
print(f"Adding {len(split_docs)} documents to Pinecone in batches of {batch_size}...")
for i in range(0, len(split_docs), batch_size):
    # Find the end of the batch
    i_end = min(i + batch_size, len(split_docs))
    # Get the batch of documents
    batch = split_docs[i:i_end]
    # Add the batch to Pinecone
    vectorstore.add_documents(batch)
    print(f"Added batch {i//batch_size + 1}/{(len(split_docs) + batch_size - 1)//batch_size}")


print("✅ Successfully stored embeddings in Pinecone!")