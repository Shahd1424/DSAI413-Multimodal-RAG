from models.colpali_embedder import ColPaliEmbedder
from vector_store import VectorStore

pdf_path = "data/sample.pdf"

embedder = ColPaliEmbedder()

print(" Processing PDF...")
images = embedder.pdf_to_images(pdf_path)

print(" Generating embeddings...")
embeddings, metadata = embedder.embed_images(images)

print(" Building vector database...")
dim = len(embeddings[0])

store = VectorStore(dim)
store.add(embeddings, metadata)

print(" Vector DB ready!")