from models.colpali_embedder import ColPaliEmbedder

pdf_path = "data/sample.pdf"

embedder = ColPaliEmbedder()

print(" Converting PDF into images...")
images = embedder.pdf_to_images(pdf_path)
print(f" Extracted {len(images)} pages as images")

print(" Generating embeddings...")
embeddings, metadata = embedder.embed_images(images)

print(f" Got {len(embeddings)} embeddings")
print("Example metadata:", metadata[0])
print("Embedding vector size:", len(embeddings[0]))