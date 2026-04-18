from models.colpali_embedder import ColPaliEmbedder
from rag_qa import RAGQA

pdf_path = "data/sample.pdf"

print(" Building system...")

embedder = ColPaliEmbedder()
images = embedder.pdf_to_images(pdf_path)
embeddings, metadata = embedder.embed_images(images)

rag = RAGQA(embeddings, metadata)

while True:
    q = input("\n Ask a question: ")

    results = rag.ask(q)

    print("\n Top relevant pages:")
    print("\n Answer:")
    print(results["answer"])

    print("\n Pages:")
    for p in results["pages"]:
        print(p)

    print("\n Images:")
    for img in results["images"]:
        img.show()