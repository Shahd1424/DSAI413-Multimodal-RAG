import torch
from models.colpali_embedder import ColPaliEmbedder
from vector_store import VectorStore
from PIL import Image

class RAGQA:
    def __init__(self, embeddings, metadata):
        self.embedder = ColPaliEmbedder()
        self.dim = len(embeddings[0])

        self.store = VectorStore(self.dim)
        self.store.add(embeddings, metadata)

    def ask(self, question):

        inputs = self.embedder.processor(
            text=[question],
            return_tensors="pt",
            padding=True
        )

        with torch.no_grad():
            outputs = self.embedder.model.get_text_features(**inputs)

        q_emb = outputs.pooler_output if hasattr(outputs, "pooler_output") else outputs

        q_emb = q_emb.detach().cpu().numpy().astype("float32").reshape(1, -1)

        results = self.store.search(q_emb, k=3)

        pages = [r["page"] for r in results]
        images = [r.get("image", None) for r in results]

        answer = f"Found {len(results)} relevant pages from the document."

        return {
            "question": question,
            "pages": pages,
            "images": images,
            "answer": answer,
            "sources": results
        }