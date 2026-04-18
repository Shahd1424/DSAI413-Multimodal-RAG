import torch
import fitz  # PyMuPDF
from PIL import Image
import numpy as np
from transformers import CLIPProcessor, CLIPModel

class ColPaliEmbedder:
    def __init__(self):
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

    def pdf_to_images(self, pdf_path):
        doc = fitz.open(pdf_path)
        images = []

        for i, page in enumerate(doc):
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

            images.append({
                "page": i + 1,
                "image": img
            })

        return images
    def embed_images(self, images):
        embeddings = []
        metadata = []

        for item in images:
            inputs = self.processor(
                text=["document page content"],
                return_tensors="pt",
                padding=True
            )

            with torch.no_grad():
                outputs = self.model.get_text_features(**inputs)

            emb = outputs.last_hidden_state.mean(dim=1).cpu().numpy().astype("float32").squeeze()

            embeddings.append(emb)

            metadata.append({
                "page": item["page"],
                "image": item["image"]
            })
            
        return embeddings, metadata
    
    def chunk_text(self, text, chunk_size=300):
        words = text.split()
        chunks = []

        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i:i+chunk_size])
            chunks.append(chunk)

        return chunks
    
    def embed_chunks(self, chunks):
        embeddings = []
        metadata = []

        for i, chunk in enumerate(chunks):

            inputs = self.processor(
                text=[chunk],
                return_tensors="pt",
                padding=True
            )

            with torch.no_grad():
                outputs = self.model.get_text_features(**inputs)

            emb = outputs.cpu().numpy().astype("float32").squeeze()

            embeddings.append(emb)

            metadata.append({
                "type": "chunk",
                "chunk_id": i,
                "text": chunk
            })

        return embeddings, metadata