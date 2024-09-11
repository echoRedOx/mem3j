import chromadb
import ollama


class OllamaNomicEmbed(chromadb.EmbeddingFunction):
    def __call__(self, input: chromadb.Documents) -> chromadb.Embeddings:
        embeddings = ollama.embeddings(model='nomic-embed-text', prompt=input)
        return embeddings
    

def embed_nomic(embed_model, text):
    return ollama.embeddings(model=embed_model, prompt=text)['embedding']
