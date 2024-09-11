import chromadb
from typing import Any
from embeddings import OllamaNomicEmbed


chroma_library_path = "library/chroma.db"


class ChromaHandler:
    def __init__(self, library_path: str, collection_name: str):
        self.library_path = library_path
        self.default_ef = OllamaNomicEmbed
        self.client = self._init_client()
        self.active_collection = self.collection_get_or_create(collection_name)

    def _init_client(self):
        try:
            return chromadb.PersistentClient(path=self.library_path)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize ChromaDB client: {e}")

    def collection_get_or_create(self, collection_name: str):
        try:
            collection = self.client.get_or_create_collection(name=collection_name,metadata={"hnsw:space": "cosine"})
            self.active_collection = collection
            return collection
        except Exception as e:
            raise RuntimeError(f"Error accessing collection {collection_name}: {e}")

    def change_active_collection(self, collection_name: str):
        self.active_collection = self.collection_get_or_create(collection_name)

    def add_to_collection(self, documents: list[str], ids: list[str], metadatas: list[dict], embedding_function=None):
        if not documents or not ids or not metadatas:
            raise ValueError("Documents, ids and metadatas must be provided.")
        if embedding_function is None:
            embedding_function = self.default_ef
        
        try:
            self.active_collection.add(ids=ids,documents=documents,metadatas=metadatas,embedding_function=embedding_function)
        except Exception as e:
            raise RuntimeError(f"Failed to add documents to collection: {e}")

    def query_collection(self, query_texts: list[str], n_results: int = 5):
        try:
            return self.active_collection.query(query_texts=query_texts, n_results=n_results)
        except Exception as e:
            raise RuntimeError(f"Error querying collection: {e}")

    def list_collections(self):
        try:
            return self.client.list_collections()
        except Exception as e:
            raise RuntimeError(f"Failed to list collections: {e}")
    
    def chroma_results_format_to_prompt(chroma_results):
        """Enforcing formatting rules for consistency."""
        if not chroma_results["documents"] or all(not doc for doc in chroma_results["documents"]):
            return "No results found."
        formatted_output = ""
        for result in chroma_results["documents"]:
            if isinstance(result, list):
                result = " ".join(result)

        # Split the result into components (sender, timestamp, message)
        components = result.split(" @ ")
        sender = components[0].strip()
        timestamp = components[1].strip()
        message = " @ ".join(components[2:]).strip()
        # Format each entry
        formatted_output += f"\n{sender} ({timestamp}):\n{message}"

        return formatted_output


def add_chunk_to_collection(collection: chromadb.Collection, docname: str, chunk_idx, chunk, embeddings):
    collection.add([docname+str(chunk_idx)], embeddings=[embeddings], documents=chunk, metadatas={"source": docname})



