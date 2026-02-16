import os
import shutil
from typing import List
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

load_dotenv()

class VectorStoreManager:
    """
    Gestisce il database vettoriale.
    """
    def __init__(self, api_key: str = None, persist_dir: str = "./chroma_db", collection_name: str = "datatrust_docs"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        
        if not self.api_key:
            raise ValueError("API Key non trovata!")

        self.persist_dir = persist_dir
        self.collection_name = collection_name

        self.embedding_model = OpenAIEmbeddings(
            model="text-embedding-3-small", 
            api_key=self.api_key
        )
        
        self._init_vector_store()

    def _init_vector_store(self):
        """Metodo interno per inizializzare Chroma."""
        self.vector_store = Chroma(
            persist_directory=self.persist_dir,
            embedding_function=self.embedding_model,
            collection_name=self.collection_name
        )

    def add_documents(self, documents: List[Document]):
        """Aggiunge documenti al database."""
        if documents:
            self.vector_store.add_documents(documents)

    def get_retriever(self, k: int = 4):
        """Restituisce l'oggetto che serve per fare le ricerche."""
        return self.vector_store.as_retriever(search_kwargs={"k": k})

    def count(self) -> int:
        """Restituisce il numero di documenti nella collezione."""
        return self.vector_store._collection.count()

    def get_all_documents(self) -> List[Document]:
        """Restituisce tutti i documenti dalla collezione (per BM25)."""
        result = self.vector_store._collection.get(include=["documents", "metadatas"])
        docs = []
        for text, meta in zip(result["documents"], result["metadatas"]):
            docs.append(Document(page_content=text, metadata=meta or {}))
        return docs

    def clear(self):
        """
        Cancella l'intera collezione e riparte da zero.
        """
        try:
            self.vector_store.delete_collection()
            self._init_vector_store()
            print(f"Collezione '{self.collection_name}' pulita con successo.")
            return True
        except Exception as e:
            print(f"Attenzione: Impossibile pulire la collezione: {e}")
            self._init_vector_store()
            return False