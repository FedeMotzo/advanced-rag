import os
import logging
from dotenv import load_dotenv

from rag_orchestrator import RAGOrchestrator
from vector_db.vector_store_manager import VectorStoreManager
from ingestion.chunking_strategies import SemanticChunkingStrategy

logging.basicConfig(level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("chromadb").setLevel(logging.WARNING)

def main():
    load_dotenv()
    print("=" * 60)
    print("  RAG Manual Test Console")
    print("=" * 60)

    print("\n[1/3] Inizializzazione Orchestrator...")
    
    db_manager = VectorStoreManager(collection_name="datatrust_manual_test_semantic")
    
    orchestrator = RAGOrchestrator(
        db_manager=db_manager,
        chunking_strategy=SemanticChunkingStrategy(),
        reranker=True,
        hybrid_search=False
    )

    print("[2/3] Verifica documenti...")
    files_to_ingest = [
        "files/EU_AI_Regulation_2024_77.pdf",
        "files/Verbale_CdA_2024_07_20.docx",
        "files/Financial_Report_Q3_2024.xlsx"
    ]

    for f in files_to_ingest:
        if os.path.exists(f):
            print(f"  -> Ingesting: {f} ...")
            orchestrator.ingest(f)
        else:
            print(f"  File mancante: {f}")

    print("\n[3/3] Sistema Pronto!")
    print("Scrivi la tua domanda ('exit' per uscire).")

    while True:
        try:
            print("-" * 60)
            question = input("\nDomanda: ").strip()
            
            if not question:
                continue
            
            if question.lower() in ["exit", "quit", "esci"]:
                print("Arrivederci!")
                break

            print("\nElaborazione...\n")
            
            response = orchestrator.ask(question, source="auto")
            
            print(f"Risposta: {response['answer']}")
            
            if response.get('routed_to') == 'text' and response.get('sources'):
                print(f"\nFonti: {response['sources']}")
            
            if response.get('routed_to') == 'excel':
                print(f"\nFoglio Excel usato: {response.get('scope', 'N/A')}")
                
        except KeyboardInterrupt:
            print("\nInterrotto dall'utente.")
            break
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"\nErrore: {e}")

if __name__ == "__main__":
    main()
