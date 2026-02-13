
import os
import sys
import subprocess
import shutil

def setup_colab_environment():
    """
    Configura l'ambiente Colab per eseguire la pipeline RAG.
    Installa le dipendenze e configura le chiavi API.
    """
    print("🔧 Configurazione ambiente Colab in corso...")
    
    # 1. Installazione dipendenze
    try:
        print("📦 Installazione dipendenze da requirements.txt...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        
        # Patch per SQLite su Colab (ChromaDB richiede versione recente)
        print("📦 Installazione pysqlite3-binary...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pysqlite3-binary"])
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Errore durante l'installazione: {e}")
        return False

    # 2. Configurazione SQLite per ChromaDB (Patch obbligatoria su Colab)
    try:
        __import__('pysqlite3')
        sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
        print("✅ SQLite patch applicata con successo.")
    except ImportError:
        print("⚠️  Impossibile applicare patch SQLite. ChromaDB potrebbe fallire.")

    # 3. Recupero API Key dai Secret di Colab
    try:
        from google.colab import userdata
        api_key = userdata.get('OPENAI_API_KEY')
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
            print("✅ OPENAI_API_KEY caricata dai Secret di Colab.")
        else:
            print("⚠️  OPENAI_API_KEY non trovata nei Secret. Assicurati di averla impostata.")
    except ImportError:
        print("⚠️  Non siamo su Google Colab? Modulo google.colab non trovato.")
    except Exception as e:
        print(f"⚠️  Errore nel recupero della chiave API: {e}")

    # 4. Verifica directory files/
    if not os.path.exists("files"):
        print("❌ Directory 'files/' non trovata. Carica i documenti (PDF/Excel) nella root.")
    
    return True

if __name__ == "__main__":
    if setup_colab_environment():
        print("\n🚀 Avvio valutazione RAGAS...")
        
        # Importa ed esegue lo script di evaluation esistente
        # Nota: facciamo l'import QUI, dopo aver settato l'ambiente (patch sqlite, env vars)
        try:
            from evaluation.evaluate import main
            main()
        except ImportError as e:
            print(f"❌ Errore importazione evaluation: {e}")
            print("Assicurati di essere nella root del progetto (dove c'è evaluation/evaluate.py)")
        except Exception as e:
            print(f"❌ Errore durante l'esecuzione: {e}")
