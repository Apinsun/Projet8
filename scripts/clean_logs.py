import os
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

def clean_test_data():
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")
    
    if not url or not key:
        print("❌ Erreur : Variables d'environnement Supabase manquantes.")
        return

    supabase: Client = create_client(url, key)
    
    print("🧹 Suppression des données marquées 'is_test=True'...")
    try:
        # Suppression ciblée
        response = supabase.table("predictions_logs").delete().eq("is_test", True).execute()
        print(f"✅ Nettoyage terminé. {len(response.data)} lignes supprimées.")
    except Exception as e:
        print(f"❌ Erreur lors du nettoyage : {e}")

if __name__ == "__main__":
    clean_test_data()
