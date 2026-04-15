-- Création de la table de logs des prédictions
CREATE TABLE IF NOT EXISTS predictions_logs (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now()) NOT NULL,
    client_features JSONB NOT NULL,
    score_defaut REAL NOT NULL,
    decision TEXT NOT NULL
);

-- Ajout d'un index sur la date pour accélérer les futurs graphiques Streamlit
CREATE INDEX IF NOT EXISTS idx_created_at ON predictions_logs(created_at);
