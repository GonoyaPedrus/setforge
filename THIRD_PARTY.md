# Dépendances tierces et licences

Règle : tout code **et tout poids de modèle** embarqué dans une version distribuée doit être sous
licence permissive (MIT, BSD, ISC, Apache-2.0). Exception tolérée : les bibliothèques LGPL liées
dynamiquement et non modifiées (libsndfile via soundfile, soxr via librosa), en livrant leur
licence et en permettant leur remplacement. GPL et AGPL restent exclues. Les composants « expérimentation seulement » ne
doivent jamais être importés hors de `experiments/`.

| Composant | Usage | Licence | Statut |
| --- | --- | --- | --- |
| typer, pydantic | CLI, modèle de données | MIT | Utilisé |
| librosa | Features DSP | ISC (dépend de soxr, LGPL-2.1+) | Extra `audio` |
| soundfile | Décodage WAV/FLAC/AIFF | BSD-3-Clause (embarque libsndfile, LGPL-2.1) | Extra `audio` |
| tinytag | Lecture des tags | MIT | Extra `audio` |
| LAION CLAP `larger_clap_general` | Embeddings audio-texte | Apache-2.0 (README du dépôt Hugging Face, vérifié le 2026-10-21) | Extra `clap` ; poids non redistribués, téléchargés par l'utilisateur. Données d'entraînement (LAION-Audio-630K et musique) à auditer avant une version payante |
| torch, transformers | Inférence CLAP | BSD-3-Clause / Apache-2.0 | Extra `clap` |
| lancedb, pyarrow | Index vectoriel | Apache-2.0 | Extra `index` |
| Demucs v4 | Séparation de stems | MIT | Prévu, optionnel |
| MERT | Embeddings musicaux | CC BY-NC 4.0 | Expérimentation seulement |
| madmom (poids) | Beat tracking | CC BY-NC-SA 4.0 | Exclu |
| Essentia | Analyse audio | AGPL-3.0, modèles NC | Exclu |
| mutagen | Tags | GPL-2.0 | Exclu (remplacé par tinytag) |
