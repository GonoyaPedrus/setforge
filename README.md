# SetForge

Assistant DJ : analyse de ta bibliothèque, recommandation du morceau suivant, construction de sets
et export Rekordbox.

## Démarrage

```bash
uv sync                      # dépendances + outils de dev
uv run setforge --help

# Dans Rekordbox : Fichier > Exporter la collection au format XML
uv run setforge stats --xml ~/rekordbox.xml
uv run setforge next --xml ~/rekordbox.xml "night drive"
```

### Analyse audio locale

```bash
uv sync --extra audio
uv run setforge scan --xml ~/rekordbox.xml --workers 4
```

`scan` lit les fichiers de la collection, calcule l'énergie (0 à 1, relative à ta bibliothèque) et
un BPM et une tonalité de secours, et les garde dans `~/.setforge/library.sqlite` (option `--db` ou
variable `SETFORGE_DB`). Un second scan ne réanalyse que les fichiers nouveaux ou modifiés.
`stats` et `next` utilisent ensuite ces analyses ; le BPM et la tonalité de Rekordbox restent
prioritaires. Les fichiers M4A/AAC demandent FFmpeg installé sur la machine.

## Développement

```bash
uv run ruff check . && uv run ruff format --check .
uv run mypy
uv run pytest
```

Branches : `feature/…`, `fix/…`, `hotfix/…`, jamais de push direct sur `main`.

## Structure

```
src/setforge/
  camelot.py     tonalités → Camelot, compatibilité harmonique
  models.py      Track, CuePoint
  rekordbox.py   import collection XML, export playlist XML
  scoring.py     score de transition (tonalité, BPM, énergie)
  store.py       cache SQLite des analyses, énergie relative à la bibliothèque
  analysis/      descripteurs audio (librosa) et scan parallèle
  cli.py         commandes Typer
tests/           pytest + fixture Rekordbox XML
```

Licences des dépendances et modèles : voir [THIRD_PARTY.md](THIRD_PARTY.md).
