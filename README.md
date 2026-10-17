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

L'analyse audio locale (étape 1) s'installe avec `uv sync --extra audio`.

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
  cli.py         commandes Typer
tests/           pytest + fixture Rekordbox XML
```

Licences des dépendances et modèles : voir [THIRD_PARTY.md](THIRD_PARTY.md).
