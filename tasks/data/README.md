Briefing - Modulo Dataset/Loader
================================

Objetivo
--------
Manter/expandir o dataset fechado (`data/challenges.yaml`) e o loader (`load_challenges`) com formatos bem definidos e recompensa base por desafio.

Escopo
------
- Validar schema do YAML: `id`, `prompt` (tokens), `output_format` (token/boolean/... ), `ground_truth` (para treino/avaliacao externa), `reward`.
- Adicionar variacoes coerentes com formatos fechados; preparar terreno para outros formatos (grid_token, som) sem texto livre.
- Loader: garantir suporte a novos formatos e validacao minima; opcionalmente adicionar script de validacao/splits.

Entregaveis
-----------
- Dataset atualizado em `data/challenges.yaml` (ou arquivos adicionais).
- Loader ajustado em `src/ubrain/data/spec.py` se necessario.
- (Opcional) Script de validacao/split em `tasks/data/`.

Referencias
-----------
- `docs/README.md` (datasets e curriculo).
- `src/ubrain/data/spec.py`.
