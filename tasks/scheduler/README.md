Briefing - Modulo Scheduler/Loop
================================

Objetivo
--------
Refinar o loop cognitivo para usar sinais coerentes com a doc: entropia normalizada a partir dos logits, estabilidade temporal, satisfacao ligada a progresso/estagnacao, confianca calibrada; escolha de best candidate usando score + estabilidade.

Escopo
------
- Em `src/ubrain/scheduler/loop.py`: `_estimate_signals` deve usar H(p_t), estabilidade (TV/KL), progresso e satisfacao conforme doc; penalizar estagnacao; normalizar entropia.
- `_extract_candidate` guiado por `output_format` (token/boolean, expansivel para grid/som); validar tipo; IDK se formato nao bater.
- `_is_better`: comparar probabilidade/score do candidato vs best, ponderando confianca/estabilidade antes de trocar.
- Integrar budget nos sinais; satisfacao e decisoes devem considerar budget_per_step.
- Deixar interfaces claras para policy consumir confianca/satisfacao/budget.

Entregaveis
-----------
- Atualizacao em `src/ubrain/scheduler/loop.py`.
- Testes simples de sinais: entropia cai quando logits se concentram; estagnacao reduz satisfacao; best_candidate so troca quando score melhora.

Referencias
-----------
- `docs/README.md` (fluxo cognitivo, sinais, invariantes).
- API atual do loop em `src/ubrain/scheduler/loop.py`.
