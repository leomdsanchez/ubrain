Briefing - Modulo Policy/Decisao
================================

Objetivo
--------
Substituir a policy heuristica por uma que use confianca, satisfacao e budget para decidir {continuar, concluir, IDK}, alinhada ao comportamento descrito na doc.

Escopo
------
- Em `src/ubrain/decision/`: implementar policy que:
  - Conclui quando confianca alta + estabilidade.
  - IDK quando confianca baixa e satisfacao caindo/estagnada ou budget apertado.
  - Continua quando ha progresso e budget remanescente.
- Usar sinais do loop (`LoopSignals`) incluindo budget e satisfacao.
- Deixar limiares configuraveis para tuning.

Entregaveis
-----------
- Nova policy (ou revisao da existente) em `src/ubrain/decision/`.
- Testes de logica: cenarios com sinais simulados (progresso/estagnacao/regressao, budget alto/baixo) e decisoes esperadas.

Referencias
-----------
- `docs/README.md` (decisao, IDK honesto, budget).
- `src/ubrain/scheduler/loop.py` para formato de `LoopSignals`.
