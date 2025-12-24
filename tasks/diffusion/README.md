Briefing - Modulo Difusao
=========================

Objetivo
--------
Implementar um modelo de difusao discreta condicional ao desafio, sem usar ground truth no episodio. `init_noisy` aplica mascara/ruido nos tokens do prompt; `step(noisy_state, challenge, t)` retorna (novo_estado, logits/probs) reduzindo entropia gradualmente com base no desafio/tokenizacao.

Escopo
------
- Manter a interface `DiffusionModel` (`init_noisy`, `step`).
- Estado inicial: aplica ruido/mask em tokens do prompt (inclui "?"), gera logits/probs iniciais condicionados ao desafio.
- Passo de refino: denoising condicional ao desafio (mas sem gabarito), produz logits/probs; entropia deve cair quando ha evidencias.
- Formatos: foco inicial em `output_format` token/boolean; planejar extensao a grid/som.
- Nao usar ground truth dentro do episodio; reward e externo.

Entregaveis
-----------
- Classe concreta em `src/ubrain/diffusion/`.
- Testes unitarios simples: logits normalizados, entropia cai quando evidencias aparecem; nao vazamento de ground truth.

Referencias
-----------
- `docs/README.md` (conceito completo).
- `src/ubrain/diffusion/model.py` (protocolo).
