Arquitetura Cognitiva - PoC (64D)
=================================

Escopo rapido
-------------
- Cerebro difusivo discreto que pensa iterativamente, mede progresso e decide parar ou dizer IDK.  
- Tarefas fechadas com ground truth; output = plano/decisao/condicionamento, nao texto longo ou execucao.  
- Budget cognitivo explicito; erro confiante e punido; IDK e saida legitima.

Essencia do modelo
------------------
- Pensar = reducao de ruido/entropia via difusao discreta em tokens/estado latente pequeno.  
- Objetivo sempre externo e mensuravel (0-1).  
- Formato de resposta fechado (opcoes, indices, sinais, patches).  
- IDK evita custo/divida cognitiva quando nao ha evidencia suficiente; pensar demais custa oportunidade, nao bloqueia.

Arquitetura e papeis
--------------------
- Cerebro (rede): modelo de difusao discreta (D3PM-like) com backbone pequeno (Transformer/MLP) que prediz logits/score por passo.  
- Scheduler/mente: laco explicito que aplica passos de difusao, mede sinais internos, atualiza satisfacao e budget, e executa a acao escolhida pela policy de decisao.  
- Decisor: cabeca pequena que escolhe continuar | concluir | IDK com base no estado atual + historico curto.  
- Memoria externa (opcional): cache de atalhos (estado inicial -> prior/hint) para acelerar convergencia.  
- Verificadores (abas): instancias curtas que testam hipoteses e retornam {true, false, unknown} com confianca e budget proprio.  
- Executores (maos): modelos separados para desenhar icones, gerar audio simples ou texto curto; nao pensam.  
- Avaliador: calcula reward final com ground truth, custo cognitivo e calibracao de confianca.

Loop cognitivo detalhado
------------------------
1) Ingestao do desafio: tarefa canonica, formato fechado, ground truth unica, sem dicas.  
2) Estado inicial ruidoso: mascara/substituicao em tokens/latente; representa ignorancia inicial.  
3) Historico inicial: confidence0, entropia0, stability0, satisfacao0, budget0, best_candidate.  
4) Repetir (sem parada externa):  
   - Difusao/refino: passo de q(x_t|x_{t-1}) condicionado no desafio.  
   - Gerar hipotese: extrair candidato no formato da tarefa.  
   - Medir sinais:  
     * confidence interna (estabilidade temporal, margem entre hipoteses, colapso do espaco de solucao)  
     * entropia residual  
     * stability_t (mudanca temporal da hipotese)  
     * progresso (delta vs passo anterior)  
   - Atualizar historico: best_candidate anytime; satisfacao sobe com progresso, cai com regressao, cai leve com estagnacao.  
   - Atualizar budget: decrementa a cada passo; pode ficar negativo (divida cognitiva) sem travar loop.  
   - Decidir: continuar | concluir (alta confidence + estabilizacao) | IDK (confidence baixa ou satisfacao caindo).  
5) Emissao: best_candidate + confidence ou <IDK>.  
6) Avaliacao externa (treino): reward final = base fixa (acerto/erro/IDK) - custo(budget_used), com penalidade forte para erro confiante e frustracao leve em IDK; nao altera o episodio em curso.

Sinais internos e invariantes
-----------------------------
- Confidence mede decidibilidade interna, nao veracidade.  
- Entropia residual indica ignorancia; deve cair ao longo do refino.  
- Satisfacao guia a decisao mas nao e reward.  
- Budget mede custo, nao autoridade; pode ir abaixo de zero.  
- IDK/IGNORE e resposta valida quando estagnacao ou incerteza alta.  
- Erro confiante e o pior resultado; honestidade > chute.

Budget, custo e reward
----------------------
- Reward so no final: base fixa (acerto/erro/IDK) menos custo proporcional ao budget usado (inclui divida).  
- Acerto confiante paga via base; erro confiante penaliza forte; IDK gera frustracao leve.  
- Pensar mais consome budget; nao ha limite de passos, mas ha custo marginal.  
- Budget negativo registra custo extra para analise de eficiencia.

Simulacao de 3 ciclos (intuicao)
--------------------------------
Premissas: delta de satisfacao = +1 progresso | -1 regressao | -0.2 estagnacao; budget por ciclo = -1; confidence sobe com progresso e cai com regressao; reward so no final.
- Cenario A (ideal): progresso, progresso, estagnacao leve -> satisfacao 1, 2, 1.8; confidence baixa->media->alta e estavel; budget -3; decide concluir; correto + alta confidence => reward alto (base de acerto - custo moderado).
- Cenario B (insiste sem evidencias): progresso pequeno, estagnacao, estagnacao -> satisfacao 0.5, 0.3, 0.1; confidence baixa; budget -3; conclui mesmo assim; reward baixo/negativo; aprende que insistir sem evidencias custa.
- Cenario C (IDK honesto): progresso, regressao, regressao -> satisfacao 1, 0, -1; confidence cai; budget -3; decide IDK; reward levemente negativo (melhor que errar confiante); sai vivo.
- Cenario D (pecado capital): progresso falso, progresso falso, estagnacao -> satisfacao 1, 2, 1.8; confidence alta e estavel; budget -3; conclui; errado + alta confidence => reward muito negativo; desincentiva se apaixonar por falso padrao.
- Cenario E (covardia): progresso, progresso, estagnacao -> satisfacao 1, 2, 1.8; confidence alta; decide IDK; reward pior que errar pouco (ou assert em treino); ensina a nao fugir quando sabe.

Memoria externa
---------------
- Guarda pares (estado inicial -> dica/prior/logits ou estado latente inicial).  
- Usada no passo 0 para puxar um ponto de partida eficiente; nao define conceitos nem decide.  
- Pode registrar confianca associada e ser usada como replay/acelerador; avaliada por ganho de convergencia.

Instancias e verificacao
------------------------
- MAX_INSTANCES fisico limita CPU/memoria; custo cognitivo de abrir instancias conta no budget.  
- Cada verificador roda poucos passos e retorna {true|false|unknown} + confianca.  
- Cerebro agrega votos; pode abortar verificadores se satisfacao cair ou budget apertar.

Modalidade e tokenizacao
------------------------
- Texto curto, som 1D (earcons/sonificacao simples), icone 32x32 -> todos como tokens discretos.  
- Mesmo mecanismo de difusao; muda o alfabeto e o embedder.  
- Som tende a convergir mais rapido (1D); icones precisam grid pequeno para manter custo.

Dataset e curriculo
-------------------
- Desafios fechados sem regras explicitas: apenas enunciado e pedido de continuar.  
- Progresso de dificuldade por composicao de regras, nao troca de dominio.  
- Faixas:  
  1) Sequencias simples (letras/numeros, par/imper, ordem direta/reversa).  
  2) Regras compostas (saltos, alternancia, ciclos).  
  3) Matematica basica (progressao, Fibonacci, operacoes simples).  
  4) Imagens discretas (icones/grids, transformacoes simples).  
- Metadados: enunciado canonico, ground truth, reward base opcional, formato de saida; dataset nao define budget.

Treino e pipeline
-----------------
- Pre-treino/fase 1: treinar difusao discreta para reconstruir estados limpos a partir de ruido (teacher-forced).  
- Fase 2: ligar loop cognitivo completo com cabeca de decisao e calibracao de confidence; heuristica apenas para bootstrap/coleta inicial, removida apos a policy aprender.  
- Fase 3: RL/fine-tune com reward final = base fixa de acerto/erro/IDK - custo(budget_used), penalidade para erro confiante e frustracao leve em IDK; logging pesado de entropia, satisfacao, taxa de IDK.  
- Distilacao/bootstrapping: cerebro pequeno ensina varicoes maiores; memoria externa usada como replay opcional.

Experimentos planejados (PoC 64D)
---------------------------------
- Kernel de difusao discreta com T=8-16 passos e ruido de mascara/substituicao; embed 64D; backbone pequeno (Transformer 2-4 camadas, d_model 128-256).  
- Cabecas: predicao de token limpo, confidence interna, decisao stop/continue/IDK condicionada em historico compacto.  
- Ablacoes: sem IDK, sem budget, sem memoria; medir acuracia, custo medio, taxa de erro confiante, taxa de IDK correto.  
- Comparacao: baseline autoregressivo em mesmas tarefas fechadas vs cerebro difusivo.  
- Modalidade: comecar em texto curto, depois som 1D, depois icone 32x32.  
- Verificadores: opcional no PoC; se ativar, medir custo vs ganho em confianca.

Invariantes operacionais
------------------------
- Pensamento e processo iterativo; nao ha geracao direta sem refino.  
- Historico e so para comparacao temporal; nao e memoria semantica.  
- IDK nunca e silencio; e decisao explicita para evitar custo/despesa cognitiva desnecessaria.  
- Executores nao atualizam o cerebro; cerebro nao aprende execucao.  
- Output final = best_candidate + confianca ou <IDK>; sem revisao apos encerrar.

Referencias chaves
------------------
- Austin et al., 2021 - Structured Denoising Diffusion Models in Discrete State-Spaces (D3PM).  
- Li et al., 2022 - Diffusion-LM: Improves Controllable Text Generation.  
- Savinov et al., 2021/2022 - SUNDAE: Step-unrolled Denoising Autoencoders for Text Generation.  
- Nie et al., 2025 - LLaDA: Large Language Diffusion Models.  
- Surveys de difusao para texto: usar na secao Related Work para posicionar o cerebro difusivo vs geracao direta.
