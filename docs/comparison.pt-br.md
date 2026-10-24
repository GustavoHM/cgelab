# Como o cgelab se compara ao que já existe

*English version: [`comparison.md`](comparison.md).*

O levantamento dos outros pacotes foi feito em 9 de setembro de 2026. As colunas do `cgelab`
foram revistas em 22 de outubro de 2026, para a versão 0.6. As linhas que mudaram desde a 0.1
estão marcadas com †. As colunas dos outros pacotes **não** foram refeitas,
exceto a célula de dinâmica recursiva do `PSLmodels/CGE`, corrigida em 22 de outubro de 2026
depois de uma nova leitura do código.

A pergunta que originou este levantamento foi se já existia um pacote de EGC em Python. A resposta
curta é que há alguns, todos em escala de livro-texto, e que o EGC aplicado se faz em GAMS ou
GEMPACK.

## Como ler a tabela

| Marca | Significa |
|---|---|
| **✅** | verificado: o pacote foi instalado ou o código foi clonado e lido |
| **📄** | consta da documentação oficial, mas **não** foi verificado no código |
| **—** | ausente até onde se apurou |
| **?** | não apurado |

A distinção importa. O código de `pycge`, `PSLmodels/CGE` e `python-cge` foi clonado e lido. Uma
célula **📄** pode estar errada.

## A tabela

| Capacidade | cgelab 0.6 | pycge | PSLmodels/CGE | python-cge | GAMS / GEMPACK (ref.) |
|---|---|---|---|---|---|
| **Origem** | escrito para uma MCS aplicada | ✅ porte do `splcge.gams` da biblioteca do GAMS | ✅ "o modelo CGE mais simples de Hosoe, Gawana [sic] e Hashimoto (2010)" | ✅ transcrição direta do livro de Hosoe, p. 72 | plataformas de propósito geral |
| **Escala demonstrada** | ✅ a MCS aplicada para a qual foi escrito, ainda não pública; em público, exemplos didáticos | ✅ 3 exemplos didáticos e um regional | ✅ MCS de **16×18**: 4 setores, 3 fatores | ✅ **2 bens** (`BRD`, `MLK`) | milhares de contas (GTAP) |
| Backend | ✅ `scipy.optimize.root` sobre poucas incógnitas | ✅ Pyomo + IPOPT | ✅ `scipy` `root` | ✅ próprio | resolvedores dedicados |
| Linhas de Python | ~11.800 † | ✅ 3.267 | ✅ 2.660 | ✅ 533 | n/a |
| **MCS: leitura, verificação de balanço, balanceamento** | ✅ formato longo ou quadrado, porta dura, RAS que preserva sinais | ? | ✅ lê `SAM.xlsx` | ✅ tem `balance.py` | ✅ |
| Fluxos da MCS que o modelo não lê | ✅ listados na calibração † | ? | ? | ? | convenção do modelador |
| Ninho de valor adicionado configurável (CES / CD / Leontief) | ✅ | ? | ✅ CES e Leontief | — | ✅ |
| Armington | ✅ | ✅ | ✅ | — | ✅ |
| CET nas exportações | ✅ † resolvida, com preço mundial de exportação; o padrão fixa participações | ? | ✅ | — | ✅ |
| **Bloco de margens de comércio e transporte** | ✅ | ? | — | — | ✅ |
| **Valor adicionado negativo tratado** | ✅ duas regras | ? | — | — | — |
| Demanda LES / Stone-Geary | ✅ calibrada pelo parâmetro de Frisch | ? | — | — | ✅ |
| **Atividades de conta própria** | ✅ detectadas e escaláveis | — | — | — | — |
| Split creditável/não creditável do tributo | ✅ | — | — | — | ✅ |
| **Impostos diretos, transferências e remessas** | ✅ † | ? | ? | ? | ✅ |
| **Capital de propriedade estrangeira e IED** | ✅ † remessa de lucros na conta corrente | ? | ? | ? | ✅ |
| **Mobilidade imperfeita de fatores** | ✅ † CET ou logit, pools de origem | ? | ? | ? | ✅ os fatores "sluggish" do GTAP |
| **Trabalho segmentado e informalidade** | ✅ † mercado dual, Harris–Todaro, transformação entre os segmentos | ? | ? | ? | convenção do modelador |
| **Microssimulação, pobreza e desigualdade** | ✅ † de cima para baixo e comportamental (BRR) | ? | ? | ? | ferramentas à parte |
| **MCS montada de tabelas de recursos e usos** | ✅ † com a procedência de cada bloco; as tabelas do IBGE | ? | ? | ? | ferramentas à parte |
| **Fechamentos alternativos** | ✅ 5 combinações, cada uma com a sua variável de ajuste | ? | — | — | ✅ |
| **Walras como verificação e não como equação** | ✅ mercado do numerário fora do sistema | ? | — | — | convenção do modelador |
| **Teste de reprodução do ano-base** | ✅ automático | 📄 "compara base e simulação" | — | — | convenção do modelador |
| **Teste de homogeneidade de grau zero** | ✅ automático | — | — | — | convenção do modelador |
| **Dinâmica recursiva** | ✅ vários estoques de capital, capital por setor, safras putty-clay, alocação pelo retorno, economia por ano, linha de base calibrada ao PIB † | — | — o laço em `execute.py` é um ponto fixo amortecido dentro de um período, sem índice de tempo | — | ✅ |
| **Bem-estar: VE e VC** | ✅ | — | — | — | convenção do modelador |
| Agregação por grupo imune à composição | ✅ | — | — | — | — |
| **Réplica independente para validação** | ✅ † JuMP/Ipopt, 50 casos | ? | ? | ? | n/a |
| Licença | MIT | ? | CC0-1.0 | ? | proprietária/comercial |
| Estado declarado pelos autores | — | — | ✅ "muito preliminar… por favor não cite" | livro-texto | maduro |

## O que a tabela diz

Nas linhas novas (†), as colunas dos outros pacotes estão como "?" porque não foram reapuradas.

Os três pacotes de EGC em Python encontrados implementam o modelo-texto de Hosoe, Gasawa e
Hashimoto (2010) ou o `splcge` da biblioteca do GAMS, com dois a quatro setores. O `PSLmodels/CGE`
diz de si mesmo, no README, que é "muito preliminar" e pede que não seja citado. Até onde se apurou,
não há pacote de EGC em Python em escala aplicada.

Isso não faz do `cgelab` uma contribuição por si. Ele preenche uma lacuna que existe porque quem
faz EGC aplicado usa GAMS ou GEMPACK, e com bons motivos: resolvedores dedicados, comunidade e
bases como a GTAP. A afirmação defensável é estreita. Para um EGC de escala aplicada em Python, sem
licença comercial e com as verificações contábeis embutidas, não se achou alternativa.

## Uma advertência

O `cgelab` foi validado de quatro formas:

- contra uma réplica independente em JuMP/Ipopt do mesmo equilíbrio, em 50 casos;
- contra uma transcrição linha a linha, em JuMP, dos programas `stdcge` e `lrgcge` da
  biblioteca do GAMS (Hosoe, Gasawa e Hashimoto, 2004): a abolição das tarifas coincide nas 49
  variáveis exibidas, a 2,4e-12;
- contra o pipeline de pesquisa que o originou, na versão 0.1.0;
- pela reprodução do ano-base, pela homogeneidade e pela lei de Walras à precisão de máquina, na
  MCS aplicada para a qual foi escrito, que ainda não é pública.

Isso é evidência de que o que ele faz, faz certo. **Não** é evidência de que cubra o que um GAMS
cobre. O ninho de várias regiões não tem migração de fatores nem comércio bilateral, não há
complementaridade, o pacote não estima elasticidades, as expectativas racionais passam pela ponte
com o `dsgelab`, e não há a maturidade de uma plataforma com trinta anos de uso.

---

## Lacunas, também contra plataformas de outras linguagens

A tabela acima compara com pacotes em Python. Esta seção amplia a régua para GAMS/MPSGE, GEMPACK,
GTAP e os modelos-padrão do IFPRI e do PEP, que não estão instalados aqui: as células vêm da
literatura e da documentação, e são **📄**. Desde 11 de outubro de 2026 o objetivo é cobrir toda a
lista até a 1.0; a versão de cada item é a do [`ROADMAP.md`](../ROADMAP.md).

| Item | Quem tem | Estado no `cgelab` | Versão |
|---|---|---|---|
| Instituições da MCS: impostos diretos, transferências, remessas, renda de fatores ao governo e ao exterior | 📄 IFPRI, PEP | **feito** na 0.2.0 | 0.2 |
| Fechamentos do mercado de trabalho | 📄 IFPRI, ORANI | **feito**: salário real fixo, curva de salário, oferta elástica | 0.2 |
| Tarifa e imposto indireto separados do imposto sobre produtos | 📄 Hosoe (`stdcge`), IFPRI | **feito**: a tarifa dentro do preço de importação e o imposto indireto sobre todo o produto, como no `stdcge`; a abolição de tarifas do `stdcge` e do `lrgcge` reproduz a transcrição dos programas do GAMS a 2,4e-12 | 0.2 |
| Exportador JuMP convergindo a frio em escala aplicada | — | **feito**, numa MCS aplicada ainda não pública | 0.2 |
| Demanda de exportação e oferta de importação com elasticidade finita | 📄 Hosoe (`lrgcge`) | **feito**: as equações do `lrgcge`; a abolição de tarifas do `lrgcge` reproduz a transcrição do programa do GAMS | 0.2 |
| Decomposição de resultados por choque (*subtotals*) | 📄 GEMPACK | **feito**: Harrison, Horridge e Pearson, integrada ao longo da trajetória reta | 0.3 |
| Sensibilidade sistemática e robustez | 📄 GEMPACK | **feito**: quadratura de Stroud de grau 3 e Monte Carlo, com recalibração a cada ponto; índices de Sobol, grades de fechamento, resultados em moeda, sorteios equilibrados da MCS com elasticidades em conjunto e relatório em uma chamada com faixas documentadas e figuras opcionais | 0.3, 0.6 |
| Ajuste por distância mínima a alvos externos | 📄 prática de estimação e validação | **feito**: campos do ModelSpec com limites ajustados por distância ponderada pela covariância, com pseudo-inversa e evidência do otimizador; exato contra mínimos quadrados ponderados em forma fechada | 0.7 |
| Contas e relatórios: PIB por despesa e renda, índices de preço, termos de troca | 📄 GEMPACK, GTAP | **feito**: PIB pelos dois lados, índices de Laspeyres, Paasche e Fisher, termos de troca, conta do governo | 0.3 |
| Decomposição do bem-estar (Huff e Hertel) | 📄 GTAP | **feito** para a absorção real, com a identidade fechando abaixo de 1e-9; a comparação com o GTAP espera a base de dados | 0.3 |
| Multiplicadores da MCS e insumo-produto | 📄 Pyatt e Round; Defourny e Thorbecke | **feito**: contábeis, a preço fixo, Leontief, encadeamentos (Rasmussen e Ghosh), análise de caminhos estruturais | 0.3 |
| Ninhos de energia (KLEM), emissões, imposto de carbono | 📄 ENVISAGE, EPPA, GTAP-E | **feito**: ninho valor adicionado-energia ou capital-energia (o arranjo do GTAP-E) com substituição entre combustíveis, emissões por fonte, preço do carbono no preço do combustível de cada atividade, reciclagem neutra em arrecadação, unidades físicas | 0.4 |
| Fechamentos do governo | 📄 IFPRI | **feito**: taxa de poupança fixa (a de Hosoe), meta de poupança e receita fixa (substituição neutra em arrecadação), atingidas pelo imposto direto, pelo imposto sobre produtos ou pelo gasto | 0.4 |
| Poupança externa fixa em moeda estrangeira | 📄 Hosoe, IFPRI | **feito** | 0.2 |
| Ponte com o `dsgelab` | — | **feito**: CGE de Ramsey com investimento por Euler; estado estacionário, primeira ordem, previsão perfeita e política anunciada; igual ao modelo de crescimento numa economia de um bem | 0.4 |
| Mobilidade imperfeita de fatores; matriz de proximidade | 📄 GTAP, ORANI | **feito**: CET (os fatores "sluggish" do GTAP) ou logit entre as atividades, pools de origem como matriz de proximidade; exato contra fatores específicos separados em ω = 0 e contra a réplica JuMP | 0.6 |
| Segmentação e informalidade | 📄 IFPRI; Harris e Todaro | **feito**: contas de trabalho que dividem uma população; mercado único, mercado dual e Harris–Todaro, ou CET ou logit entre os segmentos com fila em cada rígido; renda que segue os trabalhadores de cada domicílio; exato contra dois fatores com salário real fixo em ω = 0 e contra a réplica JuMP | 0.6 |
| Elo com microssimulação; Gini e FGT | 📄 prática corrente | **feito**: domicílios da pesquisa reconciliados com a MCS por conta e fonte, com o próprio índice de preços; Gini, FGT, entropia generalizada e Atkinson com decomposições exatas; pesos recalibrados por raking para o emprego por segmento e o desemprego do modelo. Microssimulação comportamental (BRR): equações de rendimento e logit multinomial da escolha ocupacional, ligados por interceptos, com os sorteios condicionais ou sua expectativa em forma fechada. leitores da PNAD Contínua e da POF do IBGE | 0.6 |
| Calibração dinâmica e histórica; capital setorial e safras | 📄 MONASH, GDyn, LINKAGE | **feito**: produtividade que atinge a trajetória do PIB real e fechamento histórico limitado com deslocadores conjuntos de tecnologia e preferências relativas, com política sobre a trajetória calibrada; capital por setor com investimento repartido pelos rendimentos; safras putty-clay contra a réplica JuMP | 0.6 |
| Construção e desagregação de MCS | 📄 IFPRI, PEP | **feito**: construção a partir das tabelas de recursos e usos, com a procedência de cada bloco, e desagregação das famílias e do trabalho por pesquisas domiciliares | 0.6 |
| Múltiplas regiões | 📄 GTAP, TERM, ORANI-G | ninho de origem feito; migração, fechamentos regionais e comércio bilateral faltam | 0.7 |
| Dados do GTAP | 📄 GTAPAgg, FlexAgg | leitura e escrita de HAR feitas, conferidas contra o harpy3 no GTAP 10A; agregação e MCS de uma região faltam | 0.7 |
| Complementaridade (MCP); resolvedores alternativos | 📄 GAMS/MPSGE com PATH; pycge (Pyomo) | falta | 0.8 |
| Troca de fechamento genérica; entropia cruzada; EGC intertemporal; concorrência imperfeita; uso da terra; exportar para GAMS e GEMPACK | 📄 GEMPACK, MPSGE, GTAP-AEZ | falta | 0.9 |


## Fontes

- [pycge](https://github.com/juanfung/pycge), [PSLmodels/CGE](https://github.com/PSLmodels/CGE),
  [python-cge](https://github.com/DavoudTaghawiNejad/python-cge): clonados e lidos
- [Banco Mundial sobre EGC](https://www.worldbank.org/en/programs/economicpolicy-macro-modeling/cge)
- [GEMPACK](https://en.wikipedia.org/wiki/Gempack)
