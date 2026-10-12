# Como o cgelab se compara ao que já existe

O levantamento dos outros pacotes foi feito em 9 de setembro de 2026. As colunas do `cgelab`
foram revistas em 25 de setembro de 2026, para a versão 0.2.0. As linhas que mudaram nessa
revisão estão marcadas com †. As colunas dos outros pacotes **não** foram refeitas.

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

| Capacidade | cgelab 0.2 | pycge | PSLmodels/CGE | python-cge | GAMS / GEMPACK (ref.) |
|---|---|---|---|---|---|
| **Origem** | escrito para uma MCS aplicada | ✅ porte do `splcge.gams` da biblioteca do GAMS | ✅ "o modelo CGE mais simples de Hosoe, Gawana e Hashimoto (2010)" | ✅ transcrição direta do livro de Hosoe, p. 72 | plataformas de propósito geral |
| **Escala demonstrada** | ✅ a MCS aplicada para a qual foi escrito, ainda não pública; em público, exemplos didáticos | ✅ 3 exemplos didáticos e um regional | ✅ MCS de **16×18**: 4 setores, 3 fatores | ✅ **2 bens** (`BRD`, `MLK`) | milhares de contas (GTAP) |
| Backend | ✅ `scipy.optimize.root` sobre poucas incógnitas | ✅ Pyomo + IPOPT | ✅ `scipy` `root` | ✅ próprio | resolvedores dedicados |
| Linhas de Python | ~2.900 † | ✅ 3.267 | ✅ 2.660 | ✅ 533 | n/a |
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
| **Fechamentos alternativos** | ✅ 5 combinações, cada uma com a sua variável de ajuste | ? | — | — | ✅ |
| **Walras como verificação e não como equação** | ✅ mercado do numerário fora do sistema | ? | — | — | convenção do modelador |
| **Teste de reprodução do ano-base** | ✅ automático | 📄 "compara base e simulação" | — | — | convenção do modelador |
| **Teste de homogeneidade de grau zero** | ✅ automático | — | — | — | convenção do modelador |
| **Dinâmica recursiva** | ✅ vários estoques de capital, alocação pelo retorno, economia por ano † | — | ✅ laço de iteração temporal | — | ✅ |
| **Bem-estar: VE e VC** | ✅ | — | — | — | convenção do modelador |
| Agregação por grupo imune à composição | ✅ | — | — | — | — |
| **Réplica independente para validação** | ✅ † JuMP/Ipopt, 19 casos | ? | ? | ? | n/a |
| Licença | MIT | ? | CC0-1.0 | ? | proprietária/comercial |
| Estado declarado pelos autores | — | — | ✅ "muito preliminar… por favor não cite" | livro-texto | maduro |

## O que a tabela diz

Nas linhas novas (†), as colunas dos outros pacotes estão como "?" porque não foram reapuradas.

Os três pacotes de EGC em Python encontrados implementam o modelo-texto de Hosoe, Gawana e
Hashimoto (2010) ou o `splcge` da biblioteca do GAMS, com dois a quatro setores. O `PSLmodels/CGE`
diz de si mesmo, no README, que é "muito preliminar" e pede que não seja citado. Até onde se apurou,
não há pacote de EGC em Python em escala aplicada.

Isso não faz do `cgelab` uma contribuição por si. Ele preenche uma lacuna que existe porque quem
faz EGC aplicado usa GAMS ou GEMPACK, e com bons motivos: resolvedores dedicados, comunidade e
bases como a GTAP. A afirmação defensável é estreita. Para um EGC de escala aplicada em Python, sem
licença comercial e com as verificações contábeis embutidas, não se achou alternativa.

## Uma advertência

O `cgelab` foi validado de três formas:

- contra uma réplica independente em JuMP/Ipopt do mesmo equilíbrio, em 19 casos;
- contra o pipeline de pesquisa que o originou, na versão 0.1.0;
- pela reprodução do ano-base, pela homogeneidade e pela lei de Walras à precisão de máquina, na
  MCS aplicada para a qual foi escrito, que ainda não é pública.

Isso é evidência de que o que ele faz, faz certo. **Não** é evidência de que cubra o que um GAMS
cobre. O ninho de várias regiões não tem migração de fatores, e o pacote não estima
elasticidades, não tem expectativas racionais, nem a maturidade de uma plataforma com trinta
anos de uso.

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
| Tarifa e imposto indireto separados do imposto sobre produtos | 📄 Hosoe (`stdcge`), IFPRI | **feito** em torno da alíquota calibrada; falta a tarifa dentro do preço de importação do ano-base | 0.2 |
| Exportador JuMP convergindo a frio em escala aplicada | — | falta: converge a frio nas economias de brinquedo e, numa MCS aplicada, só com partida quente | 0.2 |
| Demanda de exportação e oferta de importação com elasticidade finita | 📄 Hosoe (`lrgcge`) | **feito**: as equações do `lrgcge`, validadas contra a réplica JuMP; falta a réplica do `lrgcge` em si, que espera a tarifa no preço de importação do ano-base | 0.2 |
| Decomposição de resultados por choque (*subtotals*) | 📄 GEMPACK | falta | 0.3 |
| Sensibilidade sistemática (quadratura, Monte Carlo) | 📄 GEMPACK | falta | 0.3 |
| Contas e relatórios: PIB por despesa e renda, índices de preço, termos de troca | 📄 GEMPACK, GTAP | `real_gdp` feito; o resto falta | 0.3 |
| Decomposição do bem-estar (Huff e Hertel) | 📄 GTAP | falta | 0.3 |
| Multiplicadores da MCS e insumo-produto | 📄 Pyatt e Round; Defourny e Thorbecke | falta | 0.3 |
| Ninhos de energia (KLEM), emissões, imposto de carbono | 📄 ENVISAGE, EPPA, GTAP-E | falta | 0.4 |
| Fechamentos do governo | 📄 IFPRI | falta | 0.4 |
| Ponte com o `dsgelab` | — | falta | 0.4 |
| Mobilidade imperfeita de fatores; informalidade | 📄 IFPRI, ORANI | falta | 0.6 |
| Elo com microssimulação; Gini e FGT | 📄 prática corrente | falta | 0.6 |
| Calibração dinâmica da linha de base; capital setorial e safras | 📄 MONASH, GDyn, LINKAGE | falta | 0.6 |
| Construção e desagregação de MCS | 📄 IFPRI, PEP | falta | 0.6 |
| Múltiplas regiões | 📄 GTAP, TERM, ORANI-G | ninho de origem feito; migração, fechamentos regionais e comércio bilateral faltam | 0.7 |
| Dados do GTAP | 📄 GTAPAgg, FlexAgg | falta | 0.7 |
| Complementaridade (MCP); resolvedores alternativos | 📄 GAMS/MPSGE com PATH; pycge (Pyomo) | falta | 0.8 |
| Troca de fechamento genérica; entropia cruzada; EGC intertemporal; concorrência imperfeita; uso da terra; exportar para GAMS e GEMPACK | 📄 GEMPACK, MPSGE, GTAP-AEZ | falta | 0.9 |


## Fontes

- [pycge](https://github.com/juanfung/pycge), [PSLmodels/CGE](https://github.com/PSLmodels/CGE),
  [python-cge](https://github.com/DavoudTaghawiNejad/python-cge): clonados e lidos
- [Banco Mundial sobre EGC](https://www.worldbank.org/en/programs/economicpolicy-macro-modeling/cge)
- [GEMPACK](https://en.wikipedia.org/wiki/Gempack)
