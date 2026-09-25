# Como o cgekit se compara ao que já existe

O levantamento dos outros pacotes foi feito em 9 de setembro de 2026. As colunas do `cgekit`
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

| Capacidade | cgekit 0.2 | pycge | PSLmodels/CGE | python-cge | GAMS / GEMPACK (ref.) |
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
| **Réplica independente para validação** | ✅ † JuMP/Ipopt, 16 casos | ? | ? | ? | n/a |
| Licença | MIT | ? | CC0-1.0 | ? | proprietária/comercial |
| Estado declarado pelos autores | — | — | ✅ "muito preliminar… por favor não cite" | livro-texto | maduro |

## O que a tabela diz

Nas linhas novas (†), as colunas dos outros pacotes estão como "?" porque não foram reapuradas.

Os três pacotes de EGC em Python encontrados implementam o modelo-texto de Hosoe, Gawana e
Hashimoto (2010) ou o `splcge` da biblioteca do GAMS, com dois a quatro setores. O `PSLmodels/CGE`
diz de si mesmo, no README, que é "muito preliminar" e pede que não seja citado. Até onde se apurou,
não há pacote de EGC em Python em escala aplicada.

Isso não faz do `cgekit` uma contribuição por si. Ele preenche uma lacuna que existe porque quem
faz EGC aplicado usa GAMS ou GEMPACK, e com bons motivos: resolvedores dedicados, comunidade e
bases como a GTAP. A afirmação defensável é estreita. Para um EGC de escala aplicada em Python, sem
licença comercial e com as verificações contábeis embutidas, não se achou alternativa.

## Uma advertência

O `cgekit` foi validado de três formas:

- contra uma réplica independente em JuMP/Ipopt do mesmo equilíbrio, em 16 casos;
- contra o pipeline de pesquisa que o originou, na versão 0.1.0;
- pela reprodução do ano-base, pela homogeneidade e pela lei de Walras à precisão de máquina, na
  MCS aplicada para a qual foi escrito, que ainda não é pública.

Isso é evidência de que o que ele faz, faz certo. **Não** é evidência de que cubra o que um GAMS
cobre. O `cgekit` não tem múltiplas regiões, nem estimação de elasticidades, nem expectativas
racionais, nem a maturidade de uma plataforma com trinta anos de uso.

## Fontes

- [pycge](https://github.com/juanfung/pycge), [PSLmodels/CGE](https://github.com/PSLmodels/CGE),
  [python-cge](https://github.com/DavoudTaghawiNejad/python-cge): clonados e lidos
- [Banco Mundial sobre EGC](https://www.worldbank.org/en/programs/economicpolicy-macro-modeling/cge)
- [GEMPACK](https://en.wikipedia.org/wiki/Gempack)
