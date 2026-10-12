# A matemática do cgelab

Nota técnica do equilíbrio que o pacote calibra e resolve. As fórmulas são as de `src/cgelab`, na versão que lê uma matriz de contabilidade social e devolve um equilíbrio de período, uma trajetória recursiva e uma variação equivalente. Onde uma convenção alternativa mudaria o sinal de um resultado, o texto registra a que está no código.

## 1. Três objetos

O ano-base é uma matriz de contabilidade social \(S\). A forma funcional e as elasticidades são um `ModelSpec`. O fechamento, isto é, quais saldos são impostos e quem ajusta, é um `Closure`. Os dois últimos viajam com o resultado. Um imposto reciclado para o capital e o mesmo imposto reciclado em parcela única são experimentos diferentes, e a diferença tem de ser legível no objeto que produziu o número, não num ramo interno do solver.

\(S\) é quadrada. Cada conta recebe o que as demais lhe pagam, e a soma da linha iguala a soma da coluna dentro de uma tolerância absoluta. Um desbalanceamento é um erro, não um aviso. O modelo não impõe depois um saldo que a matriz não tinha.

## 2. Preços do ano-base

A matriz registra compras a preço de adquirente, já com o imposto indireto. As quantidades do modelo estão em unidades do preço básico do ano-base. O bloco de insumos intermediários, de mercadorias para atividades, é portanto deflacionado na calibração,

\[
io_{c,a} = \frac{S_{c,a}}{X_{0,a}\,(1+t_c)},
\]

com \(X_{0,a}\) o recebimento total da atividade. Omitir \(1+t_c\) faz o imposto aparecer duas vezes. O sintoma não é uma exceção. O ano-base deixa de se reproduzir.

A base do imposto sobre a absorção é a quantidade doméstica mais a importada mais as margens que entram na mercadoria, \(A_0 = Q^{\mathrm{dom}}_0 + (1+t_m^0)\,M_0 + \text{margem}\). O preço de adquirente de base é \(1+t\), e a alíquota é o imposto dividido por \(A_0\). Dividi-lo pelo total da linha da mercadoria, que já contém o imposto, produz outra alíquota.

Uma conta nomeada de tarifa ou de imposto indireto sobre o produto é dinheiro que o comprador pagou e o governo recebeu. As duas saem do imposto de produto e entram no preço sobre o qual incidem, como no `stdcge` de Hosoe, Gasawa e Hashimoto. A alíquota da tarifa é \(t_m^0 = T_m / M_0\), e a importação entra no Armington pelo valor com tarifa, \((1+t_m^0)\,M_0\). A alíquota do imposto indireto é \(t_z^0 = T_z / Z_0\), e a produção que a CET reparte vale \((1+t_z^0)\,Z_0\). As vendas domésticas são esse valor menos as exportações. O imposto de produto que resta é o recebimento do governo menos as duas receitas, dividido por \(A_0\). O preço de adquirente da matriz não muda. O pagamento dessas contas de volta ao governo é o mesmo recebimento, não um segundo.

## 3. Calibração das participações

A produção de mercadorias pelas atividades é o bloco \(S_{a,c}\). A participação de mercado é fixa no ano-base,

\[
MSH_{a,c} = \frac{S_{a,c}}{Q_{0,c}}, \qquad Q_{0,c} = \sum_a S_{a,c},
\]

e o nível da atividade segue da produção de mercadorias, \(X = MSH\, Q\). Uma atividade sem participação em mercadoria alguma e com recebimento positivo, típica de produção para uso próprio, não é recuperável por esse produto. Ela é carregada no nível de base e só um cenário a escala.

O valor adicionado por unidade de atividade é a soma dos pagamentos a fatores dividida por \(X_0\). As participações \(\theta_{f,a}\) são esses pagamentos normalizados pela coluna, e alimentam o ninho CES ou Cobb–Douglas. Uma coluna com pagamento negativo a um fator não é uma participação. Sob a regra de Leontief a atividade inteira passa a proporções fixas. Sob a regra líquida os pagamentos positivos são reescalados para somar o valor adicionado líquido, e a oferta agregada do fator cai no mesmo montante. Sem esse ajuste o ano-base não fecha.

A participação doméstica no composto Armington, antes das margens, é \(Q^{\mathrm{dom}}_0 / (Q^{\mathrm{dom}}_0 + M_0)\). A participação das exportações na produção é \(E_0 / Q_0\). As duas são os \(\theta\) dos ninhos da seção 4. A elasticidade de transformação das exportações não sai da matriz. O padrão fixa as participações de exportação. A CET entra como variante.

A poupança das famílias é uma fração da renda disponível, renda bruta menos imposto direto. Sem imposto direto a taxa coincide com a fração da renda bruta. A renda bruta é o total da linha da conta, não uma lista escolhida de colunas. Pagamentos ao resto do mundo são líquidos dos recebimentos do resto do mundo e saem do orçamento antes do imposto, da poupança e do consumo.

A renda dos fatores que não chega às famílias residentes, porque vai ao governo ou ao exterior, é uma participação do pagamento total do fator. Creditá-la às famílias produziria renda que elas não receberam, e o ano-base não se reproduziria.

Margens de comércio e transporte aparecem no bloco mercadoria-mercadoria como vendas das mercadorias de margem para as demais, com um lançamento negativo grande na diagonal. A diagonal é um offset contábil dos dois lados do mesmo saldo e é descartada. O que resta, por unidade de absorção e em preço básico, é a matriz \(mq\). Ignorar o bloco perde esse valor, e o resíduo aparece numa mercadoria só.

### Stone–Geary

Com demanda Cobb–Douglas as quantidades de subsistência são zero. Com LES, o parâmetro de Frisch \(\phi < 0\) define a renda supernumerária \(Y^\star = -Y/\phi\), onde \(Y\) é o gasto em mercadorias no ano-base. As participações marginais são as participações orçamentárias reponderadas pelas elasticidades-renda e renormalizadas para somar um. A quantidade comprometida segue de

\[
p_c \gamma_c = p_c q_c - \beta_c Y^\star,
\]

com \(p_c = 1+t_c\) o preço de adquirente do ano-base. Uma \(\gamma\) negativa é uma contradição da calibração, não um número a ser clipado em silêncio. Cobb–Douglas é o LES sem subsistência.

## 4. Ninhos de preço e de quantidade

Um ninho CES com participações \(\theta_i\) que somam um, elasticidade \(\sigma\) e preços \(p_i\) tem custo unitário

\[
P = \Big(\sum_i \theta_i\, p_i^{1-\sigma}\Big)^{1/(1-\sigma)}.
\]

O insumo por unidade do composto é \(\theta_i (P/p_i)^{\sigma}\). Em \(\sigma = 1\) a expressão é o limite de Cobb–Douglas, \(P = \prod_i p_i^{\theta_i}\), calculado diretamente, não por uma aproximação da fórmula CES. Participação nula contribui zero e não produz \(0^0\). Uma coluna cuja soma de participações é zero devolve custo zero.

O valor adicionado de uma atividade é esse ninho sobre os fatores, com elasticidade por atividade. Atividades marcadas como Leontief usam os coeficientes calibrados \(a_{f,a}/v_a\), o pagamento do fator por unidade de valor adicionado. A produtividade de um cenário escala esses coeficientes. O ramo CES é homogêneo de grau zero nessa escala. O ramo de proporções fixas é uma razão dos dois coeficientes e precisa ver os dois.

A produção combina o agregado intermediário e o valor adicionado em Leontief. Dentro dos intermediários as proporções também são fixas. O preço da atividade é

\[
p_a = \sum_c io_{c,a}\, p^f_c + p^{va}_a\, v_a + \tau_a,
\]

com \(p^f\) o preço de adquirente pago pela firma, \(v_a\) o valor adicionado por unidade e \(\tau_a\) o imposto específico sobre a atividade. O preço básico da mercadoria, pelo lado da oferta, é a média das atividades que a produzem, \(p_d = MSH^{\mathsf T} p_a\).

### Armington

Para cada mercadoria, com participação doméstica \(s_d\), preço doméstico \(p_d\), preço de importação \(p_m\) e elasticidade \(\sigma_A\),

\begin{align*}
p_q &= \big(s_d\, p_d^{1-\sigma_A} + (1-s_d)\, p_m^{1-\sigma_A}\big)^{1/(1-\sigma_A)}, \\
d &= s_d \big(p_q / p_d\big)^{\sigma_A}, \qquad
m &= (1-s_d) \big(p_q / p_m\big)^{\sigma_A}.
\end{align*}

\(d\) e \(m\) são intensidades por unidade do composto. O preço de importação de fronteira é a taxa de câmbio vezes o preço mundial. A importação é medida pelo valor com tarifa do ano-base, então o seu preço no Armington é o de fronteira vezes \((1+t_m)/(1+t_m^0)\), que vale 1 no ano-base. O governo recebe o nível \(t_m\, p_m^{\mathrm{fronteira}} M\), com \(M\) a importação na fronteira. Uma tarifa zero a abole, e a variedade doméstica nunca a pagou.

### Exportações

O imposto indireto incide sobre toda a produção. O preço bruto que a fronteira reparte é o do produtor vezes \((1+t_z)/(1+t_z^0)\), e as vendas domésticas e as exportações o pagam. O governo recebe o nível \(t_z\, p_z Z\). Com participações fixas, a fração exportada é a do ano-base e o preço recebido é o preço bruto. Com CET de elasticidade \(\sigma_T > 0\), participação de exportação \(s_e\) e preços \(p_d\), \(p_e\),

\begin{align*}
p_z &= \big((1-s_e)\, p_d^{1+\sigma_T} + s_e\, p_e^{1+\sigma_T}\big)^{1/(1+\sigma_T)}, \\
\text{fração doméstica} &= (1-s_e)\,(p_d / p_z)^{\sigma_T}, \\
\text{fração exportada} &= s_e\,(p_e / p_z)^{\sigma_T}.
\end{align*}

Sob retornos constantes o preço do produtor é o custo unitário. A fronteira de transformação não determina esse preço. Ela reparte a produção. Dado o preço bruto do produto e o preço de exportação, o preço doméstico é o que iguala \(p_z\) ao preço bruto, a inversa da fórmula acima.

O preço de exportação é a taxa de câmbio vezes o preço mundial. Uma economia tomadora de preço recebe o preço mundial do cenário. Com elasticidade de demanda \(\eta\), a exportação fica numa curva, \(E/E_0 = (p^W_e/\bar p^W_e)^{-\eta}\), em que \(\bar p^W_e\) é o preço do cenário. A oferta de importação com elasticidade \(\psi\) é o espelho, \(M/M_0 = (p^W_m/\bar p^W_m)^{\psi}\). São as equações do `lrgcge` de Hosoe. Cada mercadoria numa curva acrescenta o log do seu preço mundial às incógnitas, e a curva é o resíduo dela. Uma mercadoria sem exportação, ou exportada por inteiro, vende ao preço do produto no mercado que lhe resta. Se o preço externo é alto o bastante para que nenhum preço doméstico positivo satisfaça a fronteira, as vendas domésticas vão a zero.

O preço de adquirente que a demanda vê incorpora margens. As margens são elas próprias mercadorias entregues, então o preço é um ponto fixo,

\[
p^{\mathrm{base}} = p_q^{\mathrm{arm}}\, s^{\mathrm{arm}} + p^f\, mq, \qquad
p^f = p^{\mathrm{base}}\,(1+t^f),
\]

iterado até o máximo absoluto da variação cair abaixo de \(10^{-14}\). Um salário real fixo entra no mesmo laço. O salário é o índice de preços de adquirente vezes a meta, e o índice é feito dos preços que o salário ajuda a determinar. Parar só no gap de preços deixaria o salário uma iteração atrás, e a jacobiana de diferenças finitas do solver externo leria um salto.

## 5. Quantidades de um período

Há mais mercadorias do que atividades num sistema aplicado. Os saldos de mercadoria não determinam os níveis de atividade diretamente. Esse sistema seria sobredeterminado, e resolvê-lo por mínimos quadrados seria aritmética no lugar da economia. Os saldos determinam a absorção de mercadorias, e as atividades seguem das participações de mercado.

Seja \(A\) a absorção, \(d\) a intensidade doméstica do Armington, \(s^{\mathrm{arm}}\) a participação do composto Armington na absorção e \(f^{\mathrm{dom}}\) a fração da produção vendida no mercado interno. Então

\[
G = \mathrm{diag}\!\left(\frac{d \odot s^{\mathrm{arm}}}{f^{\mathrm{dom}} \odot (1+t_z^0)}\right), \qquad
B = io\, MSH, \qquad
Q = G A, \qquad
X = MSH\, Q.
\]

A absorção cumpre

\[
(I - BG - mq)\, A = \text{demanda final}.
\]

A demanda final de consumo e de governo é taxada à alíquota do comprador. O investimento e os intermediários exógenos são taxados à alíquota da firma. \(Q\) é a produção doméstica de mercadorias que abastece o mercado interno. As importações na fronteira são \(m \odot s^{\mathrm{arm}} \odot A / (1+t_m^0)\). As exportações são a fração de exportação vezes \((1+t_z^0) \odot Q\).

A demanda de fatores por atividade é a intensidade do ninho de valor adicionado vezes o valor adicionado da atividade. A demanda agregada do fator é a soma sobre as atividades.

### Famílias

O preço efetivo da mercadoria para a família é o preço de adquirente vezes \(1+w\), onde \(w\) soma a cunha privada e a cunha do governo. Sob Cobb–Douglas a quantidade é a participação orçamentária vezes o gasto em mercadorias, dividida pelo preço efetivo. Sob LES,

\[
q = \gamma + \beta\, \frac{Y^{\mathrm{sup}}}{p^{\mathrm{ef}}},
\]

com \(Y^{\mathrm{sup}}\) o gasto em mercadorias menos o valor comprometido \(\,p^{\mathrm{ef}}\cdot\gamma\).

Creditar a receita da cunha de volta à renda aumenta o gasto, que aumenta a cunha. A demanda é afim no gasto nos dois sistemas, então a receita da família \(h\) é \(A_h + B_h \times \text{gasto}_h\), e a série soma-se em forma fechada. Iterá-la deixaria descontinuidades pequenas que a jacobiana externa de diferenças finitas leria como quinas. Compras diretas a uma atividade, fora do mercado de mercadorias, saem do orçamento antes desse gasto. Um orçamento que não as cobre é um erro. Cortá-lo em zero destruiria renda e romperia a lei de Walras sem aviso.

O índice de custo de vida da família, usado na renda real e na variação equivalente, é o índice exato de Cobb–Douglas e o índice marginal do LES,

\[
P_h = \exp\Big(\sum_c \beta_{h,c} \log p^{\mathrm{ef}}_{h,c}\Big).
\]

O salário real usa outro índice, de Laspeyres, com a cesta de quantidades do ano-base. Uma cesta que seguisse o cenário faria a regra salarial ser uma regra diferente em cada cenário.

### Poupança e investimento

Com investimento dirigido pela poupança, a poupança financia o investimento, o investimento move a produção, a produção move a receita tributária e a receita é poupança do governo. Quando o governo poupa uma fração fixa \(s_g\) da renda e gasta o resto nas parcelas de valor do ano-base, a receita também é gasto público. O mapa é afim no orçamento do governo e na poupança. Os dois são resolvidos juntos em forma fechada, a partir de sondagens do mapa, em vez de iterados. Com investimento real fixo, as quantidades são as do ano-base, possivelmente reponderadas por uma cesta, e a poupança forçada das famílias é a incógnita que iguala o financiamento ao custo desse investimento.

O saldo do governo que não passa pelo bloco de impostos indiretos é o imposto direto mais a renda de fatores do governo, líquidos das transferências às famílias.

## 6. Incógnitas e fechamento

O mercado do fator numerário não é resolvido. Fica para a lei de Walras, e o seu resíduo é um teste da contabilidade, não uma identidade que o solver foi instruído a cumprir. Ele é registrado em todo período.

As incógnitas de um período são um preço por fator que não seja o numerário, mais uma incógnita externa, mais a poupança forçada quando o investimento real é fixo. A incógnita externa é a taxa de câmbio quando a poupança externa é fixa no valor do ano-base, e a própria poupança externa quando a taxa de câmbio é exógena. A poupança externa fixa pode estar em unidades do numerário ou em dólares, e em dólares o que ela financia em moeda doméstica é o câmbio vezes a meta. Um salário real fixo retira o preço do trabalho do vetor de incógnitas. O salário é o índice vezes a meta, e o emprego é a demanda que esse salário produz. Cada fechamento traz a variável que ajusta. Fixar uma magnitude sem liberar outra em algum lugar não fecha o modelo, e o sintoma é um resíduo de Walras, não uma falha anunciada do solver.

A conta corrente que a poupança externa financia é o comércio, mais a renda de fatores paga ao exterior, menos as transferências líquidas recebidas do exterior. Não é o saldo comercial sozinho. O investimento direto entra na poupança externa do período.

A convergência é o máximo dos desvios de mercado, com tolerância padrão \(10^{-9}\), julgada no resíduo. O mesmo critério da seção 2 do outro pacote vale aqui, pelo mesmo comportamento de `hybr`.

### Trabalho

Com pleno emprego a oferta é a da matriz e o choque aparece no salário. Com salário real fixo a oferta de referência deixa de ser a da matriz e o emprego é a demanda. Com oferta elástica,

\[
L = L_0 \left(\frac{w/P}{w_0/P_0}\right)^{\varepsilon}, \qquad \varepsilon \ge 0.
\]

A curva salarial de Blanchflower e Oswald entra como a equação de equilíbrio do fator trabalho,

\[
\log\frac{w}{P} + \beta \log\frac{u}{u_0} = 0, \qquad \beta > 0.
\]

A matriz registra emprego, não força de trabalho. A força é o emprego de base dividido por \(1-u_0\), e o ano-base está exatamente em \(u_0\). A taxa \(u_0 = 0\) deixa o logaritmo indefinido e é recusada. A demanda de trabalho depende da renda que o emprego paga. O mapa é afim, e duas sondagens o resolvem em forma fechada. Iterá-lo deixaria um resíduo que a jacobiana externa leria como quina.

O fator trabalho não pode ser o numerário quando o fechamento fixa ou liga o seu salário. Esse mercado é o teste de Walras. Pinçá-lo transformaria o teste numa identidade, ou numa fuga.

### Homogeneidade

Nenhuma magnitude real depende do nível do preço do numerário. Magnitudes nominais exógenas denominadas nele, a poupança externa, a meta de saldo comercial e o imposto específico sobre a atividade, escalam com ele. A poupança externa em dólares escala pela taxa de câmbio, que é o que converte dólares em numerário. Deixar alguma delas fixa enquanto o nível de preços se move mudaria a taxa real de câmbio, o investimento real ou a cunha real do imposto, e um teste de homogeneidade falharia por um motivo que não é o modelo.

## 7. Dinâmica recursiva

Cada ano é um equilíbrio da seção 5. O que passa de um ano ao outro é o estoque de capital, a dotação de trabalho e a produtividade. Nenhum agente otimiza ao longo do horizonte. O modelo responde ao que uma economia com essas dotações é em cada ano. Não responde ao que um agente que olha para a frente escolheria.

A linha de capital da matriz é o pagamento anual pelo serviço do capital, não o estoque. Somar o investimento a esse fluxo compõe fluxo com fluxo, e em vinte e cinco anos o estoque foge por ordens de grandeza. O estoque é recuperado do custo de uso,

\[
r_k = \rho + \delta_k, \qquad K_k = \frac{F_k}{r_k},
\]

com \(F_k\) o serviço do ano-base e \(\rho\) a taxa de desconto declarada em `Growth`. No ano seguinte a oferta de serviço é \(K \cdot r\), nas mesmas unidades da matriz. Vários capitais, um por setor, uma vintagem ou um estoque público, acumulam em separado. O investimento doméstico reparte-se entre eles pelas participações do estoque, ou por

\[
\omega_k \propto K_k \left(\frac{r_k}{\bar r}\right)^{e},
\]

quando se declara um expoente \(e\). O investimento vai para o capital que rende mais, como nos modelos recursivos de Dervis, de Melo e Robinson. \(e\) não tem fonte nas contas nacionais.

O investimento real que entra no estoque é, no deflator padrão, a quantidade do pacote de investimento a preços de adquirente do ano-base. No deflator pelo preço próprio, a poupança nominal é dividida pelo preço corrente do pacote. Os dois coincidem no ano-base e se separam quando os preços relativos se movem. A lei do estoque é

\[
K' = (1-\delta) K + I^{\mathrm{dom}} \omega + I^{\mathrm{IED}}.
\]

O investimento direto compra capital que permanece de propriedade estrangeira e vai ao fator a que foi destinado. A participação estrangeira do ano seguinte é o estoque estrangeiro depreciado mais esse fluxo, e a renda correspondente sai pela conta corrente.

O trabalho cresce à taxa declarada. A produtividade de Hicks neutra escala os requisitos de insumo de toda atividade. A produtividade aumentadora de fator escala as dotações. Sob Hicks neutro a mesma dotação produz mais, e os preços dos fatores sobem contra o numerário. Sob a escala da dotação o sistema de preços é homogêneo no vetor de dotações, e os preços relativos dos fatores só se movem quando a composição da oferta muda. Um modelo que cresce de um modo e reporta o outro descreve outra economia.

Uma trajetória de política é uma função do ano que devolve um cenário. O que não é preço, imposto ou dotação, como um coeficiente técnico que muda ao longo do tempo, entra por uma função que devolve a economia daquele ano, com as mesmas contas.

`run_path` não move fatores entre regiões. O ninho regional de um período é resolvido à parte. Acumulação de um estoque de capital por região, e migração, não estão nesta lei de movimento.

## 8. Bem-estar

A renda que entra na variação equivalente é a renda menos as compras diretas a atividades e menos o imposto direto, dividida pelo índice de custo de vida da seção 5. Para um caminho de referência e um alternativo, no mesmo ano e na mesma conta,

\begin{align*}
EV &= Y^{\mathrm{ref}} \left(\frac{R^{\mathrm{alt}}}{R^{\mathrm{ref}}} - 1\right), \\
EV^{\mathrm{base}} &= \big(R^{\mathrm{alt}} - R^{\mathrm{ref}}\big)\, P^{\mathrm{base}}, \\
CV &= Y^{\mathrm{alt}} \left(1 - \frac{R^{\mathrm{ref}}}{R^{\mathrm{alt}}}\right).
\end{align*}

\(EV\) valoriza a variação aos preços do cenário de referência. \(EV^{\mathrm{base}}\) valoriza aos preços do ano-base. A diferença é o drift de preços acumulado no horizonte, um efeito de nível que não encolhe quando o choque de política encolhe. Conciliar uma soma de variações equivalentes com a absorção real, medida a preços do ano-base, usa \(EV^{\mathrm{base}}\). A variação compensatória valoriza aos preços do cenário alternativo. As duas coincidem em primeira ordem. Reportar só uma, chamando-a de variação de bem-estar, esconde a escolha do preço de referência.

Uma conta não é a mesma população em dois cenários quando famílias mudam de conta, como adotantes e não adotantes. A renda real por conta carrega essa diferença de tamanho e a reporta como bem-estar. A agregação por grupo soma as rendas reais das contas do grupo antes de tomar a razão, de modo que a migração dentro do grupo se cancele. Uma conta sem grupo é um erro. Omiti-la em silêncio subestimaria o grupo, e o erro teria a aparência de um resultado.

## 9. O que este sistema não resolve

A trajetória da seção 7 é recursiva. Uma política anunciada para uma data futura não move o investimento do primeiro ano, porque nenhum agente a vê. Levar esse efeito a sério é outro objeto. O equilíbrio do período continua sendo o desta nota. O resíduo intertemporal, uma Euler ou um \(q\) de Tobin ao longo de uma trajetória anunciada, seria resolvido pelo `dsgelab`, que já empilha uma função de resíduos e já aceita uma sequência de parâmetros. A matriz não entraria naquele pacote. A dependência seria opcional. O desenho está no `ROADMAP.md`, na seção *Extensions that call dsgelab*.

Ninhos além do valor adicionado, contas de emissões, unidades físicas de energia e um banco global no estilo GTAP também não estão nas equações acima. São extensões do equilíbrio de período, não do bloco intertemporal.

## Referências

Armington, P. S. (1969). A theory of demand for products distinguished by place of production. *IMF Staff Papers*, 16(1), 159–178.

Blanchflower, D. G. e Oswald, A. J. (1994). *The Wage Curve*. Cambridge, MA: MIT Press.

Dervis, K., de Melo, J. e Robinson, S. (1982). *General Equilibrium Models for Development Policy*. Cambridge: Cambridge University Press.

Hosoe, N., Gasawa, K. e Hashimoto, H. (2010). *Textbook of Computable General Equilibrium Modelling: Programming and Simulations*. Basingstoke: Palgrave Macmillan.

Stone, R. (1954). Linear expenditure systems and demand analysis: an application to the pattern of British demand. *The Economic Journal*, 64(255), 511–527.
