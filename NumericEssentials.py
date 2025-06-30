# # Questão 4: Interpolação
# 
# Para a função
# 
# $$
# f(t) = \frac{1}{1 + 25t^2}
# $$
# 
# no intervalo $[-1, 1]$, faça:

# a)  Implemente as interpolações de Lagrange e de Newton.
# 
# b) Usando 11 pontos igualmente espaçados dentro do intervalo dado, calcule as interpolações de Lagrange e Newton com o código implementado no item anterior.  Mostre os resultados em dois gráficos separados. Que resultado teórico justifica o fato das duas soluções serem iguais?
# 
# c) Repita o processo com 21 pontos. O que acontece? Exiba o gráfico das soluções comparando com a exata.
# 
# d) Usando a função `scipy.interpolate.interp1d`, calcule a interpolação usando *spline* linear e cúbica, considerando 21 pontos igualmente espaçados. Exiba os gráficos e comente as diferenças das soluções deste item para os anteriores.
# 
# e) Repita os itens b) e c) com nós de Chebyshev. Comente os resultados obtidos. Por que este resultado é melhor do que os resultados obtidos nos itens b) e c)?

# ## Interpolação por Sistema Linear
# 
# **Antes de começarmos a resolver as questões de fato, precimamos entender os tópicos que ela aborda.**
# 
# Este é o primeiro método de interpolação polinomial que veremos, servindo de base para as formas de Lagrange, Newton e para as análises de erro que virão em seguida. Aqui, buscamos um polinômio
# 
# $$
# p_n(x) = a_0 + a_1x + a_2x^2 + \cdots + a_nx^n
# $$
# 
# que satisfaça
# 
# $$
# p_n(x_i) = f(x_i)\quad\text{para }i=0,\dots,n.
# $$
# 
# ### 1. Formulação como sistema linear
# 
# Para $n+1$ nós $\{(x_i,f(x_i))\}$, impomos
# 
# $$
# \begin{cases}
# a_0 + a_1x_0 + a_2x_0^2 + \cdots + a_nx_0^n = f(x_0),\\
# a_0 + a_1x_1 + a_2x_1^2 + \cdots + a_nx_1^n = f(x_1),\\
# \quad\vdots\\
# a_0 + a_1x_n + a_2x_n^2 + \cdots + a_nx_n^n = f(x_n).
# \end{cases}
# $$
# 
# Em forma matricial, temos o sistema
# 
# $$
# \underbrace{
# \begin{bmatrix}
# 1 & x_0   & x_0^2   & \cdots & x_0^n\\
# 1 & x_1   & x_1^2   & \cdots & x_1^n\\
# \vdots  & \vdots & \vdots  & \ddots & \vdots\\
# 1 & x_n   & x_n^2   & \cdots & x_n^n
# \end{bmatrix}
# }_{\text{matriz de Vandermonde}}
# \!
# \begin{bmatrix}a_0\\a_1\\\vdots\\a_n\end{bmatrix}
# =
# \begin{bmatrix}f(x_0)\\f(x_1)\\\vdots\\f(x_n)\end{bmatrix}.
# $$
# 
# ### 2. Exemplo numérico
# 
# Suponha os pontos
# 
# $$
# \{(-1,4),\,(0,1),\,(2,3)\},
# $$
# 
# logo $n=2$ e
# 
# $$
# p_2(x) = a_0 + a_1x + a_2x^2.
# $$
# 
# O sistema é
# 
# $$
# \begin{bmatrix}
# 1 & -1 & 1\\
# 1 &  0 & 0\\
# 1 &  2 & 4
# \end{bmatrix}
# \begin{bmatrix}a_0\\a_1\\a_2\end{bmatrix}
# =
# \begin{bmatrix}4\\1\\3\end{bmatrix}.
# $$
# 
# Da segunda equação já vem $a_0=1$. Substituindo nas demais e resolvendo:
# 
# 1. $1 - a_1 + a_2 = 4$
# 2. $1 + 2a_1 + 4a_2 = 3$
# 
# Desses, obtemos
# 
# $$
# a_2 = \tfrac{4}{3},\quad
# a_1 = -\tfrac{5}{3}.
# $$
# 
# Portanto,
# 
# $$
# \boxed{p_2(x) = 1 - \tfrac{5}{3}\,x + \tfrac{4}{3}\,x^2.}
# $$

# define os pontos de interpolação
xi = np.array([-1, 0, 2], dtype=float)
fi = np.array([4, 1, 3], dtype=float)

# monta a matriz do sistema (Vandermonde)
A = np.column_stack([xi**0, xi, xi**2])  # colunas: [1, x, x^2]

# resolve para encontrar os coeficientes [a0, a1, a2]
a0, a1, a2 = np.linalg.solve(A, fi)

# define a função do polinômio interpolador p2(x)
def p2(x):
    return a0 + a1*x + a2*x**2

# gera pontos para o gráfico do polinômio
xx = np.linspace(xi.min() - 0.5, xi.max() + 0.5, 400)
yy = p2(xx)

# gráfico 1: dispersão dos pontos originais
plt.figure()
plt.scatter(xi, fi, marker='o', label='pontos originais')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Dispersão dos pontos de interpolação')
plt.legend()
plt.grid(True)

# gráfico 2: polinômio p2(x) sobreposto aos pontos
plt.figure()
plt.scatter(xi, fi, marker='o', label='pontos originais')
plt.plot(xx, yy, label=r'$p_2(x) = a_0 + a_1 x + a_2 x^2$')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Interpolação por Sistema Linear (p2)')
plt.legend()
plt.grid(True)

plt.show()

# ## Polinômios Interpoladores de Lagrange
# 
# Após formularmos a interpolação por sistema linear (matriz de Vandermonde), é comum recorrer à forma de Lagrange quando queremos, sem resolver sistemas potencialmente mal condicionados, um polinômio que passe exatamente pelos nós dados.
# 
# ### 1. Motivação
# 
# * Sistemas de Vandermonde com valores muito próximos podem ser instáveis
# 
#   $$
#     \begin{bmatrix}
#       1 & 0.001 & 0.002\\
#       1 & 0.003 & 0.005\\
#       1 & 0.007 & 0.007
#     \end{bmatrix}
#   $$
# 
#   exigem cuidado numérico.
# * A forma de Lagrange constrói o mesmo polinômio $p_n(x)$ sem montar nem inverter essa matriz.
# 
# 
# ### 2. Definição dos polinômios base $L_k(x)$
# 
# Queremos um polinômio $p_n(x)$ de grau $n$ que satisfaça $p_n(x_i)=y_i$, onde $y_i=f(x_i)$. Para isso, definimos
# 
# $$
# L_k(x) = \prod_{\substack{j=0 \\ j\neq k}}^{n}
# \frac{x - x_j}{x_k - x_j},
# $$
# 
# que garante
# 
# $$
# L_k(x_i)=
# \begin{cases}
# 1,& i=k,\\
# 0,& i\neq k.
# \end{cases}
# $$
# 
# Então
# 
# $$
# \boxed{p_n(x)=\sum_{k=0}^{n}y_k\,L_k(x).}
# $$
# 
# ### 3. Exemplo numérico (com destaque ao polinômio)
# 
# Dados os pontos
# 
# $$
# (x_0,y_0)=(-1,6),\quad
# (x_1,y_1)=(0,1),\quad
# (x_2,y_2)=(2,0).
# $$
# 
# Temos
# 
# $$
# p_2(x)=6\,L_0(x)+1\,L_1(x)+0\,L_2(x).
# $$
# 
# Calculamos cada base:
# 
# $$
# \begin{aligned}
# L_0(x)
# &=\frac{(x-x_1)(x-x_2)}{(x_0-x_1)(x_0-x_2)}
# =\frac{x(x-2)}{(-1)(-3)}
# =\frac{x(x-2)}{3},\\[6pt]
# L_1(x)
# &=\frac{(x-x_0)(x-x_2)}{(x_1-x_0)(x_1-x_2)}
# =\frac{(x+1)(x-2)}{1\cdot(-2)}
# =-\frac{(x+1)(x-2)}{2},\\[6pt]
# L_2(x)
# &=\frac{(x-x_0)(x-x_1)}{(x_2-x_0)(x_2-x_1)}
# =\frac{x(x+1)}{3\cdot2}
# =\frac{x(x+1)}{6}.
# \end{aligned}
# $$
# 
# Substituindo:
# 
# $$
# \begin{aligned}
# p_2(x)
# &=6\cdot\frac{x(x-2)}{3}
# \;+\;1\cdot\Bigl(-\tfrac{(x+1)(x-2)}{2}\Bigr)
# \;+\;0\cdot\frac{x(x+1)}{6}\\[4pt]
# &=2x(x-2)\;-\;\frac{(x+1)(x-2)}{2}\\[4pt]
# &=\frac{3}{2}\,x^2-\frac{7}{2}\,x+1.
# \end{aligned}
# $$
# 
# **Portanto, o polinômio interpolador final é**
# 
# $$
# \boxed{p_2(x) = \frac{3}{2}\,x^2 - \frac{7}{2}\,x + 1.}
# $$
# 
# 
# ### 4. Algoritmo de avaliação direta
# 
# Para avaliar $p_n$ num ponto $\bar x$ sem montar o polinômio completo:
# 
# 1. **Entradas**
#    $\{x_i\}_{i=0}^n$, $\{y_i\}_{i=0}^n$, ponto $\bar x$.
# 2. **Cálculo**
# 
#    $$
#    p_n(\bar x)
#    = \sum_{i=0}^n y_i
#      \prod_{\substack{j=0\\j\neq i}}^n
#        \frac{\bar x - x_j}{x_i - x_j}.
#    $$
# 3. **Pseudocódigo**
# 
#    ```text
#    result ← 0
#    for i from 0 to n:
#      term ← y[i]
#      for j from 0 to n:
#        if j≠i:
#          term ← term * ((x_eval - x[j]) / (x[i] - x[j]))
#      result ← result + term
#    return result
#    ```
# 
# Esse método é direto, evita sistemas lineares e mantém coesão com a sequência de interpolação polinomial.

# ## Interpolação de Newton
# 
# A forma de Newton integra naturalmente o que vimos sobre sistemas lineares e os polinômios de Lagrange, oferecendo um método incremental para construir o interpolador à medida que adicionamos nós.
# 
# ### 1. Forma do polinômio
# 
# Para $n+1$ nós $\{(x_i,f(x_i))\}_{i=0}^n$, o polinômio de grau $n$ em forma de Newton é
# 
# $$
# p_n(x)
# = d_0
# + d_1\,(x - x_0)
# + d_2\,(x - x_0)(x - x_1)
# + \dots
# + d_n\,(x - x_0)\cdots(x - x_{n-1}),
# $$
# 
# em que cada $d_i$ é uma **diferença dividida** e reflete apenas as informações dos nós até $x_i$.
# 
# ### 2. Definição recursiva das diferenças divididas
# 
# 1. **Ordem zero**:
# 
#    $$
#    d_0 = f[x_0] = f(x_0).
#    $$
# 2. **Ordem um**:
# 
#    $$
#    d_1
#    = f[x_0,x_1]
#    = \frac{f(x_1)-f(x_0)}{x_1 - x_0}.
#    $$
# 3. **Ordem $k$** (para $k\ge2$):
# 
#    $$
#    d_k
#    = f[x_0,\dots,x_k]
#    = \frac{f[x_1,\dots,x_k] - f[x_0,\dots,x_{k-1}]}{x_k - x_0}.
#    $$
# 
# Cada $d_k$ corresponde ao primeiro valor na coluna $k$ da tabela de diferenças.
# 
# ### 3. Tabela de diferenças divididas
# 
# Agrupamos os cálculos em colunas de ordem crescente:
# 
# | $x$   | Ordem 0  | Ordem 1      | Ordem 2          | Ordem 3            |
# | ----- | -------- | ------------ | ---------------- | ------------------ |
# | $x_0$ | $f[x_0]$ |              |                  |                    |
# | $x_1$ | $f[x_1]$ | $f[x_0,x_1]$ |                  |                    |
# | $x_2$ | $f[x_2]$ | $f[x_1,x_2]$ | $f[x_0,x_1,x_2]$ |                    |
# | $x_3$ | $f[x_3]$ | $f[x_2,x_3]$ | $f[x_1,x_2,x_3]$ | $f[x_0,\dots,x_3]$ |
# 
# O valor de $d_k$ é sempre o primeiro elemento da coluna de ordem $k$.
# 
# ### 4. Exemplo prático
# 
# Dados os pontos
# 
# $$
# \{(-3,-5),\;(0,2),\;(2,4)\},
# $$
# 
# queremos $p_2(x)=d_0 + d_1(x+3) + d_2(x+3)(x)$.
# 
# 1. **Cálculo de $d_0$, $d_1$ e $d_2$:**
# 
#    $$
#    \begin{aligned}
#      d_0 &= f[-3] = -5,\\
#      d_1 &= f[-3,0] = \frac{2 - (-5)}{0 - (-3)} = \frac{7}{3},\\
#      d_2 &= f[-3,0,2]
#            = \frac{f[0,2] - f[-3,0]}{2 - (-3)}
#            = \frac{\frac{4-2}{2-0} - \frac{7}{3}}{5}
#            = -\frac{4}{15}\quad(\text{o valor aproximado pode variar}).
#    \end{aligned}
#    $$
# 2. **Montagem de $p_2(x)$:**
# 
#    $$
#    p_2(x)
#    = -5
#    + \tfrac{7}{3}\,(x + 3)
#    - \tfrac{4}{15}\,(x + 3)\,x.
#    $$
# 3. **Forma polinomial expandida** (opcional):
# 
#    $$
#    p_2(x)
#    = \frac{17}{30}\,x^2 + \frac{19}{30}\,x + 2.
#    $$
# 
# ### 5. Escolha do grau adequado
# 
# A tabela de diferenças também indica a ordem necessária:
# 
# * Se as diferenças de ordem $j$ se estabilizam (coluna $j+1$ tende a zero), então um polinômio de grau $j$ é suficiente naquela região.
# * Exemplo rápido: com nós em $0,0.1,0.2,0.3,0.4$, as diferenças de ordem 2 ficaram quase constantes e as de ordem 3 muito pequenas, indicando grau 2.
# 
# ### 6. Algoritmo de implementação 
# 
# Definimos $F_{i,j}=f[x_{i-j},\dots,x_i]$.
# 
# 1. **Entradas**
# 
#    * $x = (x_0, x_1, \dots, x_n)$
#    * $y = (f(x_0), f(x_1), \dots, f(x_n))$
#    * ponto de avaliação $\xi$
# 
# 2. **Construção da tabela de diferenças**
# 
#    ```text
#    F[i,0] ← y[i]                       para i = 0,…,n
#    para i de 1 até n:
#      para j de 1 até i:
#        F[i,j] ← (F[i,j−1] − F[i−1,j−1]) / (x[i] − x[i−j])
#    ```
# 
#    Ao final, os coeficientes de Newton são
# 
#    $$
#      D_k = F[k,k],\quad k=0,\dots,n.
#    $$
# 
# 3. **Avaliação direta em $\xi$**
#    
#    Em vez de reconstruir o polinômio completo, calculamos
# 
#    $$
#      p_n(\xi)
#      = D_0
#      + D_1(\xi - x_0)
#      + D_2(\xi - x_0)(\xi - x_1)
#      + \cdots
#      + D_n(\xi - x_0)\cdots(\xi - x_{n-1}).
#    $$
# 
#    **Passo a passo**:
# 
#    * Inicialize
# 
#      $$
#        \text{resultado} \leftarrow D_0,\quad
#        \text{produto} \leftarrow 1.
#      $$
#    * Para cada $k$ de 1 até $n$:
# 
#      1. Atualize o produto acumulado:
# 
#         $$
#           \text{produto} \leftarrow \text{produto}\times(\xi - x_{k-1}).
#         $$
#      2. Some ao resultado:
# 
#         $$
#           \text{resultado} \leftarrow \text{resultado} + D_k \times \text{produto}.
#         $$
#    * No fim, **resultado** contém $p_n(\xi)$.
# 
#    Fazer isso implica em:
# 
#    * **$\mathbf{O}(n^2)$** operações (aproximadamente $\tfrac{n(n+1)}2$ multiplicações),
#    * aproveita todo o trabalho da tabela de diferenças,
#    * e evita manipular somas de potências ou refazer coeficientes.
# 
# Com isso podemos fazer o métodos de Newton mantendo coesão e eficiência.
# 
# ### 7. Devo usar o método de Newton?
# 
# #### Vantagens
# 
# * **Incrementalidade**
#   Você pode adicionar um novo nó $x_{n+1}$ sem refazer todo o cálculo: basta estender a tabela de diferenças divididas e acrescentar mais um termo $d_{n+1}(x - x_0)\cdots(x - x_n)$.
# * **Estabilidade computacional**
#   Quando os nós estão razoavelmente bem espaçados, as diferenças divididas tendem a crescer com moderação, evitando coeficientes gigantes que aparecem em outras formas (por exemplo, a forma monomial).
# * **Reuso de cálculos**
#   Parte da tabela já calculada para grau $k$ serve diretamente para grau $k+1$.
# 
# #### Desvantagens
# 
# * **Ordem dos nós importa**
#   Se você mudar a ordem dos $x_i$, os coeficientes $d_i$ mudam completamente e é preciso reconstruir toda a tabela.
# * **Custo para muitos nós**
#   Para $n$ muito grande, a tabela de diferenças tem $\tfrac{n(n+1)}2$ entradas, e a avaliação direta acumula muitos fatores $(\xi - x_m)$.
# * **Sensível a nós mal escolhidos**
#   Nós muito próximos podem causar cancelamentos numéricos; nós muito espaçados podem gerar oscilações.

# ## Erro exato na interpolação
# 
# Após resolvermos o sistema linear para determinar os coeficientes de um polinômio interpolador, seja na forma de Lagrange ou de Newton, precisamos quantificar o desvio entre o valor real de $f(x)$ e sua aproximação $p_n(x)$. 
# 
# ### 1. Contexto e exemplo
# 
# Como vimos nas formas de Newton e Lagrange, construímos um polinômio de grau 2 para aproximar
# 
# $$
# f(x) = \ln(x) + 3x^2
# $$
# 
# usando os nós $x_0=1$, $x_1=2$ e $x_2=3$. A partir da tabela de diferenças divididas:
# 
# | nó | $d_0$   | $d_1$   | $d_2$  |
# | -- | ------- | ------- | ------ |
# | 1  | 3.0000  |         |        |
# | 2  | 12.6931 | 9.6931  |        |
# | 3  | 28.0986 | 15.4055 | 2.8562 |
# 
# obtemos
# 
# $$
# d_0 = 3,\quad d_1 = 9.6931,\quad d_2 = 2.8562
# $$
# 
# e, por definição na forma de Newton,
# 
# $$
# p_2(x)
# = d_0 + d_1(x - x_0) + d_2(x - x_0)(x - x_1)
# = 3 + 9.6931(x - 1) + 2.8562\,(x - 1)(x - 2).
# $$
# 
# Expandindo,
# 
# $$
# p_2(x) = 2.8562\,x^2 + 1.1245\,x - 0.9807.
# $$
# 
# Logo,
# 
# $$
# p_2(2.5) = 19.6818.
# $$
# 
# ### 2. Definição de erro exato
# 
# Já que $p_2(2.5)$ é apenas uma aproximação, definimos o **erro exato** como
# 
# $$
# E(2.5) = \bigl|f(2.5) - p_2(2.5)\bigr|
# = \bigl|\ln(2.5) + 3\cdot2.5^2 - 19.6818\bigr|
# = \bigl|19.6663 - 19.6818\bigr|
# = 0.0155.
# $$
# 
# 
# ### 3. Limitante superior do erro
# 
# Para garantir um máximo para o desvio em todo o intervalo $[1,3]$, utilizamos o teorema do erro em interpolação: se $f$ é $n+1$ vezes diferenciável e os nós são equidistantes com passo $h$, então
# 
# $$
# E(x)\le \frac{h^{\,n+1}\,M_{n+1}}{4\,(n+1)},
# $$
# 
# onde
# 
# $$
# M_{n+1} = \max_{x\in [x_0,x_n]}\bigl|f^{(n+1)}(x)\bigr|.
# $$
# 
# Aqui, $n=2$, $h=1$ e
# 
# $$
# f^{(3)}(x) = \frac{2}{x^3},
# $$
# 
# que atinge seu valor máximo em $x=1$, logo $M_3 = 2$. Assim,
# 
# $$
# E(x)\le \frac{1^3\cdot 2}{4\cdot 3} = 0.1667,
# $$
# 
# o que é válido para qualquer $x\in[1,3]$.
# 
# Quando os nós não são equidistantes, o limite geral é
# 
# $$
# E(x)\le \bigl|(x - x_0)\,(x - x_1)\dots (x - x_n)\bigr|\;\frac{M_{n+1}}{(n+1)!}.
# $$
# 
# 
# ### 4. Estimativa prática do erro
# 
# Em aplicações reais, $f$ pode não ser conhecida, tornando $\xi$ inalcançável. Para contornar isso, usamos a diferença dividida de ordem $n+1$ como aproximação de $\tfrac{M_{n+1}}{(n+1)!}$:
# 
# $$
# \frac{M_{n+1}}{(n+1)!}\approx |d_{n+1}|.
# $$
# 
# Portanto,
# 
# $$
# E(x)\approx \bigl|(x - x_0)\dots(x - x_n)\bigr|\;|d_{n+1}|.
# $$
# 
# No exemplo, $d_3=0.0283$ e
# 
# $$
# E(2.5)\approx (2.5 - 1)(2.5 - 2)(2.5 - 3)\times 0.0283 = 0.0106,
# $$
# 
# uma boa estimativa frente ao erro exato de $0.0155$.

# ## Interpolação por Splines
# 
# Quando o grau do polinômio cresce, as oscilações indesejadas surgem (fenômeno de Runge) e montar um polinômio único torna-se instável. Splines resolvem isso dividindo o domínio em subintervalos e usando polinômios de grau baixo, que se encaixam suavemente.
# 
# ### 1. Definição geral
# 
# Dado um conjunto de nós $x_0< x_1<\dots< x_n$, uma spline de grau $m$ é uma função
# 
# $$
# S(x)=
# \begin{cases}
# S_0(x),&x\in[x_0,x_1],\\
# S_1(x),&x\in[x_1,x_2],\\
# \;\vdots\\
# S_{n-1}(x),&x\in[x_{n-1},x_n],
# \end{cases}
# $$
# 
# onde cada $S_i(x)$ é um polinômio de grau $m$, tal que:
# 
# 1. **Interpolação:**   $S_i(x_i)=f(x_i)$ e $S_i(x_{i+1})=f(x_{i+1})$.
# 2. **Suavidade:**     $S_i$ e suas derivadas até a ordem $m-1$ coincidem nos nós internos.
# 
# ### 2. Spline linear
# 
# * **Grau 1** em cada $[x_i,x_{i+1}]$:
# 
#   $$
#     s_i(x)
#     =\frac{x_{i+1}-x}{h_i}f(x_i)
#     +\frac{x - x_i}{h_i}f(x_{i+1}),\quad
#     h_i = x_{i+1}-x_i.
#   $$
# * **Características:**
# 
#   * Fácil de implementar e avaliar.
#   * Contínua, mas sem derivada contínua nos nós.
# * **Algoritmo de avaliação em $\bar x$:**
# 
#   1. Encontre $i$ tal que $\bar x\in[x_i,x_{i+1}]$.
#   2. Calcule $s_i(\bar x)$ pela fórmula acima.

# nós e valores da função
xi = np.array([0, 1, 2, 3, 4], dtype=float)
fi = np.array([1, 2, 1.5, 3, 2], dtype=float)

# plot dos nós
plt.figure()
plt.scatter(xi, fi, marker='o', label='nós')
plt.title('Spline Linear – nós de interpolação')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.grid(True)
plt.legend()

# plot dos segmentos de spline linear
plt.figure()
plt.scatter(xi, fi, marker='o', label='nós')
for i in range(len(xi) - 1):
    x_segment = np.linspace(xi[i], xi[i+1], 100)
    h = xi[i+1] - xi[i]
    y_segment = ((xi[i+1] - x_segment) / h) * fi[i] + ((x_segment - xi[i]) / h) * fi[i+1]
    plt.plot(x_segment, y_segment, label=f'segmento [{xi[i]}, {xi[i+1]}]')
plt.title('Spline Linear Interpolante')
plt.xlabel('x')
plt.ylabel('S(x)')
plt.grid(True)
plt.legend()
plt.show()

# ### 3. Spline cúbica natural
# 
# Usa polinômios de grau 3 em cada intervalo, impondo:
# 
# 1. **Interpolação:**
#    $S_i(x_i)=f(x_i)$, $S_i(x_{i+1})=f(x_{i+1})$.
# 2. **Suavidade:**
#    $S_i'(x_{i+1})=S_{i+1}'(x_{i+1})$,
#    $S_i''(x_{i+1})=S_{i+1}''(x_{i+1})$.
# 3. **Condições naturais:**
#    $S''(x_0)=0$ e $S''(x_n)=0$.
# 
# Chamando $h_i=x_{i+1}-x_i$ e $\Delta y_i=f(x_{i+1})-f(x_i)$, montamos o sistema tridiagonal para os coeficientes $b_i=S_i''(x_i)$:
# 
# $$
# \begin{bmatrix}
# 1 &        &        &   &   \\[-3pt]
# h_0 & 2(h_0+h_1) & h_1 &   &   \\[-3pt]
#     & \ddots & \ddots & \ddots &   \\[-3pt]
#     &        & h_{n-1} & 2(h_{n-1}+h_n) & h_n \\[-3pt]
#     &        &        &        & 1
# \end{bmatrix}
# \!
# \begin{bmatrix}b_0\\b_1\\\vdots\\b_{n-1}\\b_n\end{bmatrix}
# =
# 3
# \begin{bmatrix}
# 0\\
# \frac{\Delta y_1}{h_1}-\frac{\Delta y_0}{h_0}\\
# \vdots\\
# \frac{\Delta y_n}{h_n}-\frac{\Delta y_{n-1}}{h_{n-1}}\\
# 0
# \end{bmatrix}.
# $$
# 
# Após resolver $\mathbf A\,\mathbf b=\mathbf g$, os outros coeficientes em cada subintervalo são:
# 
# $$
# \begin{aligned}
# a_i &= \frac{b_i - b_{i-1}}{3\,h_i},\\
# c_i &= \frac{\Delta y_{i-1}}{h_{i-1}}
#         -\frac{h_{i-1}}{3}(2b_{i-1}+b_i),\\
# d_i &= f(x_{i-1}).
# \end{aligned}
# $$
# 
# O polinômio em $[x_{i-1},x_i]$ é
# 
# $$
# S_{i-1}(x)
# = a_i\,(x - x_{i-1})^3
# + b_{i-1}\,(x - x_{i-1})^2
# + c_i\,(x - x_{i-1})
# + d_i.
# $$
# 
# * **Algoritmo resumido:**
# 
#   1. Calcule todos os $h_i$ e $\Delta y_i$.
#   2. Monte $A$ e $\mathbf g$.
#   3. Resolva o sistema para $\mathbf b$.
#   4. Calcule $a_i,c_i,d_i$.
#   5. Para cada $\bar x$, identifique o subintervalo e avalie o polinômio correspondente.

# defininfo os nós e valores da função
xi = np.array([0, 1, 2, 3, 4], dtype=float)
fi = np.array([1, 2, 1.5, 3, 2], dtype=float)

n = len(xi) - 1
h = xi[1:] - xi[:-1]
deltay = fi[1:] - fi[:-1]

# monta a matriz tridiagonal A e o vetor g
A = np.zeros((n+1, n+1))
g = np.zeros(n+1)
A[0, 0] = 1
A[n, n] = 1
for i in range(1, n):
    A[i, i-1] = h[i-1]
    A[i, i] = 2 * (h[i-1] + h[i])
    A[i, i+1] = h[i]
    g[i] = 3 * (deltay[i] / h[i] - deltay[i-1] / h[i-1])

# resolve para obter b = S'' nos nós
b = np.linalg.solve(A, g)

# calcula coeficientes a_i, c_i, d_i em cada subintervalo
a = np.zeros(n+1)
c = np.zeros(n+1)
d = np.zeros(n+1)
for i in range(1, n+1):
    a[i] = (b[i] - b[i-1]) / (3 * h[i-1])
    c[i] = deltay[i-1] / h[i-1] - h[i-1] * (2*b[i-1] + b[i]) / 3
    d[i] = fi[i-1]

# função que avalia a spline natural em x
def S(x):
    idx = np.searchsorted(xi, x)
    if idx == 0:
        idx = 1
    elif idx > n:
        idx = n
    dx = x - xi[idx-1]
    return a[idx]*dx**3 + b[idx-1]*dx**2 + c[idx]*dx + d[idx]

# gera pontos para plotar a curva suave
xx = np.linspace(xi[0], xi[-1], 400)
yy = np.array([S(x) for x in xx])

# gráfico 1: scatter dos nós
plt.figure()
plt.scatter(xi, fi, marker='o', label='nós')
plt.title('Nós de Interpolação (Spline Cúbica Natural)')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.grid(True)
plt.legend()

# gráfico 2: curva suave piecewise + nós
plt.figure()
plt.scatter(xi, fi, marker='o', label='nós')
plt.plot(xx, yy, label='Spline cúbica natural')
plt.title('Spline Cúbica Natural Interpolante')
plt.xlabel('x')
plt.ylabel('S(x)')
plt.grid(True)
plt.legend()

plt.show()

# ### 4. Considerações finais
# 
# * **Spline linear** é simples, mas apenas contínua.
# * **Spline cúbica natural** oferece suavidade até a segunda derivada e evita oscilações de alto grau.
# * Ambas mantêm coesão com os métodos polinomiais anteriores, mas dividem o problema em pedaços, o que melhora estabilidade e flexibilidade.
