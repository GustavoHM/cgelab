# Site do `cgekit`

Página estática de uma só peça. Sem Jekyll, sem build, sem dependência de CDN: `index.html`,
`style.css` e um realçador de sintaxe de trinta linhas. Abre num navegador a partir do disco e
publica em GitHub Pages sem etapa nenhuma.

```
site/
  index.html      a página
  style.css       tema claro e escuro, seguindo a preferência do sistema
  highlight.js    realce de Python e shell, sem dependências
  .nojekyll       impede o GitHub de processar a pasta com Jekyll
```

## Publicar

Três caminhos, do mais simples ao mais arrumado.

**1. Uma pasta `docs/` no repositório.** Copie `site/` para `docs/` na raiz e, em
*Settings → Pages*, escolha *Deploy from a branch*, branch `main`, pasta `/docs`.

```bash
cp -r packages/cgekit/site docs
git add docs && git commit -m "site: publicar cgekit" && git push
```

**2. Um branch `gh-pages`.**

```bash
git subtree push --prefix packages/cgekit/site origin gh-pages
```

**3. Dois pacotes, um site só.** Publique um `docs/` com `docs/cgekit/` e a pasta do outro lado a
lado, e uma `index.html` na raiz apontando para as duas.

## Antes de publicar: os links da documentação

O site aponta para `../docs/*.md`, que funciona num repositório clonado e **não** funciona numa
página publicada — o GitHub Pages não serve o que está fora da pasta publicada. Duas saídas:

- **Apontar para o GitHub.** Troque `../docs/` pela URL dos arquivos no repositório:

```bash
sed -i '' 's|href="../docs/|href="https://github.com/USUARIO/REPO/blob/main/packages/cgekit/docs/|g' \
  packages/cgekit/site/index.html
```

- **Levar a documentação junto.** Copie `docs/` para dentro da pasta publicada. O GitHub Pages
  renderiza `.md` como texto puro, não como HTML; para renderizado, converta antes (`pandoc`,
  `mkdocs`) ou reescreva os links para o GitHub, que é o caminho mais curto.

## Adaptar

A folha de estilo é uma só, com todas as cores em variáveis CSS no topo. Trocar `--accent` muda a
identidade da página inteira. O tema escuro segue `prefers-color-scheme` e não pede nada do usuário.

## Os exemplos são verificados

Todo bloco de código executável da página é extraído e executado por
`packages/check_site_examples.py`, na ordem em que aparece e num espaço de nomes compartilhado —
porque os exemplos formam uma narrativa em que um bloco usa o que o anterior definiu.

```bash
python packages/check_site_examples.py cgekit
```

Ao editar um exemplo, rode isso. Um exemplo que não roda é pior do que exemplo nenhum: convida o
leitor a copiar algo quebrado e sugere que a documentação não é verificada.

Blocos marcados `class="plain"` são pulados de propósito — linhas de shell, ilustrações de sintaxe
e trechos que dependem de dados do usuário. A marca é o que os declara não executáveis.
