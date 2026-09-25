# Site do `cgekit`

Página estática em dois idiomas, publicada em <https://gustavohm.github.io/cgekit/>. Não usa
Jekyll, etapa de build nem CDN: são duas páginas HTML, `style.css` e um realçador de sintaxe de
trinta linhas. Abre num navegador direto do disco.

```
site/
  index.html                 a página em inglês (padrão)
  replications.html          replicações, em inglês
  pt-br/index.html           a página em português
  pt-br/replications.html    replicações, em português
  style.css         tema claro e escuro, seguindo a preferência do sistema
  highlight.js      realce de Python e shell, sem dependências
  .nojekyll         impede o GitHub de processar a pasta com Jekyll
```

## Os dois idiomas

As duas versões de cada página têm o mesmo conteúdo e os mesmos exemplos. A página de
replicações entra no mesmo verificador: o bloco do ano-base de Hosoe roda, e o número de blocos
executáveis tem de coincidir entre o par de idiomas. Cada versão tem um seletor de idioma no
topo e marcações `hreflang` apontando para a outra. Ao mudar uma, mude a outra: o teste
`test_the_language_versions_have_the_same_examples` falha se o número de exemplos executáveis
divergir. A seção de citação muda de propósito: ABNT em português, APA em inglês.

## Publicar

O workflow `.github/workflows/pages.yml` publica esta pasta a cada push no `main` que mexa nela.
É preciso configurar uma vez, nas configurações do repositório: *Settings → Pages → Build and
deployment → Source: GitHub Actions*.

Os links para a documentação apontam para os arquivos `.md` no GitHub, que os renderiza. O GitHub
Pages serviria o Markdown como texto puro.

## Os exemplos são verificados

Todo bloco de código executável da página é extraído e executado por
`tools/check_site_examples.py`, página por página, na ordem em que aparece e num espaço de nomes
compartilhado dentro de cada página. Os exemplos formam uma narrativa em que um bloco usa o que o
anterior definiu.

```bash
python tools/check_site_examples.py
```

O mesmo verificador roda na suíte de testes (`tests/test_site_examples.py`), então um exemplo
quebrado faz o `pytest` falhar. Um exemplo que não roda é pior do que exemplo nenhum: convida o
leitor a copiar algo quebrado.

Blocos marcados `class="plain"` são pulados de propósito: linhas de shell, ilustrações de sintaxe
e trechos que dependem de dados do usuário. A marca é o que os declara não executáveis.

## Adaptar

A folha de estilo é uma só, com todas as cores em variáveis CSS no topo. Trocar `--accent` muda a
identidade da página inteira. O tema escuro segue `prefers-color-scheme`.
