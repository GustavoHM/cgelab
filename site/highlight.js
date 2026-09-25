// Realce mínimo para Python e shell. Trinta linhas em vez de uma dependência de CDN: a página
// tem de abrir num repositório clonado, sem rede e sem etapa de build.
(function () {
  var KW = /\b(import|from|as|def|class|return|for|while|in|if|elif|else|with|lambda|None|True|False|and|or|not|print|raise|try|except|yield|assert)\b/g;
  function esc(s) { return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;"); }
  function py(src) {
    var out = "", i = 0;
    // Fatia por comentários e literais primeiro, para não realçar palavras dentro deles.
    var re = /(#[^\n]*)|('(?:[^'\\\n]|\\.)*'|"(?:[^"\\\n]|\\.)*")/g, m;
    while ((m = re.exec(src)) !== null) {
      out += mark(esc(src.slice(i, m.index)));
      out += '<span class="' + (m[1] ? "tok-com" : "tok-str") + '">' + esc(m[0]) + "</span>";
      i = m.index + m[0].length;
    }
    return out + mark(esc(src.slice(i)));
  }
  function mark(s) {
    return s.replace(KW, '<span class="tok-kw">$1</span>')
            .replace(/\b(\d+\.?\d*(?:e-?\d+)?)\b/g, '<span class="tok-num">$1</span>');
  }
  document.querySelectorAll("pre > code").forEach(function (el) {
    if (el.classList.contains("plain")) return;
    el.innerHTML = py(el.textContent);
  });
})();

// Tabelas largas: envolver num contêiner que rola por si, em vez de empurrar a PÁGINA para os
// lados. Uma página cujo corpo rola na horizontal é desconfortável em qualquer largura, e a
// tabela de comparação tem sete colunas.
(function () {
  document.querySelectorAll("table").forEach(function (t) {
    if (t.parentElement.classList.contains("scroll")) return;
    var box = document.createElement("div");
    box.className = "scroll";
    t.parentNode.insertBefore(box, t);
    box.appendChild(t);
  });
})();
