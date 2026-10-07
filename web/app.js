const form = document.getElementById("search-form");
const queryInput = document.getElementById("query");
const modelSelect = document.getElementById("model");
const searchButton = document.getElementById("search-button");
const booleanHint = document.getElementById("boolean-hint");
const status = document.getElementById("status");
const results = document.getElementById("results");

function updateHint() {
  booleanHint.hidden = modelSelect.value !== "boolean";
}

modelSelect.addEventListener("change", () => {
  updateHint();
  if (queryInput.value.trim()) {
    form.requestSubmit();
  }
});

form.addEventListener("submit", (event) => {
  event.preventDefault();
  const query = queryInput.value.trim();
  if (query) {
    runSearch(query, modelSelect.value);
  }
});

async function runSearch(query, model) {
  searchButton.disabled = true;
  status.textContent = "Searching…";
  results.replaceChildren();
  const params = new URLSearchParams({ q: query, model });
  try {
    const response = await fetch(`/api/search?${params}`);
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.error || "Search failed");
    }
    renderResults(data);
  } catch (error) {
    status.textContent = error.message || "Search failed";
  } finally {
    searchButton.disabled = false;
  }
}

function renderResults(data) {
  const count = data.results.length;
  status.replaceChildren();
  if (!count) {
    status.textContent = "No documents matched.";
    return;
  }

  const noun = count === 1 ? "result" : "results";
  const summary = document.createElement("span");
  summary.textContent = `${count} ${noun}. `;
  status.append(summary);
  if (data.expanded && data.expanded !== data.query) {
    const note = document.createElement("span");
    note.append("Expanded query: ");
    const strong = document.createElement("strong");
    strong.textContent = data.expanded;
    note.append(strong);
    status.append(note);
  }

  for (const hit of data.results) {
    const item = document.createElement("li");
    item.className = "result";

    const rank = document.createElement("p");
    rank.className = "rank";
    rank.textContent = String(hit.rank);

    const title = document.createElement("h2");
    const link = document.createElement("a");
    link.href = hit.url;
    link.target = "_blank";
    link.rel = "noreferrer";
    link.append(highlight(hit.title, data.terms));
    title.append(link);

    const meta = document.createElement("p");
    meta.className = "meta";
    const bits = [hit.doc_id, hit.source, hit.topic];
    if (typeof hit.score === "number") {
      bits.push(`score ${hit.score.toFixed(3)}`);
    }
    meta.textContent = bits.filter(Boolean).join(" · ");

    const snippet = document.createElement("p");
    snippet.className = "snippet";
    snippet.append(highlight(hit.snippet, data.terms));

    item.append(rank, title, meta, snippet);
    results.append(item);
  }
}

function highlight(text, terms) {
  const fragment = document.createDocumentFragment();
  const usable = (terms || []).filter((term) => term.length > 2);
  if (!text || !usable.length) {
    fragment.append(document.createTextNode(text || ""));
    return fragment;
  }
  const pattern = new RegExp(`\\b(${usable.map(escapeRegExp).join("|")})\\b`, "gi");
  let last = 0;
  for (const match of text.matchAll(pattern)) {
    const start = match.index ?? 0;
    if (start > last) {
      fragment.append(document.createTextNode(text.slice(last, start)));
    }
    const mark = document.createElement("mark");
    mark.textContent = match[0];
    fragment.append(mark);
    last = start + match[0].length;
  }
  if (last < text.length) {
    fragment.append(document.createTextNode(text.slice(last)));
  }
  return fragment;
}

function escapeRegExp(value) {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

fetch("/api/info")
  .then((response) => response.json())
  .then((info) => {
    const lede = document.getElementById("lede");
    lede.textContent = `${info.documents} pages on places, festivals, food, and practical travel.`;
  })
  .catch(() => {
    status.textContent = "The search service is not responding.";
  });
