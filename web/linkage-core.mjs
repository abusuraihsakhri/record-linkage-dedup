// Browser-only subset of the Python linkage pipeline. No network requests.
export const FIELDS = ["first_name", "last_name", "dob"];
export const REPORT_COLUMNS = ["row_a", "row_b", "source_a", "source_b", "score", "classification",
  ...FIELDS.flatMap(field => [field + "_a", field + "_b", field + "_similarity", field + "_agree"])];

export function parseCsv(content) {
  if (typeof content !== "string" || !content.trim()) throw new Error("CSV is empty.");
  const text = content.replace(/^\uFEFF/, "");
  const rows = [];
  let row = [], cell = "", quoted = false, closedQuote = false;
  for (let i = 0; i < text.length; i++) {
    const ch = text[i];
    if (quoted) {
      if (ch === '"' && text[i + 1] === '"') { cell += '"'; i++; }
      else if (ch === '"') { quoted = false; closedQuote = true; }
      else { cell += ch; }
    } else if (ch === "," || ch === "\r" || ch === "\n") {
      row.push(cell);
      cell = ""; closedQuote = false;
      if (ch !== ",") {
        if (row.some(v => v !== "")) rows.push(row);
        row = [];
        if (ch === "\r" && text[i + 1] === "\n") i++;
      }
    } else if (ch === '"' && cell === "" && !closedQuote) {
      quoted = true;
    } else if (ch === '"' || closedQuote) {
      throw new Error("Malformed CSV: unexpected character after closing quote.");
    } else { cell += ch; }
  }
  if (quoted) throw new Error("Malformed CSV: unclosed quoted field.");
  if (cell !== "" || row.length) {
    row.push(cell);
    if (row.some(v => v !== "")) rows.push(row);
  }
  if (!rows.length) throw new Error("CSV has no header.");
  const headings = rows.shift().map(v => v.trim().toLowerCase());
  if (headings.some(v => !v) || new Set(headings).size !== headings.length)
    throw new Error("CSV headers must be nonempty and unique (case-insensitive).");
  if (rows.some(r => r.length !== headings.length))
    throw new Error("CSV rows must have the same number of columns as the header.");
  const missing = FIELDS.filter(v => !headings.includes(v));
  if (missing.length) throw new Error("Missing columns: " + missing.join(", "));
  return rows.map(r => Object.fromEntries(headings.map((h, i) => [h, r[i]])));
}

export function normalizeDob(value) {
  const text = String(value ?? "").trim();
  const match = /^(\d{4})-(\d{2})-(\d{2})$/.exec(text);
  if (!match) return "";
  const year = Number(match[1]), month = Number(match[2]), day = Number(match[3]);
  const date = new Date(0);
  date.setUTCFullYear(year, month - 1, day);
  return date.getUTCFullYear() === year && date.getUTCMonth() === month - 1 &&
    date.getUTCDate() === day ? text : "";
}

export function jaro(first, second) {
  if (first === second) return 1;
  if (!first.length || !second.length) return 0;
  const range = Math.max(0, Math.floor(Math.max(first.length, second.length) / 2) - 1);
  const left = Array(first.length).fill(false), right = Array(second.length).fill(false);
  let matches = 0;
  for (let i = 0; i < first.length; i++) {
    for (let j = Math.max(0, i - range); j < Math.min(second.length, i + range + 1); j++) {
      if (!right[j] && first[i] === second[j]) {
        left[i] = right[j] = true; matches++; break;
      }
    }
  }
  if (!matches) return 0;
  let mismatches = 0, j = 0;
  for (let i = 0; i < first.length; i++) {
    if (!left[i]) continue;
    while (!right[j]) j++;
    if (first[i] !== second[j]) mismatches++;
    j++;
  }
  const swaps = Math.floor(mismatches / 2);
  return (matches / first.length + matches / second.length + (matches - swaps) / matches) / 3;
}

export function jaroWinkler(first, second) {
  const score = jaro(first, second);
  let prefix = 0;
  for (let i = 0; i < Math.min(4, first.length, second.length); i++) {
    if (first[i] !== second[i]) break;
    prefix++;
  }
  return score + prefix * 0.1 * (1 - score);
}

export function linkDatasets(datasets, { minSimilarity = 0.85, maxPairs = 100000 } = {}) {
  if (!Array.isArray(datasets) || !datasets.length) throw new Error("Select at least one CSV.");
  if (!Number.isFinite(minSimilarity) || minSimilarity < 0 || minSimilarity > 1)
    throw new Error("Similarity threshold must be between 0 and 1.");
  const records = [];
  datasets.forEach(({ name, rows }) => rows.forEach(row => records.push({
    source: name,
    fields: {
      first_name: String(row.first_name ?? "").trim().toLowerCase(),
      last_name: String(row.last_name ?? "").trim().toLowerCase(),
      dob: normalizeDob(row.dob)
    }
  })));
  const groups = new Map();
  for (let i = 0; i < records.length; i++) {
    const { first_name: first, dob } = records[i].fields;
    // Incomplete blocking keys are excluded rather than comparing all missing values.
    if (!first || !dob) continue;
    const key = dob.slice(0, 4) + "|" + first[0];
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(i);
  }
  const candidates = [];
  for (const indices of groups.values()) {
    const n = indices.length * (indices.length - 1) / 2;
    if (candidates.length + n > maxPairs)
      throw new Error("Too many candidate pairs. Split the dataset or use the Python CLI.");
    for (let i = 0; i < indices.length; i++) {
      for (let j = i + 1; j < indices.length; j++) candidates.push([indices[i], indices[j]]);
    }
  }
  const weight = Math.log2(0.9 / 0.1);
  const upper = 0.75 * FIELDS.length * weight, lower = 0.25 * FIELDS.length * weight;
  const report = candidates.map(([a, b]) => {
    const A = records[a], B = records[b];
    const result = { row_a: a, row_b: b, source_a: A.source, source_b: B.source };
    let agreements = 0;
    for (const field of FIELDS) {
      const av = A.fields[field], bv = B.fields[field];
      const sim = av && bv ? jaroWinkler(av, bv) : 0;
      const agree = !!av && !!bv && sim >= (field === "dob" ? 0.90 : minSimilarity);
      result[field + "_a"] = av;
      result[field + "_b"] = bv;
      result[field + "_similarity"] = Math.round(sim * 10000) / 10000;
      result[field + "_agree"] = agree;
      if (agree) agreements++;
    }
    result.score = (2 * agreements - FIELDS.length) * weight;
    result.classification = result.score >= upper ? "match" :
      result.score >= lower ? "possible_match" : "non_match";
    return result;
  }).sort((a, b) => b.score - a.score);
  return { records: records.length, candidates: candidates.length, report };
}

export function reportCsv(rows) {
  const escapeCell = (value, untrusted) => {
    let text = String(value ?? "");
    // Prevent spreadsheet formula execution from untrusted CSV fields.
    if (untrusted && /^[\s]*[=+@\-\t\r]/.test(text)) text = "'" + text;
    return '"' + text.replace(/"/g, '""') + '"';
  };
  return REPORT_COLUMNS.map(c => escapeCell(c, false)).join(",") + "\r\n" +
    rows.map(row => REPORT_COLUMNS.map(col =>
      escapeCell(row[col], col.endsWith("_a") || col.endsWith("_b"))
    ).join(",")).join("\r\n") + (rows.length ? "\r\n" : "");
}
