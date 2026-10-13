import { parseCsv, linkDatasets, reportCsv } from "./linkage-core.mjs";

const $ = id => document.getElementById(id);
const SAMPLE_A = "patient_id,first_name,last_name,dob\n1,John,Smith,1980-05-14\n2,Robert,Brown,1990-01-23\n3,Linda,Davis,1965-07-30\n";
const SAMPLE_B = "patient_id,first_name,last_name,dob\n101,Jon,Smith,1980-05-14\n102,Robert,Braun,1990-01-23\n103,Susan,Miller,1972-09-19\n";
let example = false, lastRows = null;

function status(message, error = false) {
  $("status").textContent = message;
  $("status").classList.toggle("error", error);
}
function showReport(result) {
  lastRows = result.report;
  $("n-records").textContent = result.records.toLocaleString();
  $("n-pairs").textContent = result.candidates.toLocaleString();
  $("n-matches").textContent = result.report.filter(r => r.classification === "match").length.toLocaleString();
  $("n-possible").textContent = result.report.filter(r => r.classification === "possible_match").length.toLocaleString();
  const target = $("matches");
  target.replaceChildren();
  for (const row of result.report.slice(0, 100)) {
    const tr = document.createElement("tr");
    const cells = [
      row.first_name_a + " " + row.last_name_a + " (" + row.source_a + ", row " + (row.row_a + 1) + ")",
      row.first_name_b + " " + row.last_name_b + " (" + row.source_b + ", row " + (row.row_b + 1) + ")",
      row.dob_a + " / " + row.dob_b,
      row.score.toFixed(3), row.classification.replace("_", " ")
    ];
    cells.forEach((value, idx) => {
      const cell = document.createElement("td");
      cell.textContent = value;
      if (idx === 3) cell.className = "score";
      if (idx === 4) cell.className = row.classification;
      tr.append(cell);
    });
    target.append(tr);
  }
  if (!result.report.length) {
    const row = document.createElement("tr"), cell = document.createElement("td");
    cell.colSpan = 5; cell.textContent = "No records share the configured blocking key.";
    row.append(cell); target.append(row);
  }
  $("preview-note").textContent = result.report.length > 100
    ? "Showing the first 100 highest-scoring pairs; the CSV export includes all candidates."
    : "All candidates are shown. The exported CSV includes comparison scores and field agreement indicators.";
  $("results").hidden = false;
  status("Analysis complete. Review possible matches before merging records.");
}

$("threshold").addEventListener("input", () => {
  $("threshold-value").textContent = Number($("threshold").value).toFixed(2);
});
for (const name of ["file-a", "file-b"]) {
  $(name).addEventListener("change", () => { example = false; lastRows = null; $("results").hidden = true; });
}
$("sample").addEventListener("click", () => {
  example = true;
  $("file-a").value = ""; $("file-b").value = "";
  $("results").hidden = true;
  status("Sample datasets loaded in memory. Choose Find candidate duplicates.");
});
$("analyze").addEventListener("click", async () => {
  $("analyze").disabled = true;
  $("results").hidden = true;
  lastRows = null;
  status("Analyzing candidate pairs…");
  try {
    let datasets;
    if (example) {
      datasets = [{ name: "sample_a.csv", rows: parseCsv(SAMPLE_A) }, { name: "sample_b.csv", rows: parseCsv(SAMPLE_B) }];
    } else {
      const files = [$("file-a").files[0], $("file-b").files[0]].filter(Boolean);
      if (!$("file-a").files.length) throw new Error("Select Dataset A or use the sample records.");
      if (files.some(f => f.size > 5_000_000)) throw new Error("Each file must be 5 MB or smaller. Use the Python CLI for larger data.");
      datasets = await Promise.all(files.map(async f => ({ name: f.name, rows: parseCsv(await f.text()) })));
    }
    const count = datasets.reduce((sum, d) => sum + d.rows.length, 0);
    if (count > 20000) throw new Error("Maximum 20,000 input rows in the browser. Use the Python CLI for larger data.");
    showReport(linkDatasets(datasets, { minSimilarity: Number($("threshold").value) }));
  } catch (err) { status(err.message || "Unable to analyze CSV.", true); }
  finally { $("analyze").disabled = false; }
});
$("download").addEventListener("click", () => {
  if (!lastRows) return;
  const url = URL.createObjectURL(new Blob([reportCsv(lastRows)], { type: "text/csv;charset=utf-8" }));
  const a = document.createElement("a"); a.href = url; a.download = "linkage_report.csv";
  document.body.append(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
});
