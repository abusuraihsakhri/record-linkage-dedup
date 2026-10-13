import test from "node:test";
import assert from "node:assert/strict";
import { parseCsv, jaroWinkler, linkDatasets, reportCsv } from "./linkage-core.mjs";

const a = parseCsv('first_name,last_name,dob\nJohn,Smith,1980-05-14\nRobert,Brown,1990-01-23\n');
const b = parseCsv('first_name,last_name,dob\nJon,Smith,1980-05-14\nRobert,Braun,1990-01-23\n');

test("RFC-style quoting and embedded line breaks", () => {
  const rows = parseCsv('first_name,last_name,dob\r\n"John, Jr.","Sm""ith",1980-05-14\r\n"Ann\nMarie",Lee,1984-03-12\r\n');
  assert.equal(rows[0].first_name, "John, Jr.");
  assert.equal(rows[0].last_name, 'Sm"ith');
  assert.equal(rows[1].first_name, "Ann\nMarie");
});
test("reject malformed CSV", () => {
  assert.throws(() => parseCsv("first_name,last_name,dob\nAnn,Lee"), /columns/);
  assert.throws(() => parseCsv('first_name,last_name,dob\n"Ann,Lee,2000-01-01'), /unclosed/);
  assert.throws(() => parseCsv("first_name,first_name,last_name,dob\nA,A,B,2000-01-01"), /unique/);
});
test("Winkler agreement and blocked candidate results", () => {
  assert.ok(jaroWinkler("martha", "marhta") > .96);
  const result = linkDatasets([{ name: "a", rows: a }, { name: "b", rows: b }]);
  assert.equal(result.records, 4);
  assert.equal(result.candidates, 2);
  assert.equal(result.report[0].classification, "match");
  assert.equal(result.report[1].classification, "possible_match");
});
test("no candidates retains export header", () => {
  const rows = parseCsv("first_name,last_name,dob\nAnn,Lee,1980-01-01\nBob,Lee,1980-01-01\n");
  const result = linkDatasets([{ name: "none", rows }]);
  assert.equal(result.candidates, 0);
  assert.match(reportCsv(result.report), /classification/);
});
test("CSV exports escape spreadsheet formulas", () => {
  const rows = parseCsv('first_name,last_name,dob\n"=HYPERLINK(1)",Smith,1980-05-14\n"=HYPERLINK(1)",Smith,1980-05-14\n');
  const result = linkDatasets([{ name: "source", rows }], { maxPairs: 10 });
  assert.match(reportCsv(result.report), /'=HYPERLINK/);
});
test("limit dense blocking groups", () => {
  const rows = Array.from({ length: 12 }, () => a[0]);
  assert.throws(() => linkDatasets([{ name: "dense", rows }], { maxPairs: 20 }), /Too many/);
});
