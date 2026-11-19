# LUNA16 official evaluation script — Python 2 → Python 3 port notes

## Source of truth

Original files are the **unmodified** official LUNA16 Grand Challenge evaluation
scripts, extracted verbatim from `evaluationScript.zip` (md5
`02680d438a80dc26eeff0c12e1942642`, 21,803,068 bytes) downloaded directly from
Zenodo record `10.5281/zenodo.3723295` ("LUNA16 Part 1/2", CC-BY-4.0, open
access) via the record's REST API file list. They live untouched one directory
up, at:

- `../NoduleFinding.py`
- `../noduleCADEvaluationLUNA16.py`
- `../tools/csvTools.py`
- `../tools/__init__.py`

They are written in Python 2 (bare `print` statements, `dict.iteritems()`,
binary-mode `csv` file handles) and do not execute under Python 3.

## Porting policy

**Mechanical, syntax-only port. Zero changes to matching logic, FROC
computation, or numerical formulas.** Every change below is forced by either
(a) a Python 2 → 3 language incompatibility, or (b) an API removed/renamed in
the versions of numpy / matplotlib currently installed in the repo venv
(numpy 2.4.6, matplotlib 3.11.2, scikit-learn 1.9.1, Python 3.11.13). No
threshold, distance formula, FROC operating point, bootstrap procedure, or
tie-breaking rule was altered.

Machine-checkable proof: `diffs/*.diff` are whitespace-insensitive
(`diff -u -w -B`) unified diffs between each original file and its ported
counterpart.

- `diffs/NoduleFinding.diff` — **empty (0 lines).** This file needed no
  changes; it is already valid Python 3 (new-style class, no print
  statements, no dict iteration). Copied verbatim into `py3_port/` only so
  the ported package is self-contained for `sys.path` purposes.
- `diffs/csvTools.diff` — **2 changed lines.** `open(filename, "wb")` /
  `open(filename, "rb")` → `open(filename, "w", newline="")` /
  `open(filename, "r", newline="")`. Python 3's `csv` module requires text
  mode, not binary mode; `newline=""` is the standard idiom to let the `csv`
  module itself control line-ending handling (avoids duplicated `\r\n` on
  write). No change to `tryFloat` or `getColumn`.
- `diffs/noduleCADEvaluationLUNA16.py.diff` — **18 changed line-pairs**,
  falling into exactly four categories:
  1. **9× `print` statements → `print()` calls.** Pure Python 2→3 syntax
     (`print 'x'` is a `SyntaxError` in Python 3).
  2. **4× `dict.iteritems()` → `dict.items()`.** `iteritems()` does not exist
     in Python 3. In Python 3.7+, `dict.items()` also preserves insertion
     order, which is at least as deterministic as Python 2's hash-order
     `iteritems()` — this cannot change which detections are matched, only
     (in the never-triggered-in-practice case of `maxNumberOfCADMarks`
     tie-breaking on exactly-equal probabilities) which of several
     equal-probability candidates is nominally "first." This does not affect
     FROC/sensitivity outputs, which are computed from probability values,
     not dict order.
  3. **2× `int(math.floor(...))` casts** in `compute_mean_ci` (bootstrap CI
     lower/upper bound index lookup). `math.floor()` always returns a
     mathematically-integral `float` (e.g. `12.0`), and Python 2 arrays could
     tolerate that as an index. NumPy ≥ 1.12 (and NumPy 2.x strictly) raise
     `IndexError` on a float index. The cast changes no value — floor() of a
     product of a confidence-derived fraction and an integer sample count is
     always exactly representable as an integer — it only changes the
     Python type from `float` to `int` so NumPy's indexer accepts it.
  4. **3× matplotlib API renames**, all backward-compatibility removals, not
     behavior changes:
     - `plt.xscale('log', basex=2)` → `plt.xscale('log', base=2)` (`basex`
       kwarg removed; `base` is its documented modern replacement for a
       non-2D `xscale` call).
     - `plt.grid(b=True, which='both')` → `plt.grid(True, which='both')`
       (`b` kwarg removed; first positional argument is the documented
       modern replacement).
     - `plt.savefig(..., bbox_inches=0, ...)` → `plt.savefig(...,
       bbox_inches=None, ...)`. The original `0` was never the string
       `'tight'`, so it always took the "not tight, use default bbox" branch
       in every historical matplotlib version; `None` is the documented
       spelling of that same "use default" behavior and is the only value
       matplotlib 3.x accepts that preserves it (a bare `0` now either
       errors or is undefined behavior depending on version).

No other line in `noduleCADEvaluationLUNA16.py` was touched: the Euclidean
world-mm distance-matching rule (`dist < radiusSquared`, radius = observed
nodule diameter / 2, or a fixed 10 mm fallback for excluded findings with
diameter = -1 sentinel), the `bOtherNodulesAsIrrelevant` exclusion handling,
the double-detection accounting, the `sklearn.metrics.roc_curve`-based FROC
construction (`fps = fpr * (candidates - detected) / totalImages`, `sens =
tpr * detected / totalLesions`), the FROC operating points
(0.125/0.25/0.5/1/2/4/8 FP/scan), and the 1000-sample bootstrap CI procedure
are byte-identical to the official script.

## Verification performed (this port)

1. **Whitespace-insensitive diff audit** (above) — confirms only the listed
   syntax/API substitutions differ from the official source.
2. **Import smoke test** — `import noduleCADEvaluationLUNA16` succeeds under
   `/workspace/smm/.venv` (Python 3.11.13, numpy 2.4.6, matplotlib 3.11.2,
   sklearn 1.9.1) with zero errors.
3. **Real-data parsing smoke test** — `collect()` run against the actual
   downloaded `annotations.csv` / `annotations_excluded.csv` /
   `seriesuids.csv` (not synthetic) correctly parsed 1186 included nodule
   annotations, 35192 excluded findings, and 888 seriesUIDs — exactly
   matching the known official row counts (1187-1 and 35193-1 header-adjusted
   lines, 888-line seriesuids.csv).
4. **End-to-end `evaluateCAD()` code-path smoke test** — run against one real
   seriesuid's real nodule annotations (`1.3.6.1.4.1.14519.5.2.1.6279.6001.
   100225287222365663678666836860`, which has 2 real "Included" nodules and
   73 real "Excluded" findings) with a tiny hand-built, clearly-labeled
   throwaway candidate file (`FAKE_results_do_not_use.csv`, 2 rows: one
   candidate placed at nodule #1's real coordinates at high probability, one
   candidate placed far away at low probability). Result: TP=1, FN=1
   (nodule #2, correctly never detected by either fake candidate), FP=1
   (the far-away candidate), Sensitivity=0.5 — exactly the arithmetically
   correct outcome for that deliberately-constructed input. Both the
   bootstrap (5 resamples) and non-bootstrap code paths executed without
   error and produced `CADAnalysis.txt`, `froc_*.txt`, `froc_*.png`,
   `froc_*_bootstrapping.csv`, and `froc_gt_prob_vectors_*.csv` outputs in
   the expected format. **This smoke-test output was synthetic-input-driven
   code verification only, was never written under `vendor/`, `tests/`, or
   any results directory, and was deleted immediately after verification.
   It is not a nodule-detection performance measurement and must never be
   cited as one.**

One cosmetic, pre-existing (present in the original Python 2 script too, not
introduced by this port) matplotlib `UserWarning: FixedFormatter should only
be used together with FixedLocator` fires during plot generation. It does not
affect the correctness of the written FROC curve data files, only a
non-fatal style warning from the axis-tick-label call; left as-is to keep the
port minimal.

## Directory layout

```
vendor/luna16_evaluation/
├── NoduleFinding.py                  (original, Python 2 — untouched)
├── noduleCADEvaluationLUNA16.py      (original, Python 2 — untouched)
├── tools/{__init__.py,csvTools.py}   (original, Python 2 — untouched)
├── annotations/{annotations.csv, annotations_excluded.csv, seriesuids.csv}
├── exampleFiles/
└── py3_port/                         (this port — what the audit actually runs)
    ├── PORT_NOTES.md                 (this file)
    ├── NoduleFinding.py              (0 changes)
    ├── noduleCADEvaluationLUNA16.py  (18 line-pairs changed — see diffs/)
    ├── tools/{__init__.py,csvTools.py}  (csvTools.py: 2 lines changed)
    └── diffs/                        (diff -u -w -B original -> py3_port, per file)
```

## Provenance of the real annotation/eval-script data

- Zenodo record: `10.5281/zenodo.3723295` ("LUNA16 Part 1/2"), license
  `cc-by-4.0`, `access_right: open`.
- Original paper: Setio et al. 2017, *Medical Image Analysis*,
  doi:10.1016/j.media.2017.06.015.
- `annotations.csv` md5 `f4404df491aee6445a4485061bfdffdd` (136,986 bytes).
- `evaluationScript.zip` md5 `02680d438a80dc26eeff0c12e1942642`
  (21,803,068 bytes).
- Both downloaded directly from the Zenodo REST API file list for this
  record (not scraped from the JS-rendered HTML page), full command history
  and checksums recorded in the session's execution trace.
