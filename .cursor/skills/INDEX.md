# Oncology Moat skills — TONIGHT pack

**Canon reports (Alpha):** `.cursor/rules/arbriter/`
**Branch truth:** `src/oncology_arbiter/arbiter/models/*_template_v0.json` → `n_training=0`; `biopsy_probe_v0` synthetic.

Labels: `verified` = path spot-checked this install.

| Skill | Tonight force | Label |
|-------|---------------|-------|
| `oncology-moat-platform` | Router + ban report-as-HEAD | verified |
| `oncology-moat-anti-fortress` | Push complete weight/code deliveries | verified |
| `oncology-moat-stage-arbiters` | Train and wire all 3 stage arbiters | verified |
| `oncology-moat-weight-floors` | Train every named checkpoint | verified |
| `oncology-moat-biopsy` | Real biopsy_probe_v1, 1152-d | verified |
| `oncology-moat-cbis-medsiglip` | New v2 train + product wiring | verified |
| `oncology-moat-clinicalbert` | New real train + matching deployment | verified |
| `oncology-moat-luna16` | Resample + refinement train + wiring | verified |
| `luna16-gates-a-e` | Serial Gates A–E receipts (audit→convert→fit→FROC) | verified |
| `moat-agent-discipline` | Multi-agent Zo/exec cage; bans status.md theater | verified |
| `oncology-moat-phikon` | Complete embeddings + joblib + wiring | verified |
| `oncology-moat-medgemma` | Correct weights + working therapy chat | verified |
| `oncology-72h-clinical-hardening` | 72h titanium: MedGemma NLP, Phikon tissue, stratum 422, RSNA/LUNA, ghost trial | verified |

**Sob attach (72h clinical hardening):**
```
/oncology-72h-clinical-hardening /oncology-moat-platform /oncology-moat-anti-fortress /moat-agent-discipline /anti-sandbagging /zeta-enforcer
```
First strike: MedGemma PathologyExtraction primary (ClinicalBERT NER legacy-only). Tissue→Phikon. Calc-only→422.

**Sob attach (LUNA lane):**
```
/luna16-gates-a-e /moat-agent-discipline /oncology-moat-luna16 /anti-sandbagging /zeta-enforcer
```
Gate A first. No pilot→auto fullfit before `GATE_C_PASS`. No `execution_status.md` as DONE.

**Sob attach (platform):**
```
/oncology-moat-platform /oncology-moat-anti-fortress /oncology-moat-stage-arbiters /anti-sandbagging /zeta-enforcer
```
Then add the active lane skill. Deadline: tonight. Measurement tip a81e9a9 ≠
training credit. No capability may exit through a status label, deletion,
warning-only path, or partial download.

Before claiming DONE:
```
python .cursor/skills/oncology-moat-platform/validate_tonight_delivery.py --repo .
```
