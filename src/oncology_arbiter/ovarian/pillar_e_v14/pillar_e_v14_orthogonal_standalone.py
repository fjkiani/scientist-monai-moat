"""Research-only v14 (orthogonal Door-1 decomposition + nested L1 heads).
Keep the required JSON artifact beside this module."""
import hashlib
import json
from math import exp, isfinite, isclose
from pathlib import Path
CLINICAL_USE = False
MODEL_SHA256 = "43409b7c685f4c2b35289d4361f01b60ba9b38af68780cfbdde56ca11da0dbea"
GENES = ("VDR","RXRA","EPHA2","SRC","FERMT2","PTK2","YES1","WWTR1","ABCB1","SLFN11","PARP1","RICTOR",
         "CCNE1","RAD51C","MYC","CHEK1","PCNA","RAD51","FANCI","SLC31A1","TUBB3","MFAP4","YAP1")
ALT_NAMES = ("BRCA12_alt","MBD4_alt")
DDR4_GENES = ("PCNA","CHEK1","RAD51","FANCI")
ANTIGENS = ("FOLR1","CDH6","TACSTD2","ERBB2","ERBB3","CD276","VTCN1","SLC34A2","MSLN","MUC16",
            "PTK7","EPHA2","CEACAM5","LRRC15","ENG","FAP")
PAYLOAD_GENES = ("TOP1","SLFN11","ABCB1","TUBB3")
MECHANO_GENES = ("FERMT2","PTK2","YAP1","WWTR1")
ALIASES = {"ceralasertib":"azd6738","adavosertib":"mk-1775","abt737":"abt-737"}
DEFAULT_HEAD = "v14_sparse_all28_logistic_q75"
GATES = ("BH3","Src_FAK","VDR","mTORC2","Mitoribo","Door1A_PARP_Specific",
         "Door1B_ATR_WEE1_RS_Specific","Door1C_Platinum_Taxane_Specific")
DOOR1_CLASSES = ("Door1A_PARP_Specific","Door1B_ATR_WEE1_RS_Specific",
                 "Door1C_Platinum_Taxane_Specific")
HEAD_TYPES = ("logistic_positive","ridge_positive","logistic_q75",
              "logistic_positive_l1","logistic_q75_l1")

def _number(x):
    if isinstance(x, bool) or not isinstance(x, (int,float)) or not isfinite(x):
        raise ValueError("Finite numeric values required")
    return float(x)

def _drugs(values, m):
    if isinstance(values,str):
        raise ValueError("Pass a list or a recognized menu name")
    names = list(values)
    if not names or any(not isinstance(d,str) for d in names):
        raise ValueError("Nonempty drug list required")
    names = [ALIASES.get(d.strip().lower(),d.strip().lower()) for d in names]
    if any(d not in m["drugs"] for d in names) or len(set(names)) != len(names):
        raise ValueError("Unknown or duplicate drugs")
    return names

def validate_model(m):
    if m.get("schema") != "pillar-e-v14-1" or m.get("clinical_use") is not False:
        raise ValueError("Invalid research artifact")
    if len(m["drugs"]) != 28 or len(set(m["drugs"])) != 28 or set(m["drug_features"]) != set(m["drugs"]):
        raise ValueError("Invalid drug dimensions")
    for ds in list(m["menus"].values())+list(m["training_menus"].values()):
        _drugs(ds,m)
    expected = {f"v14_{r}_{t}_{h}" for r in ("sparse","unified") for t in ("all28","core4","core8")
                for h in HEAD_TYPES}
    if set(m["models"]) != expected or set(m["transforms"]) != {"all28","core4","core8"}:
        raise ValueError("Invalid model set")
    if set(m["class_of"]) != set(m["drugs"]) or set(m["drug_targets"]) != set(m["drugs"]):
        raise ValueError("Invalid gate or target maps")
    for d in m["drugs"]:
        if m["class_of"][d] not in GATES+("Other",) or not isinstance(m["drug_targets"][d], list):
            raise ValueError("Invalid gate or target entry")
    if not set(m["efflux_drugs"]) <= set(m["drugs"]):
        raise ValueError("Invalid efflux list")
    for t in m["transforms"].values():
        for k,n in [("expression_center",23),("expression_scale",23),("driver_center",8),("driver_scale",8),
                    ("alt_center",2),("alt_scale",2),("pure_impute",2),("pure_center",2),("pure_scale",2)]:
            if len(t[k]) != n:
                raise ValueError("Invalid transform dimensions")
            for x in t[k]:
                _number(x)
            if k.endswith("scale") and any(x <= 0 for x in t[k]):
                raise ValueError("Invalid scale")
        for k,n in [("driver_constant",8),("alt_constant",2),("ddr4_constant",1),
                    ("backbone_constant",1)]:
            if len(t[k]) != n or any(type(x) is not bool for x in t[k]):
                raise ValueError("Invalid constant flags")
        for k in ("ddr4_center","ddr4_scale","backbone_center","backbone_scale"):
            if len(t[k]) != 1:
                raise ValueError("Invalid DDR proxy or backbone transform")
            _number(t[k][0])
        if t["ddr4_scale"][0] <= 0 or t["backbone_scale"][0] <= 0:
            raise ValueError("Invalid DDR proxy or backbone scale")
        for k in ("S_center","S_scale","E3_impute_training_menu_median"):
            _number(t[k])
        for k in ("slope","intercept","center","scale"):
            _number(t["proxy"][k])
        if t["S_scale"] <= 0 or t["proxy"]["scale"] <= 0 or not 0 <= t["E3_impute_training_menu_median"] <= 1:
            raise ValueError("Invalid proxy or E3")
    for key,h in m["models"].items():
        names = m["feature_names"][h["representation"]]
        n = len(names)
        if len(h["beta"]) != n+1 or len(h["platt"]) != 2 or h["menu"] not in m["training_menus"]:
            raise ValueError("Invalid coefficient dimensions")
        for x in h["beta"]+h["platt"]:
            _number(x)
        if (h["beta"][names.index("E3_in_range")+1] < 0 or
                h["beta"][names.index("negative_ABCB1_substrate")+1] < 0 or h["platt"][0] < 0):
            raise ValueError("Invalid direction constraints")
    for f in m["drug_features"].values():
        if len(f) != 13 or any(x not in (0,1) for x in f[:8]+f[12:]) or sum(f[:8]) > 1:
            raise ValueError("Invalid gate or substrate masks")
        for x in f:
            if x is not None:
                _number(x)
        if f[10] is not None and not 0 <= f[10] <= 1:
            raise ValueError("Invalid E3 fraction")
    sp = m["surface_payload"]
    if (len(sp["antigens"]) != 16 or len(set(sp["antigens"])) != 16 or
            len(sp["payload_genes"]) != 4 or len(set(sp["payload_genes"])) != 4):
        raise ValueError("Invalid surface panel")
    for k,n in [("antigen_center",16),("antigen_scale",16),("payload_center",4),("payload_scale",4)]:
        if len(sp[k]) != n:
            raise ValueError("Invalid surface transform dimensions")
        for x in sp[k]:
            _number(x)
        if k.endswith("scale") and any(x <= 0 for x in sp[k]):
            raise ValueError("Invalid surface scale")
    obs = m["observed"]
    if len(obs) != 45 or len({r["PatientID"] for r in obs}) != 45 or sum(r["Door2"] for r in obs) != 11:
        raise ValueError("Invalid observed population")
    for r in obs:
        if set(r["responses"]) != set(m["drugs"]) or set(r["references"]) != set(m["drugs"]):
            raise ValueError("Invalid observed menu")
        for d in m["drugs"]:
            if r["responses"][d] is not None:
                _number(r["responses"][d])
            a = r["references"][d]
            if not a or a != sorted(a):
                raise ValueError("Invalid reference")
            for x in a:
                _number(x)
    return m

def load_model(path=None):
    p = Path(path) if path is not None else Path(__file__).with_name("pillar_e_v14_model.json")
    raw = p.read_bytes()
    if hashlib.sha256(raw).hexdigest() != MODEL_SHA256:
        raise ValueError("Artifact SHA-256 mismatch")
    return validate_model(json.loads(raw))

def _sigmoid(x):
    return 1/(1+exp(-x)) if x >= 0 else exp(x)/(1+exp(x))

def _surface_report(surface, m):
    sp = m["surface_payload"]
    if not isinstance(surface, dict) or not surface:
        raise ValueError("Surface must be a nonempty dict of antigen or payload gene expression values")
    unknown = set(surface) - set(sp["antigens"]) - set(sp["payload_genes"])
    if unknown:
        raise ValueError("Unknown surface genes: %s" % sorted(unknown))
    values = {k:_number(v) for k,v in surface.items()}
    zA = {a:(values[a]-sp["antigen_center"][i])/sp["antigen_scale"][i]
           for i,a in enumerate(sp["antigens"]) if a in values}
    report = {"elevated_antigens_z_gt0":sorted(a for a,z in zA.items() if z > 0),
              "strong_antigens_z_gt1":sorted(a for a,z in zA.items() if z > 1),
              "not_supplied_antigens":sorted(a for a in sp["antigens"] if a not in values),
              "TOP1_Payload_Index":None,"DM4_Taxane_Index":None,"payload_preference":None,
              "label":"HYPOTHESIS_ONLY_not_validated_clinical_routing"}
    if set(sp["payload_genes"]) <= set(values):
        zP = {g:(values[g]-sp["payload_center"][i])/sp["payload_scale"][i]
              for i,g in enumerate(sp["payload_genes"])}
        top1 = zP["TOP1"]+zP["SLFN11"]-zP["ABCB1"]
        dm4 = -zP["TUBB3"]-zP["ABCB1"]
        report["TOP1_Payload_Index"] = top1
        report["DM4_Taxane_Index"] = dm4
        report["payload_preference"] = ("TOP1_payload" if top1 > dm4 else
                                        "DM4_taxane_payload" if dm4 > top1 else "indeterminate")
    else:
        report["payload_note"] = "payload_indices_require_TOP1_SLFN11_ABCB1_TUBB3"
    return report

def predict(molecular, drugs=None, head=DEFAULT_HEAD, model=None, explain=False, surface=None):
    """Frozen H/S, optional consistent buffer, supplied-scale expression, optional binary
    likely-LOF alteration flags (default wild-type); no test Chronos. Door-1 subgates are
    orthogonal mechanism residuals; the shared -H+zSLFN11 term enters once as
    Door1_Backbone. ADC surface and payload recommendations are hypothesis generation
    only, not validated clinical payload prediction."""
    m = load_model() if model is None else validate_model(model)
    if head not in m["models"]:
        raise ValueError("Unknown head")
    required = set(("H_clean","S_sen")+GENES)
    allowed = required | {"buffer"} | set(ALT_NAMES)
    if not set(molecular) <= allowed or not required <= set(molecular):
        raise ValueError("Supply H_clean, S_sen and all 23 expression genes "
                         "(optional: buffer, BRCA12_alt, MBD4_alt)")
    u = {k:_number(x) for k,x in molecular.items()}
    wt_default = False
    for name in ALT_NAMES:
        if name in u:
            if u[name] not in (0.,1.):
                raise ValueError("Alteration flags must be 0 or 1")
        else:
            u[name] = 0.
            wt_default = True
    H,S = u["H_clean"],u["S_sen"]
    B = S-H
    if "buffer" in u and not isclose(B,u["buffer"],rel_tol=1e-10,abs_tol=1e-10):
        raise ValueError("Inconsistent buffer")
    h = m["models"][head]; t = m["transforms"][h["menu"]]
    names = m["training_menus"][h["menu"]] if drugs is None else (
        m["menus"][drugs] if isinstance(drugs,str) and drugs in m["menus"] else drugs)
    names = _drugs(names,m)
    if not set(names) <= set(m["training_menus"][h["menu"]]):
        raise ValueError("Requested drugs outside this head's training menu")
    z = {g:(u[g]-t["expression_center"][i])/t["expression_scale"][i] for i,g in enumerate(GENES)}
    p = t["proxy"]; proxy = (p["intercept"]+p["slope"]*z["RICTOR"]-p["center"])/p["scale"]
    ddr4 = sum(u[g] for g in DDR4_GENES)/len(DDR4_GENES)
    zddr4 = 0. if t["ddr4_constant"][0] else (ddr4-t["ddr4_center"][0])/t["ddr4_scale"][0]
    za = {name:(0. if t["alt_constant"][i] else (u[name]-t["alt_center"][i])/t["alt_scale"][i])
          for i,name in enumerate(ALT_NAMES)}
    raw = [S+B,
           H-B+z["EPHA2"]+z["SRC"]+z["FERMT2"]+z["PTK2"]+.5*(z["YES1"]+z["WWTR1"]),
           H-.5*B+z["VDR"]+z["RXRA"],S-proxy,S,
           z["PARP1"]+za["BRCA12_alt"]-z["RAD51C"]-z["CCNE1"],
           zddr4+z["CCNE1"]+z["MYC"]+z["CHEK1"]+za["MBD4_alt"],
           z["SLC31A1"]-z["TUBB3"]-z["MFAP4"]]
    drivers = [0. if t["driver_constant"][i] else (a-t["driver_center"][i])/t["driver_scale"][i]
               for i,a in enumerate(raw)]
    backbone = 0. if t["backbone_constant"][0] else (
        -H+z["SLFN11"]-t["backbone_center"][0])/t["backbone_scale"][0]
    sz = 0. if t["S_constant"] else (S-t["S_center"])/t["S_scale"]
    mechano = sum(z[g] for g in MECHANO_GENES)/len(MECHANO_GENES)
    surface_report = _surface_report(surface,m) if surface is not None else None
    feature_names = m["feature_names"][h["representation"]]
    output = []
    for d in names:
        f = m["drug_features"][d]; e1,coupling,e3,state = f[8:12]
        gated = [a*b for a,b in zip(drivers,f[:8])]
        door1f = 1. if m["class_of"][d] in DOOR1_CLASSES else 0.
        unified = (sum(gated)+backbone*door1f) if sum(f[:8]) else (
            sz*(0 if e1 is None or e1 == 0 else (1 if e1 > 0 else -1)))
        pure = [None if e1 is None or coupling is None else e1*coupling,state]
        pure = [((t["pure_impute"][i] if a is None else a)-t["pure_center"][i])/t["pure_scale"][i]
                for i,a in enumerate(pure)]
        shared = [sz*(0 if e1 is None else e1),pure[0],
                  t["E3_impute_training_menu_median"] if e3 is None else e3,pure[1],
                  -z["ABCB1"]*f[12],backbone*door1f]
        x = (gated if h["representation"] == "sparse" else [unified])+shared
        score = h["beta"][0]+sum(a*b for a,b in zip(x,h["beta"][1:]))
        if not all(isfinite(a) for a in x+[score]):
            raise ValueError("Nonfinite transformed input")
        row = {"drug":d,"score":score,"probability":_sigmoid(h["platt"][0]*score+h["platt"][1]),
               "probability_target":h["probability_target"]}
        if explain:
            gate = m["class_of"][d]
            gi = GATES.index(gate) if gate in GATES else None
            warnings = []
            if f[12] and z["ABCB1"] > 1:
                warnings.append("abcb1_efflux")
            if d == "venetoclax" and drivers[0] > 0:
                warnings.append("venetoclax_bcl2_decoy")
            if d == "vismodegib" and mechano > 1:
                warnings.append("vismodegib_mycaf_collapse_hazard")
            if wt_default:
                warnings.append("alteration_status_not_supplied_wt_default")
            row.update({
                "active_mechanistic_gate":gate,
                "target_genes":list(m["drug_targets"][d]),
                "driver_breakdown":{"active_gate":gate,
                    "active_driver_standardized":drivers[gi] if gi is not None else None,
                    "gated_value":gated[gi] if gi is not None else None,
                    "door1_backbone":backbone if door1f else None,
                    "unified_context":unified},
                "feature_contributions":[{"feature":"intercept","x":1.,"beta":h["beta"][0],
                                          "contribution":h["beta"][0]}]+
                    [{"feature":feature_names[i],"x":x[i],"beta":h["beta"][i+1],
                      "contribution":h["beta"][i+1]*x[i]} for i in range(len(x))],
                "adc_surface_payload_recommendations":surface_report,
                "warnings_and_flags":warnings})
        output.append(row)
    return sorted(output,key=lambda r:(-r["score"],m["drugs"].index(r["drug"])))

def _quantile(a,q):
    t = (len(a)-1)*q; i = int(t)
    return a[i]+(t-i)*(a[min(i+1,len(a)-1)]-a[i])

def observed_coverage(drugs, threshold=.5, endpoint="training_drug_reference", population="door2", model=None):
    """Descriptive observed OR coverage; not efficacy or combination prediction."""
    q = _number(threshold)
    if not 0 <= q <= 1 or endpoint not in ("training_drug_reference","within_donor"):
        raise ValueError("Invalid threshold or endpoint")
    if population not in ("all45","door2"):
        raise ValueError("Unknown population")
    m = load_model() if model is None else validate_model(model)
    names = _drugs(drugs,m); counts = dict(active=0,inactive=0,unknown=0)
    for r in m["observed"]:
        if population == "door2" and not r["Door2"]:
            continue
        measured = sorted(r["responses"][d] for d in names if r["responses"][d] is not None)
        states = [None if r["responses"][d] is None else r["responses"][d] > _quantile(
                  r["references"][d] if endpoint == "training_drug_reference" else measured,q) for d in names]
        key = "active" if any(s is True for s in states) else (
            "unknown" if any(s is None for s in states) else "inactive")
        counts[key] += 1
    return counts
