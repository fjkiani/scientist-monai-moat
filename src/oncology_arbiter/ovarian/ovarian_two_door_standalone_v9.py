"""Offline exploratory results; six-decimal displays, not a clinical rule."""
from math import isfinite
CLINICAL_USE = False
CRISPR_STANDARDIZERS = {'SRC': (-0.04431320654006006, 0.13523309695684188), 'YES1': (0.017572078872620245, 0.1519428992424816), 'BCL2L1': (-1.2655787932680864, 0.6186291608658663), 'RICTOR': (-0.41514941774116426, 0.2643878955551146), 'STAT3': (-0.06873868319136048, 0.12132418186629806), 'CPEB4': (0.010088965647040657, 0.14474796007935972), 'WASHC4': (-0.14727245286307036, 0.14703915164241624)}
FAMILIES = {0: 'panel13_Fisher_S_sen', 1: 'panel13_Consensus_Z_H_clean', 2: 'panel13_Consensus_Z_S_sen', 3: 'original13|PRISM_secondary|S_sen', 4: 'original13|PRISM_primary|S_sen', 5: 'original13|Fisher|S_sen', 6: 'original13|DoseResponse_Consensus_Z|S_sen'}
LEDGER_FIELDS = ['version', 'drug', 'endpoint', 'score', 'family', 'n', 'rho', 'p', 'q']
LEDGER_ROWS = [
    ('v8', 'abt-737', 'Fisher', 'S_sen', 0, 45, 0.328568, 0.00208, 0.02704),
    ('v8', 'abt-737', 'Consensus_Z', 'H_clean', 1, 45, 0.352174, 0.00252, 0.03276),
    ('v8', 'abt-737', 'Consensus_Z', 'S_sen', 2, 45, 0.350198, 0.00934, 0.048186),
    ('v8', 'navitoclax', 'Fisher', 'S_sen', 0, 45, 0.307477, 0.01007, 0.049833),
    ('v8', 'navitoclax', 'Consensus_Z', 'S_sen', 2, 45, 0.368643, 0.003, 0.039),
    ('v8', 'dasatinib', 'Consensus_Z', 'S_sen', 2, 45, 0.411067, 0.01112, 0.048186),
    ('v8', 'doxercalciferol', 'Fisher', 'S_sen', 0, 22, 0.46358, 0.01534, 0.049855),
    ('v8', 'azithromycin', 'Fisher', 'S_sen', 0, 26, 0.446838, 0.0115, 0.049833),
    ('v9', 'abt-737', 'PRISM_secondary', 'S_sen', 3, 27, 0.431624, 0.012286, 0.035734),
    ('v9', 'navitoclax', 'PRISM_secondary', 'S_sen', 3, 23, 0.473995, 0.011159, 0.035734),
    ('v9', 'doxercalciferol', 'PRISM_secondary', 'S_sen', 3, 22, 0.46358, 0.014889, 0.035734),
    ('v9', 'azithromycin', 'PRISM_secondary', 'S_sen', 3, 26, 0.446838, 0.011054, 0.035734),
    ('v9', 'vismodegib', 'PRISM_secondary', 'S_sen', 3, 26, 0.432682, 0.013631, 0.035734),
    ('v9', 'navitoclax', 'PRISM_primary', 'S_sen', 4, 34, 0.472574, 0.002385, 0.028617),
    ('v9', 'abt-737', 'Fisher', 'S_sen', 5, 45, 0.328568, 0.00208, 0.02704),
    ('v9', 'navitoclax', 'Fisher', 'S_sen', 5, 45, 0.307477, 0.01007, 0.049833),
    ('v9', 'doxercalciferol', 'Fisher', 'S_sen', 5, 22, 0.46358, 0.01534, 0.049855),
    ('v9', 'azithromycin', 'Fisher', 'S_sen', 5, 26, 0.446838, 0.0115, 0.049833),
    ('v9', 'abt-737', 'DoseResponse_Consensus_Z', 'S_sen', 6, 45, 0.326614, 0.00997, 0.047242),
    ('v9', 'navitoclax', 'DoseResponse_Consensus_Z', 'S_sen', 6, 45, 0.339657, 0.01139, 0.047242),
    ('v9', 'dasatinib', 'DoseResponse_Consensus_Z', 'S_sen', 6, 45, 0.369433, 0.01817, 0.047242),
    ('v9', 'doxercalciferol', 'DoseResponse_Consensus_Z', 'S_sen', 6, 22, 0.46358, 0.01534, 0.047242),
    ('v9', 'azithromycin', 'DoseResponse_Consensus_Z', 'S_sen', 6, 26, 0.446838, 0.0115, 0.047242),
]
DRUG_LEDGER = [dict(zip(LEDGER_FIELDS, r)) for r in LEDGER_ROWS]
for r in DRUG_LEDGER: r['family'] = FAMILIES[r['family']]
V8_COMPOSITES = [{'composite': 'dasatinib+navitoclax', 'predictor': 'H_clean', 'n': 45, 'rho': 0.268906, 'p_positive': 0.01424, 'p_two_sided': 0.02848, 'ci_low': -0.039512, 'ci_high': 0.500247, 'permutations': 100000, 'bootstrap': 2000, 'q_positive': 0.018986}, {'composite': 'dasatinib+navitoclax', 'predictor': 'S_sen', 'n': 45, 'rho': 0.527931, 'p_positive': 0.00015, 'p_two_sided': 0.0003, 'ci_low': 0.224457, 'ci_high': 0.719325, 'permutations': 100000, 'bootstrap': 2000, 'q_positive': 0.0006}, {'composite': 'dasatinib+abt-737', 'predictor': 'H_clean', 'n': 45, 'rho': 0.327273, 'p_positive': 0.03122, 'p_two_sided': 0.062439, 'ci_low': 0.033294, 'ci_high': 0.564548, 'permutations': 100000, 'bootstrap': 2000, 'q_positive': 0.03122}, {'composite': 'dasatinib+abt-737', 'predictor': 'S_sen', 'n': 45, 'rho': 0.479578, 'p_positive': 0.00297, 'p_two_sided': 0.00594, 'ci_low': 0.192123, 'ci_high': 0.68353, 'permutations': 100000, 'bootstrap': 2000, 'q_positive': 0.00594}]
V8_CRISPR_INDICES = [{'index': 'seven_gene_vulnerability', 'predictor': 'H_clean', 'raw_negative_dependency': True, 'n': 51, 'rho': -0.498552, 'p_positive': 0.0001, 'p_two_sided': 0.0002, 'ci_low': -0.707548, 'ci_high': -0.22935, 'permutations': 100000, 'bootstrap': 2000, 'q_two_sided': 0.0006}, {'index': 'seven_gene_vulnerability', 'predictor': 'S_sen', 'raw_negative_dependency': True, 'n': 51, 'rho': -0.664525, 'p_positive': 1e-05, 'p_two_sided': 2e-05, 'ci_low': -0.79275, 'ci_high': -0.463334, 'permutations': 100000, 'bootstrap': 2000, 'q_two_sided': 0.00012}]
DRUGS = ['abt-737', 'navitoclax', 'dasatinib', 'paricalcitol', 'doxercalciferol', 'omipalisib', 'wye-125132', 'way-600', 'torin-1', 'everolimus', 'temsirolimus', 'onc201', '10-debc', 'mk-2206', 'afuresertib', 'ipatasertib', 'tic10']
DOOR2 = [
    {'cell': 'KURAMOCHI', 'donor': 'PT-6ZFCGi', 'model': 'ACH-000524', 'subtype': '2A', 'dr_subtype': '2B', 'H': 1.707732, 'S': 1.432597, 'buffer': -0.275135, 'v8': [0.622222, 0.266667, 0.6, 0.771429, 0.971429, 0.742857, 0.8, 0.314286, 0.742857, 0.514286, 0.266667, 0.205882, 0.628571, 0.8, 0.666667, 0.595238, 0.971429], 'dr': [0.422222, 0.422222, 0.533333, None, 0.272727, 0.565217, 0.740741, 0.428571, 0.653846, None, 0.136364, None, None, 0.711111, 0.487805, 0.575, None], 'primary_pct': {'onc201': 0.294118, 'tic10': 0.971429, '10-debc': 0.628571}},
    {'cell': 'OVKATE', 'donor': 'PT-9fFef6', 'model': 'ACH-000443', 'subtype': '2A', 'dr_subtype': '2B', 'H': 1.336345, 'S': 1.001592, 'buffer': -0.334753, 'v8': [0.777778, 0.2, 0.2, 0.485714, 0.371429, 0.057143, 0.114286, 0.342857, 0.085714, 0.857143, 0.088889, 0.5, 0.2, 0.088889, 0.047619, 0.071429, 0.542857], 'dr': [0.377778, 0.266667, 0.244444, None, None, None, None, None, None, None, 0.022727, None, None, 0.111111, 0.04878, 0.075, None], 'primary_pct': {'onc201': 0.558824, 'tic10': 0.542857, '10-debc': 0.2}},
    {'cell': 'OAW42', 'donor': 'PT-9pirK7', 'model': 'ACH-000704', 'subtype': '2A', 'dr_subtype': '2A', 'H': 0.815292, 'S': 1.954013, 'buffer': 1.138721, 'v8': [0.755556, 0.688889, 0.466667, 0.2, 0.285714, 0.657143, 0.685714, 0.428571, 0.428571, 0.628571, 0.044444, 0.588235, 0.285714, 0.733333, 0.690476, 0.309524, 0.485714], 'dr': [0.688889, 0.777778, 0.444444, 0.318182, None, 0.869565, 0.925926, 0.321429, 0.807692, 0.85, 0.522727, 0.692308, None, 0.866667, 0.780488, 0.275, None], 'primary_pct': {'onc201': 0.470588, 'tic10': 0.485714, '10-debc': 0.285714}},
    {'cell': 'COV362', 'donor': 'PT-BexFYH', 'model': 'ACH-000278', 'subtype': '2B', 'dr_subtype': '2B', 'H': 1.115709, 'S': 0.384253, 'buffer': -0.731456, 'v8': [0.4, 0.155556, 0.733333, 0.571429, 0.685714, 0.457143, 0.742857, 0.028571, 0.542857, 0.2, 0.133333, 0.382353, 0.457143, 0.222222, 0.285714, 0.166667, 0.342857], 'dr': [0.444444, 0.311111, 0.711111, 0.727273, 0.909091, 0.173913, 0.37037, 0.035714, 0.192308, 0.4, 0.113636, 0.461538, None, 0.266667, 0.414634, 0.2, None], 'primary_pct': {'onc201': 0.352941, 'tic10': 0.342857, '10-debc': 0.457143}},
    {'cell': 'JHOS-2', 'donor': 'PT-F6NvL5', 'model': 'ACH-000132', 'subtype': '2A', 'dr_subtype': '2A', 'H': 1.437536, 'S': 1.134838, 'buffer': -0.302698, 'v8': [0.288889, 0.666667, 0.222222, 0.057143, 0.057143, 0.371429, 0.228571, 0.085714, 0.514286, 0.285714, 0.177778, 0.529412, 0.685714, 0.133333, 0.261905, 0.142857, 0.171429], 'dr': [0.4, 0.666667, 0.266667, 0.227273, 0.181818, 0.086957, 0.148148, 0.071429, 0.230769, 0.2, 0.181818, 0.576923, None, 0.2, 0.317073, 0.225, None], 'primary_pct': {'onc201': 0.588235, 'tic10': 0.171429, '10-debc': 0.685714}},
    {'cell': 'SNU-840', 'donor': 'PT-S0isOw', 'model': 'ACH-000280', 'subtype': '2B', 'dr_subtype': '2A', 'H': 1.063988, 'S': -0.973136, 'buffer': -2.037124, 'v8': [0.466667, 0.133333, 0.822222, 1.0, 0.4, 0.6, 0.628571, 0.771429, 0.914286, 0.371429, 0.688889, 0.235294, 0.514286, 0.822222, 0.5, 0.571429, 0.085714], 'dr': [0.644444, 0.177778, 0.666667, 0.636364, 0.227273, 0.826087, 0.407407, 0.535714, 0.615385, None, 0.727273, 0.384615, None, 0.777778, 0.585366, 0.625, None], 'primary_pct': {'onc201': 0.205882, 'tic10': 0.085714, '10-debc': 0.514286}},
    {'cell': 'RMG-I', 'donor': 'PT-bnZfQQ', 'model': 'ACH-000719', 'subtype': '2A', 'dr_subtype': '2A', 'H': 2.168039, 'S': 5.575772, 'buffer': 3.407733, 'v8': [0.866667, 0.888889, 0.711111, 0.942857, 0.857143, 0.514286, 0.4, 0.628571, 0.457143, 0.714286, 0.644444, 0.294118, 0.314286, 0.622222, 0.52381, 0.714286, 0.571429], 'dr': [0.844444, 0.888889, 0.688889, 1.0, 0.954545, 0.782609, 0.851852, 0.464286, 0.923077, 0.75, 0.704545, 0.423077, None, 0.555556, 0.268293, 0.55, None], 'primary_pct': {'onc201': 0.323529, 'tic10': 0.571429, '10-debc': 0.314286}},
    {'cell': 'EFO-21', 'donor': 'PT-cdCVJ4', 'model': 'ACH-000308', 'subtype': '2B', 'dr_subtype': '2B', 'H': 2.996812, 'S': 4.084614, 'buffer': 1.087802, 'v8': [0.244444, 0.022222, 0.533333, 0.857143, 0.628571, 0.142857, 0.714286, 0.2, 0.028571, 0.085714, 0.111111, 0.441176, 1.0, 0.2, 0.214286, 0.285714, 1.0], 'dr': [0.355556, 0.022222, 0.355556, 0.818182, 0.363636, None, 0.259259, 0.214286, None, 0.2, 0.090909, 0.730769, None, 0.088889, 0.243902, 0.325, None], 'primary_pct': {'onc201': 0.264706, 'tic10': 1.0, '10-debc': 1.0}},
    {'cell': 'Caov-4', 'donor': 'PT-gCHPXX', 'model': 'ACH-000103', 'subtype': '2A', 'dr_subtype': '2A', 'H': 1.052132, 'S': -2.436435, 'buffer': -3.488567, 'v8': [0.822222, 0.6, 0.022222, None, None, None, None, None, None, None, 0.488889, None, None, 0.911111, 0.309524, 0.47619, None], 'dr': [0.8, 0.644444, 0.022222, None, None, None, None, None, None, None, 0.5, None, None, 0.888889, 0.219512, 0.425, None], 'primary_pct': {'onc201': None, 'tic10': None, '10-debc': None}},
    {'cell': 'OV56', 'donor': 'PT-kmyeHR', 'model': 'ACH-000091', 'subtype': '2A', 'dr_subtype': '2A', 'H': 5.089128, 'S': 4.528038, 'buffer': -0.56109, 'v8': [0.644444, 0.711111, 0.644444, 0.257143, 0.314286, 0.771429, 0.457143, 0.257143, 0.628571, 0.742857, 0.422222, 0.735294, 0.714286, 0.288889, 0.571429, 0.238095, 0.771429], 'dr': [0.6, 0.577778, 0.555556, 0.272727, None, None, 0.444444, 0.821429, 0.346154, 0.95, 0.636364, 0.538462, None, 0.422222, 0.658537, 0.35, None], 'primary_pct': {'onc201': 0.735294, 'tic10': 0.771429, '10-debc': 0.714286}},
    {'cell': 'OVISE', 'donor': 'PT-oYazft', 'model': 'ACH-000527', 'subtype': '2A', 'dr_subtype': '2A', 'H': 2.088325, 'S': 3.4671, 'buffer': 1.378775, 'v8': [0.955556, 0.866667, 0.622222, 0.685714, 0.942857, 0.4, 0.885714, 0.828571, 0.4, 0.142857, 0.222222, 0.352941, 0.942857, 0.711111, 0.642857, 0.619048, 0.657143], 'dr': [0.888889, 0.866667, 0.755556, 0.863636, 0.772727, 0.347826, 0.962963, 0.892857, 0.307692, 0.2, 0.295455, 0.153846, None, 0.666667, 0.609756, 0.475, None], 'primary_pct': {'onc201': 0.5, 'tic10': 0.657143, '10-debc': 0.942857}},
]
for r in DOOR2:
    for ep in ('v8', 'dr'): r[ep] = dict(zip(DRUGS, r[ep]))
CAVEATS = ("Post-hoc families; no independent validation.",
           "Percentile OR coverage is not rescue, synergy or efficacy.",
           "ONC201 and TIC10 cache identities are not collapsed.",
           "Primary-only TIC10 and 10-DEBC have no dose-response consensus.",
           "Expression subtype data: 8 of 8 2A and 2 of 3 2B.",
           "CRISPR subtype data: 6 of 8 2A and 2 of 3 2B.")
def crispr_index(raw):
    """Mean population-z raw Chronos; lower means stronger dependency."""
    if not all(g in raw and isfinite(raw[g]) for g in CRISPR_STANDARDIZERS):
        raise ValueError("All seven finite raw Chronos values are required")
    return sum((raw[g]-m)/s for g,(m,s) in CRISPR_STANDARDIZERS.items())/7
def observed_coverage(drugs, threshold=0.5, endpoint='dr'):
    """Strict OR of observed percentiles; unknown if incomplete and not active."""
    counts = dict(active=0, inactive=0, unknown=0)
    for row in DOOR2:
        vals = [row[endpoint].get(g) for g in drugs]
        state = 'active' if any(v is not None and v > threshold for v in vals) else (
            'inactive' if vals and all(v is not None for v in vals) else 'unknown')
        counts[state] += 1
    return counts
