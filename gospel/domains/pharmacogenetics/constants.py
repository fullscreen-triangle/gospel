"""
Pharmacogenetics domain constants for Gospel.

This module defines reference data and constants for pharmacogenetic analysis.
"""

# Genes involved in drug metabolism
METABOLISM_GENES = {
    "CYP2D6": {
        "impact_score": 0.95,
        "description": "Cytochrome P450 2D6, metabolizes ~25% of all drugs",
        "affected_drugs": [
            "codeine", "tramadol", "tamoxifen", "fluoxetine", "paroxetine",
            "amitriptyline", "metoprolol", "ondansetron"
        ],
        "evidence_level": "high"
    },
    "CYP2C19": {
        "impact_score": 0.90,
        "description": "Cytochrome P450 2C19, metabolizes many antidepressants and PPIs",
        "affected_drugs": [
            "clopidogrel", "escitalopram", "citalopram", "omeprazole",
            "pantoprazole", "diazepam", "voriconazole"
        ],
        "evidence_level": "high"
    },
    "CYP2C9": {
        "impact_score": 0.85,
        "description": "Cytochrome P450 2C9, metabolizes warfarin and many NSAIDs",
        "affected_drugs": [
            "warfarin", "phenytoin", "celecoxib", "ibuprofen", "naproxen",
            "fluvastatin", "losartan"
        ],
        "evidence_level": "high"
    },
    "CYP3A4": {
        "impact_score": 0.80,
        "description": "Cytochrome P450 3A4, metabolizes ~50% of all drugs",
        "affected_drugs": [
            "atorvastatin", "simvastatin", "tacrolimus", "cyclosporine",
            "midazolam", "carbamazepine", "erythromycin"
        ],
        "evidence_level": "moderate"
    },
    "CYP3A5": {
        "impact_score": 0.75,
        "description": "Cytochrome P450 3A5, contributes to metabolism of many drugs",
        "affected_drugs": [
            "tacrolimus", "cyclosporine", "saquinavir", "midazolam"
        ],
        "evidence_level": "moderate"
    },
    "VKORC1": {
        "impact_score": 0.90,
        "description": "Vitamin K epoxide reductase, target of warfarin",
        "affected_drugs": ["warfarin"],
        "evidence_level": "high"
    },
    "DPYD": {
        "impact_score": 0.95,
        "description": "Dihydropyrimidine dehydrogenase, metabolizes fluoropyrimidines",
        "affected_drugs": ["fluorouracil", "capecitabine"],
        "evidence_level": "high"
    },
    "TPMT": {
        "impact_score": 0.90,
        "description": "Thiopurine S-methyltransferase, metabolizes thiopurines",
        "affected_drugs": ["azathioprine", "mercaptopurine", "thioguanine"],
        "evidence_level": "high"
    },
    "UGT1A1": {
        "impact_score": 0.85,
        "description": "UDP glucuronosyltransferase, metabolizes many drugs",
        "affected_drugs": ["irinotecan", "atazanavir", "raltegravir"],
        "evidence_level": "high"
    },
    "SLCO1B1": {
        "impact_score": 0.80,
        "description": "Solute carrier organic anion transporter, transports statins",
        "affected_drugs": ["simvastatin", "pravastatin", "atorvastatin", "rosuvastatin"],
        "evidence_level": "high"
    }
}

# Genes encoding drug targets
DRUG_TARGET_GENES = {
    "ADRB2": {
        "impact_score": 0.75,
        "description": "Beta-2 adrenergic receptor, target of beta agonists",
        "affected_drugs": ["albuterol", "salmeterol", "formoterol"],
        "evidence_level": "moderate"
    },
    "HTR2A": {
        "impact_score": 0.70,
        "description": "Serotonin receptor 2A, target of many antipsychotics and antidepressants",
        "affected_drugs": ["clozapine", "olanzapine", "risperidone", "quetiapine"],
        "evidence_level": "moderate"
    },
    "DRD2": {
        "impact_score": 0.75,
        "description": "Dopamine receptor D2, target of antipsychotics",
        "affected_drugs": ["haloperidol", "risperidone", "aripiprazole"],
        "evidence_level": "moderate"
    },
    "OPRM1": {
        "impact_score": 0.70,
        "description": "Mu opioid receptor, target of opioid analgesics",
        "affected_drugs": ["morphine", "fentanyl", "oxycodone", "naloxone"],
        "evidence_level": "moderate"
    },
    "ACE": {
        "impact_score": 0.65,
        "description": "Angiotensin converting enzyme, target of ACE inhibitors",
        "affected_drugs": ["lisinopril", "enalapril", "ramipril"],
        "evidence_level": "moderate"
    },
    "CFTR": {
        "impact_score": 0.95,
        "description": "Cystic fibrosis transmembrane conductance regulator",
        "affected_drugs": ["ivacaftor", "lumacaftor", "tezacaftor", "elexacaftor"],
        "evidence_level": "high"
    }
}

# Genes involved in drug pathways
DRUG_PATHWAY_GENES = {
    "EGFR": {
        "impact_score": 0.85,
        "description": "Epidermal growth factor receptor, target of cancer therapies",
        "affected_drugs": ["erlotinib", "gefitinib", "afatinib", "osimertinib"],
        "evidence_level": "high"
    },
    "BRAF": {
        "impact_score": 0.90,
        "description": "B-Raf proto-oncogene, target of cancer therapies",
        "affected_drugs": ["vemurafenib", "dabrafenib", "encorafenib"],
        "evidence_level": "high"
    },
    "KRAS": {
        "impact_score": 0.80,
        "description": "KRAS proto-oncogene, biomarker for cancer therapies",
        "affected_drugs": ["cetuximab", "panitumumab"],
        "evidence_level": "high"
    },
    "BRCA1": {
        "impact_score": 0.85,
        "description": "BRCA1 DNA repair associated, biomarker for PARP inhibitors",
        "affected_drugs": ["olaparib", "rucaparib", "niraparib"],
        "evidence_level": "high"
    },
    "BRCA2": {
        "impact_score": 0.85,
        "description": "BRCA2 DNA repair associated, biomarker for PARP inhibitors",
        "affected_drugs": ["olaparib", "rucaparib", "niraparib"],
        "evidence_level": "high"
    },
    "ERBB2": {  # HER2
        "impact_score": 0.85,
        "description": "Erb-B2 receptor tyrosine kinase 2, target of cancer therapies",
        "affected_drugs": ["trastuzumab", "pertuzumab", "lapatinib", "ado-trastuzumab emtansine"],
        "evidence_level": "high"
    }
}

# Genes associated with adverse drug reactions
ADVERSE_EFFECT_GENES = {
    "HLA-B": {
        "impact_score": 0.95,
        "description": "HLA-B gene, associated with severe drug reactions",
        "affected_drugs": ["abacavir", "allopurinol", "carbamazepine"],
        "evidence_level": "high"
    },
    "HLA-A": {
        "impact_score": 0.90,
        "description": "HLA-A gene, associated with drug reactions",
        "affected_drugs": ["carbamazepine", "oxcarbazepine"],
        "evidence_level": "high"
    },
    "G6PD": {
        "impact_score": 0.95,
        "description": "Glucose-6-phosphate dehydrogenase, deficiency causes hemolysis",
        "affected_drugs": ["primaquine", "rasburicase", "nitrofurantoin", "sulfamethoxazole"],
        "evidence_level": "high"
    },
    "NUDT15": {
        "impact_score": 0.90,
        "description": "Nudix hydrolase 15, variants cause thiopurine toxicity",
        "affected_drugs": ["azathioprine", "mercaptopurine", "thioguanine"],
        "evidence_level": "high"
    },
    "RYR1": {
        "impact_score": 0.90,
        "description": "Ryanodine receptor 1, associated with malignant hyperthermia",
        "affected_drugs": ["succinylcholine", "halothane", "isoflurane", "desflurane"],
        "evidence_level": "high"
    },
    "SCN5A": {
        "impact_score": 0.85,
        "description": "Sodium voltage-gated channel alpha subunit 5, associated with drug-induced QT prolongation",
        "affected_drugs": ["quinidine", "flecainide", "amiodarone"],
        "evidence_level": "moderate"
    },
    "KCNH2": {
        "impact_score": 0.85,
        "description": "Potassium voltage-gated channel subfamily H member 2, associated with drug-induced QT prolongation",
        "affected_drugs": ["erythromycin", "clarithromycin", "haloperidol", "quetiapine"],
        "evidence_level": "moderate"
    }
}

# Pharmacogenetic drug information
PGX_DRUGS = {
    "warfarin": {
        "description": "Anticoagulant affected by CYP2C9 and VKORC1 variants",
        "key_genes": ["CYP2C9", "VKORC1", "CYP4F2"],
        "recommendation_logic": {
            "poor": "Consider 20-50% dose reduction and more frequent INR monitoring.",
            "intermediate": "Consider 10-25% dose reduction and more frequent INR monitoring.",
            "ultrarapid": "Standard initial dose may be insufficient, monitor closely.",
            "general": "Genetic testing can help determine optimal warfarin dosing."
        }
    },
    "clopidogrel": {
        "description": "Antiplatelet affected by CYP2C19 variants",
        "key_genes": ["CYP2C19"],
        "recommendation_logic": {
            "poor": "Consider alternative antiplatelet therapy (e.g., prasugrel, ticagrelor).",
            "intermediate": "Consider alternative antiplatelet therapy if high risk.",
            "general": "CYP2C19 poor metabolizers may have reduced efficacy of clopidogrel."
        }
    },
    "simvastatin": {
        "description": "Statin affected by SLCO1B1 variants",
        "key_genes": ["SLCO1B1", "CYP3A4", "CYP3A5"],
        "recommendation_logic": {
            "general": "SLCO1B1 variants increase risk of myopathy with high-dose simvastatin."
        }
    },
    "codeine": {
        "description": "Opioid prodrug metabolized by CYP2D6",
        "key_genes": ["CYP2D6"],
        "recommendation_logic": {
            "poor": "Poor metabolizers have reduced analgesic effect, consider alternative.",
            "ultrarapid": "Ultrarapid metabolizers at increased risk for toxicity, avoid use.",
            "general": "Efficacy and safety of codeine affected by CYP2D6 metabolism."
        }
    },
    "tamoxifen": {
        "description": "Selective estrogen receptor modulator metabolized by CYP2D6",
        "key_genes": ["CYP2D6"],
        "recommendation_logic": {
            "poor": "Poor metabolizers may have reduced efficacy, consider alternative.",
            "intermediate": "May have reduced efficacy, consider higher dose or alternative.",
            "general": "CYP2D6 activity affects conversion to active metabolite endoxifen."
        }
    },
    "abacavir": {
        "description": "Antiretroviral associated with hypersensitivity in HLA-B*57:01 carriers",
        "key_genes": ["HLA-B"],
        "recommendation_logic": {
            "general": "Screen for HLA-B*57:01 before abacavir use to prevent hypersensitivity."
        }
    },
    "fluorouracil": {
        "description": "Chemotherapy metabolized by DPYD",
        "key_genes": ["DPYD"],
        "recommendation_logic": {
            "poor": "Consider 50% or greater dose reduction or alternative therapy.",
            "intermediate": "Consider 25-50% dose reduction and close monitoring.",
            "general": "DPYD deficiency increases risk of severe toxicity."
        }
    },
    "azathioprine": {
        "description": "Immunosuppressant metabolized by TPMT and NUDT15",
        "key_genes": ["TPMT", "NUDT15"],
        "recommendation_logic": {
            "poor": "Consider 90% dose reduction or alternative therapy.",
            "intermediate": "Consider 30-50% dose reduction.",
            "general": "Deficiency in TPMT or NUDT15 increases risk of myelosuppression."
        }
    }
}

# Key pharmacogenetic variants with clinical significance
PGX_VARIANTS = {
    "rs4244285": {
        "gene": "CYP2C19",
        "allele": "*2",
        "effect": "Loss of function",
        "drugs": ["clopidogrel", "escitalopram", "omeprazole"],
        "clinical_significance": "Poor metabolism of CYP2C19 substrates"
    },
    "rs1799853": {
        "gene": "CYP2C9",
        "allele": "*2",
        "effect": "Reduced function",
        "drugs": ["warfarin", "phenytoin", "NSAIDs"],
        "clinical_significance": "Reduced metabolism of CYP2C9 substrates"
    },
    "rs1057910": {
        "gene": "CYP2C9",
        "allele": "*3",
        "effect": "Reduced function",
        "drugs": ["warfarin", "phenytoin", "NSAIDs"],
        "clinical_significance": "Greatly reduced metabolism of CYP2C9 substrates"
    },
    "rs16947": {
        "gene": "CYP2D6",
        "allele": "*2",
        "effect": "Normal function",
        "drugs": ["codeine", "tamoxifen", "antidepressants"],
        "clinical_significance": "Normal metabolism of CYP2D6 substrates"
    },
    "rs3892097": {
        "gene": "CYP2D6",
        "allele": "*4",
        "effect": "Loss of function",
        "drugs": ["codeine", "tamoxifen", "antidepressants"],
        "clinical_significance": "Poor metabolism of CYP2D6 substrates"
    },
    "rs9923231": {
        "gene": "VKORC1",
        "allele": "-1639G>A",
        "effect": "Reduced expression",
        "drugs": ["warfarin"],
        "clinical_significance": "Increased sensitivity to warfarin"
    },
    "rs4149056": {
        "gene": "SLCO1B1",
        "allele": "*5",
        "effect": "Reduced function",
        "drugs": ["simvastatin", "other statins"],
        "clinical_significance": "Increased risk of statin-induced myopathy"
    },
    "rs3211371": {
        "gene": "CYP2B6",
        "allele": "*5",
        "effect": "Reduced function",
        "drugs": ["efavirenz", "nevirapine"],
        "clinical_significance": "Altered metabolism of CYP2B6 substrates"
    },
    "rs2231142": {
        "gene": "ABCG2",
        "allele": "Q141K",
        "effect": "Reduced function",
        "drugs": ["rosuvastatin", "allopurinol"],
        "clinical_significance": "Reduced drug efflux"
    },
    "rs1045642": {
        "gene": "ABCB1",
        "allele": "3435C>T",
        "effect": "Altered function",
        "drugs": ["digoxin", "cyclosporine", "many others"],
        "clinical_significance": "Altered drug transport"
    },
    "rs4149013": {
        "gene": "SLCO1A2",
        "allele": "38T>C",
        "effect": "Reduced function",
        "drugs": ["methotrexate", "fexofenadine"],
        "clinical_significance": "Reduced drug uptake"
    },
    "rs2470890": {
        "gene": "CYP1A2",
        "allele": "*1F",
        "effect": "Increased inducibility",
        "drugs": ["caffeine", "clozapine", "olanzapine"],
        "clinical_significance": "Increased metabolism with smoking"
    }
} 