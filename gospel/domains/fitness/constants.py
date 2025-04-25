"""
Fitness domain constants for Gospel.

This module defines reference data and constants for fitness-related genomic analysis.
"""

# List of all fitness-related traits
FITNESS_TRAITS = [
    "sprint",
    "power",
    "endurance",
    "injury_risk",
    "recovery",
    "muscle_composition",
    "vo2_max",
    "metabolism",
    "tendon_strength",
    "joint_flexibility",
    "balance",
    "coordination"
]

# Genes associated with general athletic performance
PERFORMANCE_GENES = {
    "ACE": {
        "score": 0.85,
        "description": "Angiotensin-converting enzyme, associated with both power and endurance",
        "alleles": {"I": "endurance", "D": "power"},
        "power_factor": 0.7,
        "endurance_factor": 0.8
    },
    "ACTN3": {
        "score": 0.90,
        "description": "Alpha-actinin-3, the 'speed gene', associated with fast-twitch muscle fibers",
        "alleles": {"R": "power", "X": "endurance"},
        "power_factor": 0.9,
        "sprint_factor": 0.95,
        "muscle_comp_factor": 0.85
    },
    "AGT": {
        "score": 0.65,
        "description": "Angiotensinogen, affects blood pressure regulation during exercise",
        "power_factor": 0.6,
        "endurance_factor": 0.7
    },
    "NRF1": {
        "score": 0.75,
        "description": "Nuclear respiratory factor 1, involved in mitochondrial biogenesis",
        "endurance_factor": 0.8,
        "metabolism_factor": 0.7
    },
    "NRF2": {
        "score": 0.70,
        "description": "Nuclear factor erythroid 2-related factor 2, antioxidant response",
        "endurance_factor": 0.75,
        "recovery_factor": 0.65
    }
}

# Genes specifically associated with endurance performance
ENDURANCE_GENES = {
    "PPARGC1A": {
        "score": 0.85,
        "description": "PGC-1α, key regulator of mitochondrial biogenesis",
        "metabolism_factor": 0.8
    },
    "EPAS1": {
        "score": 0.80,
        "description": "Endothelial PAS domain protein 1, adaptation to high-altitude/low oxygen",
        "vo2_max_factor": 0.85
    },
    "ADRB2": {
        "score": 0.70,
        "description": "Beta-2 adrenergic receptor, affects lung function and cardiac output",
        "metabolism_factor": 0.65
    },
    "VEGFA": {
        "score": 0.75,
        "description": "Vascular endothelial growth factor A, promotes angiogenesis",
        "vo2_max_factor": 0.7
    },
    "HIF1A": {
        "score": 0.80,
        "description": "Hypoxia-inducible factor 1-alpha, cellular response to hypoxia",
        "vo2_max_factor": 0.75,
        "metabolism_factor": 0.7
    },
    "AMPD1": {
        "score": 0.65,
        "description": "Adenosine monophosphate deaminase 1, energy metabolism in muscle",
        "metabolism_factor": 0.6
    }
}

# Genes specifically associated with power/strength performance
POWER_GENES = {
    "ACTN3": {  # Duplicated from performance genes for convenience
        "score": 0.90,
        "description": "Alpha-actinin-3, the 'speed gene', associated with fast-twitch muscle fibers",
        "sprint_factor": 0.95,
        "muscle_comp_factor": 0.85
    },
    "MSTN": {
        "score": 0.85,
        "description": "Myostatin, negative regulator of muscle growth",
        "muscle_comp_factor": 0.9
    },
    "IGF1": {
        "score": 0.80,
        "description": "Insulin-like growth factor 1, promotes muscle growth",
        "muscle_comp_factor": 0.85,
        "recovery_factor": 0.7
    },
    "IL6": {
        "score": 0.65,
        "description": "Interleukin 6, inflammation and muscle repair",
        "recovery_factor": 0.7,
        "sprint_factor": 0.6
    },
    "MTOR": {
        "score": 0.75,
        "description": "Mechanistic target of rapamycin, protein synthesis and muscle growth",
        "muscle_comp_factor": 0.8,
        "recovery_factor": 0.7
    }
}

# Genes associated with injury risk
INJURY_RISK_GENES = {
    "COL1A1": {
        "score": 0.85,
        "description": "Collagen type I alpha 1 chain, major component of connective tissue",
        "tendon_strength_factor": 0.9
    },
    "COL5A1": {
        "score": 0.80,
        "description": "Collagen type V alpha 1 chain, regulates collagen fibril assembly",
        "tendon_strength_factor": 0.85,
        "joint_flexibility_factor": 0.75
    },
    "MMP3": {
        "score": 0.70,
        "description": "Matrix metallopeptidase 3, degrades collagens and proteoglycans",
        "tendon_strength_factor": 0.65,
        "recovery_factor": 0.6
    },
    "ACAN": {
        "score": 0.75,
        "description": "Aggrecan, major component of cartilage",
        "joint_flexibility_factor": 0.8
    },
    "GDF5": {
        "score": 0.65,
        "description": "Growth differentiation factor 5, cartilage and joint development",
        "joint_flexibility_factor": 0.7
    },
    "TIMP2": {
        "score": 0.60,
        "description": "TIMP metallopeptidase inhibitor 2, inhibits MMPs",
        "tendon_strength_factor": 0.65
    }
}

# Genes associated with recovery
RECOVERY_GENES = {
    "IL6": {  # Duplicated from power genes for convenience
        "score": 0.75,
        "description": "Interleukin 6, inflammation and muscle repair",
        "recovery_factor": 0.8
    },
    "TNF": {
        "score": 0.70,
        "description": "Tumor necrosis factor, systemic inflammation",
        "recovery_factor": 0.75
    },
    "IL10": {
        "score": 0.80,
        "description": "Interleukin 10, anti-inflammatory cytokine",
        "recovery_factor": 0.85
    },
    "CRP": {
        "score": 0.65,
        "description": "C-reactive protein, acute phase inflammation marker",
        "recovery_factor": 0.7
    },
    "SOD2": {
        "score": 0.75,
        "description": "Superoxide dismutase 2, antioxidant defense",
        "recovery_factor": 0.8,
        "metabolism_factor": 0.65
    },
    "HSPA1A": {
        "score": 0.70,
        "description": "Heat shock protein family A member 1A, stress response",
        "recovery_factor": 0.75
    }
}

# Key SNPs associated with fitness traits
FITNESS_SNPS = {
    "rs1815739": {
        "gene": "ACTN3",
        "trait": "power",
        "effect_allele": "C",
        "other_allele": "T",
        "effect_size": 0.85,
        "description": "R577X polymorphism, T allele results in non-functional protein"
    },
    "rs4253778": {
        "gene": "PPARA",
        "trait": "endurance",
        "effect_allele": "G",
        "other_allele": "C",
        "effect_size": 0.70,
        "description": "G allele associated with endurance performance"
    },
    "rs1799752": {
        "gene": "ACE",
        "trait": "mixed",
        "effect_allele": "D",
        "other_allele": "I",
        "effect_size": 0.75,
        "description": "I/D polymorphism, I allele with endurance, D with power"
    },
    "rs8192678": {
        "gene": "PPARGC1A",
        "trait": "endurance",
        "effect_allele": "A",
        "other_allele": "G",
        "effect_size": 0.65,
        "description": "Gly482Ser polymorphism, G allele associated with better endurance"
    },
    "rs2228570": {
        "gene": "VDR",
        "trait": "muscle_strength",
        "effect_allele": "C",
        "other_allele": "T",
        "effect_size": 0.60,
        "description": "FokI polymorphism, impacts vitamin D signaling"
    },
    "rs12722": {
        "gene": "COL5A1",
        "trait": "injury_risk",
        "effect_allele": "C",
        "other_allele": "T",
        "effect_size": 0.70,
        "description": "C allele associated with reduced injury risk"
    }
} 