"""
Constants and reference data for the nutrition domain.

This module provides reference data for nutrients, food sensitivities, gene-nutrient
relationships, and SNP-effect mappings used in nutritional genomics analysis.
"""

# Nutrient reference data
NUTRIENTS = {
    "VIT_D": {
        "name": "Vitamin D",
        "description": "Fat-soluble vitamin important for calcium absorption and bone health",
        "rda": 600,  # IU per day for adults
        "units": "IU",
        "upper_limit": 4000,  # IU per day
        "food_sources": [
            "Fatty fish (salmon, mackerel)",
            "Fortified milk",
            "Fortified orange juice",
            "Egg yolks",
            "Mushrooms exposed to UV light"
        ],
        "supplement_forms": ["D2 (ergocalciferol)", "D3 (cholecalciferol)"]
    },
    "VIT_B12": {
        "name": "Vitamin B12",
        "description": "Water-soluble vitamin essential for nerve function and blood cell formation",
        "rda": 2.4,  # mcg per day for adults
        "units": "mcg",
        "upper_limit": None,  # No established upper limit
        "food_sources": [
            "Shellfish (clams, oysters)",
            "Liver",
            "Fish (salmon, trout)",
            "Fortified nutritional yeast",
            "Fortified plant milks"
        ],
        "supplement_forms": ["Cyanocobalamin", "Methylcobalamin"]
    },
    "FOLATE": {
        "name": "Folate",
        "description": "B vitamin essential for cell division and DNA synthesis",
        "rda": 400,  # mcg DFE per day for adults
        "units": "mcg DFE",
        "upper_limit": 1000,  # mcg per day from supplements
        "food_sources": [
            "Leafy greens (spinach, kale)",
            "Legumes (lentils, beans)",
            "Asparagus",
            "Avocado",
            "Fortified grains"
        ],
        "supplement_forms": ["Folic acid", "Methylfolate"]
    },
    "IRON": {
        "name": "Iron",
        "description": "Mineral essential for oxygen transport in the blood",
        "rda": 18,  # mg per day for adult women (8mg for men)
        "units": "mg",
        "upper_limit": 45,  # mg per day
        "food_sources": [
            "Red meat",
            "Oysters",
            "Spinach",
            "Lentils",
            "Pumpkin seeds"
        ],
        "supplement_forms": ["Ferrous sulfate", "Ferrous gluconate", "Iron bisglycinate"]
    },
    "OMEGA3": {
        "name": "Omega-3 Fatty Acids",
        "description": "Essential fatty acids important for heart and brain health",
        "rda": 1.6,  # g per day for adult men (1.1g for women)
        "units": "g",
        "upper_limit": 3,  # g per day from supplements
        "food_sources": [
            "Fatty fish (salmon, sardines)",
            "Flaxseeds",
            "Chia seeds",
            "Walnuts",
            "Algal oil"
        ],
        "supplement_forms": ["Fish oil", "Algal oil", "Krill oil"]
    },
    "VIT_A": {
        "name": "Vitamin A",
        "description": "Fat-soluble vitamin important for vision and immune function",
        "rda": 900,  # mcg RAE per day for adult men (700mcg for women)
        "units": "mcg RAE",
        "upper_limit": 3000,  # mcg per day
        "food_sources": [
            "Sweet potato",
            "Beef liver",
            "Spinach",
            "Carrots",
            "Red bell peppers"
        ],
        "supplement_forms": ["Retinol", "Beta-carotene"]
    },
    "CALCIUM": {
        "name": "Calcium",
        "description": "Mineral essential for bone health and muscle function",
        "rda": 1000,  # mg per day for adults
        "units": "mg",
        "upper_limit": 2500,  # mg per day
        "food_sources": [
            "Dairy products",
            "Fortified plant milks",
            "Tofu (made with calcium sulfate)",
            "Sardines (with bones)",
            "Leafy greens (kale, bok choy)"
        ],
        "supplement_forms": ["Calcium carbonate", "Calcium citrate"]
    },
    "ZINC": {
        "name": "Zinc",
        "description": "Mineral essential for immune function and wound healing",
        "rda": 11,  # mg per day for adult men (8mg for women)
        "units": "mg",
        "upper_limit": 40,  # mg per day
        "food_sources": [
            "Oysters",
            "Red meat",
            "Pumpkin seeds",
            "Lentils",
            "Hemp seeds"
        ],
        "supplement_forms": ["Zinc gluconate", "Zinc picolinate", "Zinc acetate"]
    },
    "MAGNESIUM": {
        "name": "Magnesium",
        "description": "Mineral essential for nerve and muscle function",
        "rda": 400,  # mg per day for adult men (310mg for women)
        "units": "mg",
        "upper_limit": 350,  # mg per day from supplements
        "food_sources": [
            "Pumpkin seeds",
            "Spinach",
            "Black beans",
            "Dark chocolate",
            "Avocado"
        ],
        "supplement_forms": ["Magnesium glycinate", "Magnesium citrate", "Magnesium oxide"]
    },
    "VIT_C": {
        "name": "Vitamin C",
        "description": "Water-soluble vitamin important for immune function and collagen synthesis",
        "rda": 90,  # mg per day for adult men (75mg for women)
        "units": "mg",
        "upper_limit": 2000,  # mg per day
        "food_sources": [
            "Citrus fruits",
            "Bell peppers",
            "Strawberries",
            "Kiwi",
            "Broccoli"
        ],
        "supplement_forms": ["Ascorbic acid", "Mineral ascorbates"]
    }
}

# Food sensitivity categories
FOOD_SENSITIVITIES = {
    "GLUTEN": {
        "name": "Gluten",
        "description": "Protein found in wheat, barley, and rye",
        "foods_to_avoid": [
            "Wheat bread and pasta",
            "Barley",
            "Rye",
            "Many processed foods",
            "Beer"
        ],
        "alternatives": [
            "Rice",
            "Quinoa",
            "Gluten-free bread and pasta",
            "Buckwheat",
            "Certified gluten-free oats"
        ],
        "related_genes": ["HLA-DQ2", "HLA-DQ8", "MTHFR", "CTLA4"]
    },
    "LACTOSE": {
        "name": "Dairy/Lactose",
        "description": "Milk sugar found in dairy products",
        "foods_to_avoid": [
            "Milk",
            "Ice cream",
            "Soft cheeses",
            "Yogurt",
            "Cream"
        ],
        "alternatives": [
            "Plant-based milks (almond, soy, oat)",
            "Coconut yogurt",
            "Dairy-free ice cream",
            "Hard aged cheeses (if tolerated)",
            "Lactose-free dairy products"
        ],
        "related_genes": ["MCM6", "LCT"]
    },
    "HISTAMINE": {
        "name": "Histamine",
        "description": "Compound involved in immune responses, found in certain foods",
        "foods_to_avoid": [
            "Fermented foods (sauerkraut, kimchi)",
            "Aged cheeses",
            "Cured meats",
            "Alcohol",
            "Vinegar"
        ],
        "alternatives": [
            "Fresh meats",
            "Fresh fruits and vegetables",
            "Fresh fish (not canned)",
            "Eggs",
            "Grains"
        ],
        "related_genes": ["DAO", "HNMT", "MTHFR"]
    },
    "FODMAPS": {
        "name": "FODMAPs",
        "description": "Fermentable carbohydrates that can cause digestive issues",
        "foods_to_avoid": [
            "Onions and garlic",
            "Wheat",
            "Certain fruits (apples, pears)",
            "Legumes",
            "Sweeteners (honey, high-fructose corn syrup)"
        ],
        "alternatives": [
            "Rice",
            "Quinoa",
            "Low-FODMAP vegetables",
            "Low-FODMAP fruits (berries, citrus)",
            "Lactose-free dairy"
        ],
        "related_genes": ["OCTN1", "OCTN2", "NOD2"]
    },
    "CAFFEINE": {
        "name": "Caffeine",
        "description": "Stimulant found in coffee, tea, and chocolate",
        "foods_to_avoid": [
            "Coffee",
            "Black and green tea",
            "Energy drinks",
            "Chocolate",
            "Some medications"
        ],
        "alternatives": [
            "Herbal tea",
            "Decaffeinated coffee",
            "Chicory root coffee",
            "Rooibos tea",
            "Carob (chocolate alternative)"
        ],
        "related_genes": ["CYP1A2", "ADORA2A"]
    },
    "NIGHTSHADES": {
        "name": "Nightshades",
        "description": "Plant family that includes tomatoes, potatoes, peppers, and eggplants",
        "foods_to_avoid": [
            "Tomatoes",
            "Potatoes (not sweet potatoes)",
            "Bell peppers and hot peppers",
            "Eggplant",
            "Goji berries"
        ],
        "alternatives": [
            "Sweet potatoes",
            "Cauliflower",
            "Zucchini",
            "Carrots",
            "Celery"
        ],
        "related_genes": ["IL6", "TNF", "IL1B"]
    }
}

# Gene-nutrient relationships
GENE_NUTRIENT_RELATIONSHIPS = {
    "MTHFR": {
        "description": "Methylenetetrahydrofolate reductase, important for folate metabolism",
        "nutrients": ["FOLATE", "VIT_B12", "VIT_B6"],
        "variants": ["rs1801133", "rs1801131"],
        "related_pathways": ["One-carbon metabolism", "Homocysteine metabolism"]
    },
    "VDR": {
        "description": "Vitamin D receptor, mediates vitamin D action",
        "nutrients": ["VIT_D", "CALCIUM"],
        "variants": ["rs1544410", "rs2228570", "rs731236"],
        "related_pathways": ["Bone metabolism", "Calcium absorption", "Immune function"]
    },
    "BCMO1": {
        "description": "Beta-carotene monooxygenase 1, converts beta-carotene to vitamin A",
        "nutrients": ["VIT_A"],
        "variants": ["rs12934922", "rs7501331"],
        "related_pathways": ["Vitamin A metabolism", "Antioxidant function"]
    },
    "FADS1": {
        "description": "Fatty acid desaturase 1, involved in omega-3 and omega-6 metabolism",
        "nutrients": ["OMEGA3", "OMEGA6"],
        "variants": ["rs174537", "rs174546", "rs174548"],
        "related_pathways": ["Essential fatty acid metabolism", "Inflammation"]
    },
    "HFE": {
        "description": "Hemochromatosis gene, regulates iron absorption",
        "nutrients": ["IRON"],
        "variants": ["rs1800562", "rs1799945"],
        "related_pathways": ["Iron metabolism", "Oxidative stress"]
    },
    "SLC30A8": {
        "description": "Solute carrier family 30 member 8, zinc transporter in pancreatic beta cells",
        "nutrients": ["ZINC"],
        "variants": ["rs13266634"],
        "related_pathways": ["Glucose metabolism", "Insulin secretion"]
    },
    "APOA5": {
        "description": "Apolipoprotein A5, involved in triglyceride metabolism",
        "nutrients": ["OMEGA3", "OMEGA6"],
        "variants": ["rs662799", "rs3135506"],
        "related_pathways": ["Lipid metabolism", "Cardiovascular health"]
    },
    "COMT": {
        "description": "Catechol-O-methyltransferase, degrades catecholamines",
        "nutrients": ["VIT_B6", "MAGNESIUM"],
        "variants": ["rs4680"],
        "related_pathways": ["Neurotransmitter metabolism", "Stress response"]
    },
    "GC": {
        "description": "Group-specific component (vitamin D binding protein)",
        "nutrients": ["VIT_D"],
        "variants": ["rs2282679", "rs7041", "rs4588"],
        "related_pathways": ["Vitamin D transport", "Immune function"]
    },
    "TCN2": {
        "description": "Transcobalamin 2, vitamin B12 binding protein",
        "nutrients": ["VIT_B12"],
        "variants": ["rs1801198"],
        "related_pathways": ["Vitamin B12 transport", "One-carbon metabolism"]
    }
}

# SNP effect mappings for nutrition analysis
SNP_EFFECT_MAPPINGS = {
    "rs1801133": {  # MTHFR C677T
        "gene": "MTHFR",
        "chromosome": "1",
        "position": 11856378,
        "genotypes": [
            {
                "genotype": "CC",
                "nutrient_effects": [
                    {
                        "nutrient_id": "FOLATE",
                        "effect_type": "normal",
                        "magnitude": 0.0,
                        "confidence": 0.9
                    },
                    {
                        "nutrient_id": "VIT_B12",
                        "effect_type": "normal",
                        "magnitude": 0.0,
                        "confidence": 0.9
                    }
                ]
            },
            {
                "genotype": "CT",
                "nutrient_effects": [
                    {
                        "nutrient_id": "FOLATE",
                        "effect_type": "increased_need",
                        "magnitude": 0.5,
                        "confidence": 0.8
                    },
                    {
                        "nutrient_id": "VIT_B12",
                        "effect_type": "increased_need",
                        "magnitude": 0.3,
                        "confidence": 0.7
                    }
                ]
            },
            {
                "genotype": "TT",
                "nutrient_effects": [
                    {
                        "nutrient_id": "FOLATE",
                        "effect_type": "increased_need",
                        "magnitude": 0.8,
                        "confidence": 0.9
                    },
                    {
                        "nutrient_id": "VIT_B12",
                        "effect_type": "increased_need",
                        "magnitude": 0.6,
                        "confidence": 0.8
                    }
                ],
                "dietary_responses": [
                    {
                        "diet_factor": "low_processed_food",
                        "response_type": "positive",
                        "magnitude": 0.7,
                        "confidence": 0.8
                    }
                ]
            }
        ]
    },
    "rs1801131": {  # MTHFR A1298C
        "gene": "MTHFR",
        "chromosome": "1",
        "position": 11854476,
        "genotypes": [
            {
                "genotype": "AA",
                "nutrient_effects": [
                    {
                        "nutrient_id": "FOLATE",
                        "effect_type": "normal",
                        "magnitude": 0.0,
                        "confidence": 0.9
                    }
                ]
            },
            {
                "genotype": "AC",
                "nutrient_effects": [
                    {
                        "nutrient_id": "FOLATE",
                        "effect_type": "increased_need",
                        "magnitude": 0.3,
                        "confidence": 0.7
                    }
                ]
            },
            {
                "genotype": "CC",
                "nutrient_effects": [
                    {
                        "nutrient_id": "FOLATE",
                        "effect_type": "increased_need",
                        "magnitude": 0.5,
                        "confidence": 0.8
                    }
                ]
            }
        ]
    },
    "rs4988235": {  # MCM6/LCT (lactase persistence)
        "gene": "MCM6/LCT",
        "chromosome": "2",
        "position": 136608646,
        "genotypes": [
            {
                "genotype": "GG",
                "food_sensitivities": [
                    {
                        "food_id": "LACTOSE",
                        "severity": "moderate",
                        "confidence": 0.8
                    }
                ]
            },
            {
                "genotype": "GA",
                "food_sensitivities": [
                    {
                        "food_id": "LACTOSE",
                        "severity": "mild",
                        "confidence": 0.7
                    }
                ]
            },
            {
                "genotype": "AA",
                "food_sensitivities": [
                    {
                        "food_id": "LACTOSE",
                        "severity": "none",
                        "confidence": 0.9
                    }
                ]
            }
        ]
    },
    "rs2282679": {  # GC (Vitamin D binding protein)
        "gene": "GC",
        "chromosome": "4",
        "position": 72618334,
        "genotypes": [
            {
                "genotype": "TT",
                "nutrient_effects": [
                    {
                        "nutrient_id": "VIT_D",
                        "effect_type": "normal",
                        "magnitude": 0.0,
                        "confidence": 0.9
                    }
                ]
            },
            {
                "genotype": "GT",
                "nutrient_effects": [
                    {
                        "nutrient_id": "VIT_D",
                        "effect_type": "increased_need",
                        "magnitude": 0.4,
                        "confidence": 0.7
                    }
                ]
            },
            {
                "genotype": "GG",
                "nutrient_effects": [
                    {
                        "nutrient_id": "VIT_D",
                        "effect_type": "increased_need",
                        "magnitude": 0.7,
                        "confidence": 0.8
                    }
                ],
                "dietary_responses": [
                    {
                        "diet_factor": "sun_exposure",
                        "response_type": "positive",
                        "magnitude": 0.8,
                        "confidence": 0.7
                    }
                ]
            }
        ]
    },
    "rs1800562": {  # HFE C282Y (hemochromatosis)
        "gene": "HFE",
        "chromosome": "6",
        "position": 26093141,
        "genotypes": [
            {
                "genotype": "GG",
                "nutrient_effects": [
                    {
                        "nutrient_id": "IRON",
                        "effect_type": "normal",
                        "magnitude": 0.0,
                        "confidence": 0.9
                    }
                ]
            },
            {
                "genotype": "GA",
                "nutrient_effects": [
                    {
                        "nutrient_id": "IRON",
                        "effect_type": "decreased_need",
                        "magnitude": 0.4,
                        "confidence": 0.7
                    }
                ],
                "dietary_responses": [
                    {
                        "diet_factor": "low_iron",
                        "response_type": "positive",
                        "magnitude": 0.5,
                        "confidence": 0.7
                    }
                ]
            },
            {
                "genotype": "AA",
                "nutrient_effects": [
                    {
                        "nutrient_id": "IRON",
                        "effect_type": "decreased_need",
                        "magnitude": 0.9,
                        "confidence": 0.9
                    },
                    {
                        "nutrient_id": "VIT_C",
                        "effect_type": "decreased_need",
                        "magnitude": 0.5,
                        "confidence": 0.6
                    }
                ],
                "dietary_responses": [
                    {
                        "diet_factor": "low_iron",
                        "response_type": "positive",
                        "magnitude": 0.9,
                        "confidence": 0.9
                    },
                    {
                        "diet_factor": "low_alcohol",
                        "response_type": "positive",
                        "magnitude": 0.8,
                        "confidence": 0.8
                    }
                ]
            }
        ]
    },
    "rs1799945": {  # HFE H63D
        "gene": "HFE",
        "chromosome": "6",
        "position": 26091179,
        "genotypes": [
            {
                "genotype": "CC",
                "nutrient_effects": [
                    {
                        "nutrient_id": "IRON",
                        "effect_type": "normal",
                        "magnitude": 0.0,
                        "confidence": 0.9
                    }
                ]
            },
            {
                "genotype": "CG",
                "nutrient_effects": [
                    {
                        "nutrient_id": "IRON",
                        "effect_type": "decreased_need",
                        "magnitude": 0.2,
                        "confidence": 0.6
                    }
                ]
            },
            {
                "genotype": "GG",
                "nutrient_effects": [
                    {
                        "nutrient_id": "IRON",
                        "effect_type": "decreased_need",
                        "magnitude": 0.5,
                        "confidence": 0.7
                    }
                ],
                "dietary_responses": [
                    {
                        "diet_factor": "low_iron",
                        "response_type": "positive",
                        "magnitude": 0.4,
                        "confidence": 0.6
                    }
                ]
            }
        ]
    },
    "rs4680": {  # COMT Val158Met
        "gene": "COMT",
        "chromosome": "22",
        "position": 19951271,
        "genotypes": [
            {
                "genotype": "GG",  # Val/Val (high activity)
                "nutrient_effects": [
                    {
                        "nutrient_id": "VIT_B6",
                        "effect_type": "normal",
                        "magnitude": 0.0,
                        "confidence": 0.7
                    },
                    {
                        "nutrient_id": "MAGNESIUM",
                        "effect_type": "normal",
                        "magnitude": 0.0,
                        "confidence": 0.7
                    }
                ],
                "dietary_responses": [
                    {
                        "diet_factor": "caffeine",
                        "response_type": "positive",
                        "magnitude": 0.6,
                        "confidence": 0.7
                    },
                    {
                        "diet_factor": "high_protein",
                        "response_type": "positive",
                        "magnitude": 0.5,
                        "confidence": 0.6
                    }
                ]
            },
            {
                "genotype": "AG",  # Val/Met (intermediate)
                "nutrient_effects": [
                    {
                        "nutrient_id": "VIT_B6",
                        "effect_type": "increased_need",
                        "magnitude": 0.3,
                        "confidence": 0.6
                    },
                    {
                        "nutrient_id": "MAGNESIUM",
                        "effect_type": "increased_need",
                        "magnitude": 0.3,
                        "confidence": 0.6
                    }
                ],
                "dietary_responses": [
                    {
                        "diet_factor": "balanced_meals",
                        "response_type": "positive",
                        "magnitude": 0.6,
                        "confidence": 0.7
                    }
                ]
            },
            {
                "genotype": "AA",  # Met/Met (low activity)
                "nutrient_effects": [
                    {
                        "nutrient_id": "VIT_B6",
                        "effect_type": "increased_need",
                        "magnitude": 0.6,
                        "confidence": 0.7
                    },
                    {
                        "nutrient_id": "MAGNESIUM",
                        "effect_type": "increased_need",
                        "magnitude": 0.6,
                        "confidence": 0.7
                    }
                ],
                "food_sensitivities": [
                    {
                        "food_id": "CAFFEINE",
                        "severity": "moderate",
                        "confidence": 0.7
                    }
                ],
                "dietary_responses": [
                    {
                        "diet_factor": "balanced_meals",
                        "response_type": "positive",
                        "magnitude": 0.8,
                        "confidence": 0.8
                    },
                    {
                        "diet_factor": "low_caffeine",
                        "response_type": "positive",
                        "magnitude": 0.7,
                        "confidence": 0.7
                    }
                ]
            }
        ]
    },
    "rs174537": {  # FADS1 (fatty acid desaturase)
        "gene": "FADS1",
        "chromosome": "11",
        "position": 61552680,
        "genotypes": [
            {
                "genotype": "GG",
                "nutrient_effects": [
                    {
                        "nutrient_id": "OMEGA3",
                        "effect_type": "normal",
                        "magnitude": 0.0,
                        "confidence": 0.8
                    }
                ],
                "dietary_responses": [
                    {
                        "diet_factor": "balanced_omega",
                        "response_type": "positive",
                        "magnitude": 0.5,
                        "confidence": 0.7
                    }
                ]
            },
            {
                "genotype": "GT",
                "nutrient_effects": [
                    {
                        "nutrient_id": "OMEGA3",
                        "effect_type": "increased_need",
                        "magnitude": 0.4,
                        "confidence": 0.7
                    }
                ],
                "dietary_responses": [
                    {
                        "diet_factor": "high_omega3",
                        "response_type": "positive",
                        "magnitude": 0.6,
                        "confidence": 0.7
                    }
                ]
            },
            {
                "genotype": "TT",
                "nutrient_effects": [
                    {
                        "nutrient_id": "OMEGA3",
                        "effect_type": "increased_need",
                        "magnitude": 0.7,
                        "confidence": 0.8
                    }
                ],
                "dietary_responses": [
                    {
                        "diet_factor": "high_omega3",
                        "response_type": "positive",
                        "magnitude": 0.8,
                        "confidence": 0.8
                    },
                    {
                        "diet_factor": "mediterranean",
                        "response_type": "positive",
                        "magnitude": 0.7,
                        "confidence": 0.7
                    }
                ]
            }
        ]
    },
    "rs12934922": {  # BCMO1 (beta-carotene conversion)
        "gene": "BCMO1",
        "chromosome": "16",
        "position": 81263094,
        "genotypes": [
            {
                "genotype": "AA",
                "nutrient_effects": [
                    {
                        "nutrient_id": "VIT_A",
                        "effect_type": "normal",
                        "magnitude": 0.0,
                        "confidence": 0.8
                    }
                ]
            },
            {
                "genotype": "AT",
                "nutrient_effects": [
                    {
                        "nutrient_id": "VIT_A",
                        "effect_type": "increased_need",
                        "magnitude": 0.3,
                        "confidence": 0.7
                    }
                ],
                "dietary_responses": [
                    {
                        "diet_factor": "varied_vitamin_a",
                        "response_type": "positive",
                        "magnitude": 0.5,
                        "confidence": 0.7
                    }
                ]
            },
            {
                "genotype": "TT",
                "nutrient_effects": [
                    {
                        "nutrient_id": "VIT_A",
                        "effect_type": "increased_need",
                        "magnitude": 0.6,
                        "confidence": 0.8
                    }
                ],
                "dietary_responses": [
                    {
                        "diet_factor": "varied_vitamin_a",
                        "response_type": "positive",
                        "magnitude": 0.8,
                        "confidence": 0.8
                    }
                ]
            }
        ]
    },
    "rs13266634": {  # SLC30A8 (zinc transporter)
        "gene": "SLC30A8",
        "chromosome": "8",
        "position": 118184783,
        "genotypes": [
            {
                "genotype": "TT",
                "nutrient_effects": [
                    {
                        "nutrient_id": "ZINC",
                        "effect_type": "normal",
                        "magnitude": 0.0,
                        "confidence": 0.7
                    }
                ]
            },
            {
                "genotype": "CT",
                "nutrient_effects": [
                    {
                        "nutrient_id": "ZINC",
                        "effect_type": "increased_need",
                        "magnitude": 0.3,
                        "confidence": 0.6
                    }
                ]
            },
            {
                "genotype": "CC",
                "nutrient_effects": [
                    {
                        "nutrient_id": "ZINC",
                        "effect_type": "increased_need",
                        "magnitude": 0.5,
                        "confidence": 0.7
                    }
                ],
                "dietary_responses": [
                    {
                        "diet_factor": "balanced_meals",
                        "response_type": "positive",
                        "magnitude": 0.6,
                        "confidence": 0.7
                    }
                ]
            }
        ]
    },
    "rs2070895": {  # LIPC (hepatic lipase)
        "gene": "LIPC",
        "chromosome": "15",
        "position": 58855748,
        "genotypes": [
            {
                "genotype": "GG",
                "dietary_responses": [
                    {
                        "diet_factor": "low_carb",
                        "response_type": "positive",
                        "magnitude": 0.6,
                        "confidence": 0.7
                    }
                ]
            },
            {
                "genotype": "GA",
                "dietary_responses": [
                    {
                        "diet_factor": "low_carb",
                        "response_type": "positive",
                        "magnitude": 0.4,
                        "confidence": 0.6
                    },
                    {
                        "diet_factor": "balanced_diet",
                        "response_type": "positive",
                        "magnitude": 0.4,
                        "confidence": 0.6
                    }
                ]
            },
            {
                "genotype": "AA",
                "dietary_responses": [
                    {
                        "diet_factor": "low_fat",
                        "response_type": "positive",
                        "magnitude": 0.6,
                        "confidence": 0.7
                    }
                ]
            }
        ]
    },
    "rs9939609": {  # FTO (fat mass and obesity associated)
        "gene": "FTO",
        "chromosome": "16",
        "position": 53820527,
        "genotypes": [
            {
                "genotype": "TT",
                "dietary_responses": [
                    {
                        "diet_factor": "balanced_diet",
                        "response_type": "positive",
                        "magnitude": 0.5,
                        "confidence": 0.7
                    }
                ]
            },
            {
                "genotype": "AT",
                "dietary_responses": [
                    {
                        "diet_factor": "high_protein",
                        "response_type": "positive",
                        "magnitude": 0.6,
                        "confidence": 0.7
                    },
                    {
                        "diet_factor": "portion_control",
                        "response_type": "positive",
                        "magnitude": 0.7,
                        "confidence": 0.8
                    }
                ]
            },
            {
                "genotype": "AA",
                "dietary_responses": [
                    {
                        "diet_factor": "high_protein",
                        "response_type": "positive",
                        "magnitude": 0.8,
                        "confidence": 0.8
                    },
                    {
                        "diet_factor": "portion_control",
                        "response_type": "positive",
                        "magnitude": 0.9,
                        "confidence": 0.9
                    }
                ]
            }
        ]
    },
    "rs1799883": {  # FABP2 (fatty acid binding protein 2)
        "gene": "FABP2",
        "chromosome": "4",
        "position": 120241902,
        "genotypes": [
            {
                "genotype": "GG",
                "dietary_responses": [
                    {
                        "diet_factor": "balanced_diet",
                        "response_type": "positive",
                        "magnitude": 0.5,
                        "confidence": 0.7
                    }
                ]
            },
            {
                "genotype": "GA",
                "dietary_responses": [
                    {
                        "diet_factor": "low_fat",
                        "response_type": "positive",
                        "magnitude": 0.5,
                        "confidence": 0.6
                    }
                ]
            },
            {
                "genotype": "AA",
                "dietary_responses": [
                    {
                        "diet_factor": "low_fat",
                        "response_type": "positive",
                        "magnitude": 0.7,
                        "confidence": 0.7
                    },
                    {
                        "diet_factor": "mediterranean",
                        "response_type": "positive",
                        "magnitude": 0.6,
                        "confidence": 0.7
                    }
                ]
            }
        ]
    }
} 