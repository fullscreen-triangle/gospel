import React, { useState } from 'react'
import Modal from 'react-modal';

export default function Service({ ActiveIndex }) {
    const [isOpen, setIsOpen] = useState(false);
    const [modalContent, setModalContent] = useState({});

    function toggleModal() {
        setIsOpen(!isOpen);
    }

    const features = [
        {
            title: "Multi-Domain Variant Analysis",
            text: "Analyze genomic variants across fitness, pharmacogenetics, and nutrition domains with integrated scoring.",
            code: `from gospel.core import GospelAnalyzer

analyzer = GospelAnalyzer()
results = analyzer.analyze_variant(
    gene="ACTN3",
    variant="R577X",
    domains=["fitness", "pharmacogenetics"]
)
print(results.score)
print(results.interpretation)`,
            detail: "The variant analysis engine processes VCF files and annotates variants with multi-domain significance scores. Each variant is evaluated against curated gene-domain mappings covering key genes including ACTN3, PPARGC1A, ACE (fitness), CYP3A4, CYP2C19, CYP2D6 (pharmacogenetics), and MCM6, HLA-DQ (nutrition)."
        },
        {
            title: "Metacognitive Orchestration",
            text: "Autonomous tool selection and analysis pipeline construction through Bayesian decision-making.",
            code: `from gospel.core import GospelAnalyzer

analyzer = GospelAnalyzer(
    metacognitive=True,
    confidence_threshold=0.95
)

# The analyzer autonomously selects tools,
# constructs pipelines, and validates results
report = analyzer.full_analysis(
    vcf_path="sample.vcf",
    reference="hg38"
)`,
            detail: "The metacognitive engine uses variational Bayesian inference to autonomously select analysis tools, construct processing pipelines, and validate results. It maintains a belief state over analysis quality and iteratively refines its approach until confidence thresholds are met."
        },
        {
            title: "Fuzzy-Bayesian Uncertainty",
            text: "Continuous uncertainty quantification combining fuzzy logic with Bayesian probabilistic reasoning.",
            code: `from gospel.core.fuzzy_system import FuzzySystem

fuzzy = FuzzySystem()
membership = fuzzy.evaluate(
    variant_frequency=0.03,
    conservation_score=0.89,
    functional_impact="moderate"
)
# Returns continuous membership values
# across pathogenicity categories`,
            detail: "The fuzzy-Bayesian system provides continuous uncertainty quantification for genomic variant interpretation. Rather than binary pathogenic/benign classifications, it maintains membership functions across the full spectrum of clinical significance, enabling nuanced interpretation of variants of uncertain significance (VUS)."
        },
        {
            title: "S-Entropy Navigation",
            text: "Navigate genomic solution space using three-dimensional S-entropy coordinates instead of sequential search.",
            code: `from gospel.core import SEntropyNavigator

nav = SEntropyNavigator(
    partition_depth=24
)
# Navigate to solution in O(log3 n)
position = nav.navigate(
    S_k=0.85,  # knowledge dimension
    S_t=0.42,  # temporal dimension
    S_e=0.67   # evolution dimension
)
features = nav.extract_features(position)`,
            detail: "S-entropy navigation replaces sequential genomic search with direct coordinate-based navigation. Each position in the three-dimensional S-space (knowledge, temporal, evolution) encodes a unique genomic feature. Navigation achieves O(log_3 n) complexity, enabling whole-genome analysis in logarithmic time."
        },
        {
            title: "Network & Pathway Analysis",
            text: "Gene interaction networks, pathway discovery, and systems biology integration.",
            code: `from gospel.network import PathwayAnalyzer

pathway = PathwayAnalyzer()
network = pathway.build_network(
    genes=["BRCA1", "TP53", "EGFR"],
    interaction_type="functional"
)
communities = network.detect_communities()
enrichment = pathway.enrichment_analysis(
    communities[0]
)`,
            detail: "The network analysis module builds gene interaction graphs, detects functional communities using graph algorithms, and performs pathway enrichment analysis. It integrates with external databases and supports both protein-protein interaction networks and regulatory networks."
        },
        {
            title: "CLI & Interactive Query",
            text: "Command-line interface with interactive LLM-powered querying for natural language genomic analysis.",
            code: `# Command-line analysis
$ gospel analyze --vcf sample.vcf \\
    --domains fitness,pharma \\
    --output report.html

# Interactive query mode
$ gospel query
> What is the impact of CYP2D6 *4
  on codeine metabolism?

# Visualization
$ gospel visualize --vcf sample.vcf \\
    --type circuit-diagram`,
            detail: "The CLI provides three primary commands: analyze (batch variant analysis), query (interactive natural language querying via LLM integration), and visualize (genomic circuit diagrams, network plots, heatmaps). Output formats include HTML reports, JSON data, and SVG visualizations."
        }
    ];

    return (
        <>
            <div className={ActiveIndex === 7 ? "cavani_tm_section active animated fadeInUp" : "cavani_tm_section hidden animated"} id="framework_">
                <div className="section_inner">
                    <div className="cavani_tm_about">
                        <div className="cavani_tm_title">
                            <span>Gospel Framework</span>
                        </div>

                        <div className="paper_section">
                            <h3 className="paper_section_title">Installation</h3>
                            <p>Gospel is a Python package with an optional high-performance Rust core. Install it locally to access the full genomic analysis framework.</p>
                            <div className="code_block">
                                <pre><code>{`# Install the Python framework
pip install gospel

# Or install from source with Rust core
git clone https://github.com/gospel-framework/gospel.git
cd gospel
pip install -e .

# Build the Rust core (optional, for maximum performance)
cd core
cargo build --release --all-features

# Verify installation
gospel --version
gospel analyze --help`}</code></pre>
                            </div>
                        </div>

                        <div className="paper_section">
                            <h3 className="paper_section_title">Requirements</h3>
                            <div className="requirements_grid">
                                <div className="req_card">
                                    <h4>Python</h4>
                                    <p>Python 3.9+ with torch, biopython, scipy, pandas, scikit-learn, networkx</p>
                                </div>
                                <div className="req_card">
                                    <h4>Rust (Optional)</h4>
                                    <p>Rust 1.70+ with tokio, ndarray, nalgebra, rayon for concurrent processing</p>
                                </div>
                                <div className="req_card">
                                    <h4>Data</h4>
                                    <p>VCF files, reference genomes (hg38/hg19), variant databases</p>
                                </div>
                            </div>
                        </div>

                        <div className="paper_section">
                            <h3 className="paper_section_title">Capabilities</h3>
                            <div className="cavani_tm_service">
                                <div className="service_list">
                                    <ul>
                                        {features.map((item, i) => (
                                            <li key={i}>
                                                <div className="list_inner" onClick={() => { setModalContent(item); toggleModal(); }}>
                                                    <h3 className="title">{item.title}</h3>
                                                    <p className="text">{item.text}</p>
                                                    <div className="code_block code_preview">
                                                        <pre><code>{item.code}</code></pre>
                                                    </div>
                                                </div>
                                            </li>
                                        ))}
                                    </ul>
                                </div>
                            </div>
                        </div>

                        <div className="paper_section">
                            <h3 className="paper_section_title">Architecture</h3>
                            <div className="architecture_overview">
                                <div className="arch_layer">
                                    <div className="arch_label">CLI Layer</div>
                                    <div className="arch_content">gospel analyze | gospel query | gospel visualize</div>
                                </div>
                                <div className="arch_arrow">&darr;</div>
                                <div className="arch_layer">
                                    <div className="arch_label">Core Engine</div>
                                    <div className="arch_content">Metacognitive Orchestrator &rarr; Fuzzy-Bayesian Reasoner &rarr; Variant Scorer</div>
                                </div>
                                <div className="arch_arrow">&darr;</div>
                                <div className="arch_layer">
                                    <div className="arch_label">Domain Modules</div>
                                    <div className="arch_content">Fitness | Pharmacogenetics | Nutrition</div>
                                </div>
                                <div className="arch_arrow">&darr;</div>
                                <div className="arch_layer">
                                    <div className="arch_label">Rust Core (Optional)</div>
                                    <div className="arch_content">S-Entropy Nav | Oscillatory Dynamics | Concurrent Processing</div>
                                </div>
                            </div>
                        </div>

                    </div>
                </div>
            </div>

            {modalContent && (
                <Modal
                    isOpen={isOpen}
                    onRequestClose={toggleModal}
                    contentLabel="Feature detail"
                    className="mymodal"
                    overlayClassName="myoverlay"
                    closeTimeoutMS={300}
                    openTimeoutMS={300}
                >
                    <div className="cavani_tm_modalbox opened">
                        <div className="box_inner">
                            <div className="close" onClick={toggleModal}>
                                <a href="#"><i className="icon-cancel"></i></a>
                            </div>
                            <div className="description_wrap">
                                <div className="service_popup_informations">
                                    <div className="details">
                                        <h3>{modalContent.title}</h3>
                                    </div>
                                    <div className="descriptions">
                                        <p>{modalContent.detail}</p>
                                        <div className="code_block">
                                            <pre><code>{modalContent.code}</code></pre>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </Modal>
            )}
        </>
    )
}
