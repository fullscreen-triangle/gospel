import React, { useEffect, useRef } from 'react';

function ComplexityChart() {
    const ref = useRef(null);
    useEffect(() => {
        if (!ref.current) return;
        const d3 = require('d3');
        const container = ref.current;
        container.innerHTML = '';
        const margin = { top: 30, right: 120, bottom: 50, left: 70 };
        const width = container.clientWidth - margin.left - margin.right || 500;
        const height = 300 - margin.top - margin.bottom;

        const svg = d3.select(container).append('svg')
            .attr('width', width + margin.left + margin.right)
            .attr('height', height + margin.top + margin.bottom)
            .append('g')
            .attr('transform', `translate(${margin.left},${margin.top})`);

        const nValues = [100, 1000, 10000, 100000, 1e6, 1e7, 1e8];
        const sequential = nValues.map(n => ({ n, ops: n * n }));
        const ternary = nValues.map(n => ({ n, ops: Math.log(n) / Math.log(3) }));
        const binary = nValues.map(n => ({ n, ops: n * Math.log2(n) }));

        const x = d3.scaleLog().domain([100, 1e8]).range([0, width]);
        const y = d3.scaleLog().domain([1, 1e16]).range([height, 0]);

        svg.append('g').attr('transform', `translate(0,${height})`)
            .call(d3.axisBottom(x).ticks(5, '.0e'))
            .selectAll('text').style('fill', '#b0adb7').style('font-size', '11px');
        svg.append('g')
            .call(d3.axisLeft(y).ticks(5, '.0e'))
            .selectAll('text').style('fill', '#b0adb7').style('font-size', '11px');

        svg.selectAll('.domain, .tick line').style('stroke', '#444');

        const line = d3.line().x(d => x(d.n)).y(d => y(d.ops));

        // Sequential O(n^2)
        svg.append('path').datum(sequential)
            .attr('fill', 'none').attr('stroke', '#ff6b6b').attr('stroke-width', 2).attr('stroke-dasharray', '8,4')
            .attr('d', line);
        // Binary O(n log n)
        svg.append('path').datum(binary)
            .attr('fill', 'none').attr('stroke', '#4ecdc4').attr('stroke-width', 2).attr('stroke-dasharray', '4,4')
            .attr('d', line);
        // Ternary O(log3 n)
        svg.append('path').datum(ternary)
            .attr('fill', 'none').attr('stroke', '#f9d77e').attr('stroke-width', 3)
            .attr('d', line);

        // Legend
        const legend = svg.append('g').attr('transform', `translate(${width + 10}, 10)`);
        [{ label: 'O(n\u00B2) Sequential', color: '#ff6b6b', dash: '8,4' },
         { label: 'O(n log n) Binary', color: '#4ecdc4', dash: '4,4' },
         { label: 'O(log\u2083 n) Ternary', color: '#f9d77e', dash: '' }].forEach((item, i) => {
            legend.append('line')
                .attr('x1', 0).attr('x2', 20).attr('y1', i * 22).attr('y2', i * 22)
                .attr('stroke', item.color).attr('stroke-width', 2)
                .attr('stroke-dasharray', item.dash);
            legend.append('text')
                .attr('x', 25).attr('y', i * 22 + 4)
                .text(item.label)
                .style('fill', '#b0adb7').style('font-size', '10px');
        });

        svg.append('text').attr('x', width / 2).attr('y', height + 42)
            .text('Input Size (n)').style('fill', '#b0adb7').style('font-size', '12px').attr('text-anchor', 'middle');
        svg.append('text').attr('transform', 'rotate(-90)').attr('x', -height / 2).attr('y', -55)
            .text('Operations').style('fill', '#b0adb7').style('font-size', '12px').attr('text-anchor', 'middle');
    }, []);
    return <div ref={ref} className="d3_chart_container" />;
}

function ResolutionChart() {
    const ref = useRef(null);
    useEffect(() => {
        if (!ref.current) return;
        const d3 = require('d3');
        const container = ref.current;
        container.innerHTML = '';
        const margin = { top: 30, right: 30, bottom: 50, left: 70 };
        const width = container.clientWidth - margin.left - margin.right || 500;
        const height = 300 - margin.top - margin.bottom;

        const svg = d3.select(container).append('svg')
            .attr('width', width + margin.left + margin.right)
            .attr('height', height + margin.top + margin.bottom)
            .append('g')
            .attr('transform', `translate(${margin.left},${margin.top})`);

        // Resolution scaling: epsilon_k = 3^(-k/3)
        const data = [
            { k: 6, cells: 729, resolution: 0.11 },
            { k: 12, cells: 531441, resolution: 0.012 },
            { k: 18, cells: 387e6, resolution: 0.0014 },
            { k: 24, cells: 282e9, resolution: 0.00016 },
            { k: 30, cells: 206e12, resolution: 1.8e-5 },
        ];

        const x = d3.scaleLinear().domain([6, 30]).range([0, width]);
        const y = d3.scaleLog().domain([1e-5, 0.2]).range([height, 0]);

        svg.append('g').attr('transform', `translate(0,${height})`)
            .call(d3.axisBottom(x).ticks(5))
            .selectAll('text').style('fill', '#b0adb7').style('font-size', '11px');
        svg.append('g')
            .call(d3.axisLeft(y).ticks(5, '.0e'))
            .selectAll('text').style('fill', '#b0adb7').style('font-size', '11px');

        svg.selectAll('.domain, .tick line').style('stroke', '#444');

        const line = d3.line().x(d => x(d.k)).y(d => y(d.resolution));
        svg.append('path').datum(data)
            .attr('fill', 'none').attr('stroke', '#f9d77e').attr('stroke-width', 2.5)
            .attr('d', line);

        // Bars for cell count
        const barWidth = 20;
        const yBar = d3.scaleLog().domain([100, 1e13]).range([height, 0]);
        data.forEach(d => {
            svg.append('rect')
                .attr('x', x(d.k) - barWidth / 2)
                .attr('y', yBar(d.cells))
                .attr('width', barWidth)
                .attr('height', height - yBar(d.cells))
                .attr('fill', 'rgba(78, 205, 196, 0.3)')
                .attr('stroke', '#4ecdc4');
        });

        svg.selectAll('.dot').data(data).enter().append('circle')
            .attr('cx', d => x(d.k)).attr('cy', d => y(d.resolution))
            .attr('r', 5).attr('fill', '#f9d77e');

        svg.append('text').attr('x', width / 2).attr('y', height + 42)
            .text('Trit Depth (k)').style('fill', '#b0adb7').style('font-size', '12px').attr('text-anchor', 'middle');
        svg.append('text').attr('transform', 'rotate(-90)').attr('x', -height / 2).attr('y', -55)
            .text('Resolution (\u03B5)').style('fill', '#b0adb7').style('font-size', '12px').attr('text-anchor', 'middle');
    }, []);
    return <div ref={ref} className="d3_chart_container" />;
}

function CardinalChart() {
    const ref = useRef(null);
    useEffect(() => {
        if (!ref.current) return;
        const d3 = require('d3');
        const container = ref.current;
        container.innerHTML = '';
        const margin = { top: 30, right: 30, bottom: 50, left: 60 };
        const size = Math.min(container.clientWidth - margin.left - margin.right, 350) || 300;

        const svg = d3.select(container).append('svg')
            .attr('width', size + margin.left + margin.right)
            .attr('height', size + margin.top + margin.bottom)
            .append('g')
            .attr('transform', `translate(${margin.left},${margin.top})`);

        const x = d3.scaleLinear().domain([-1.5, 1.5]).range([0, size]);
        const y = d3.scaleLinear().domain([-1.5, 1.5]).range([size, 0]);

        svg.append('g').attr('transform', `translate(0,${size})`)
            .call(d3.axisBottom(x).ticks(5))
            .selectAll('text').style('fill', '#b0adb7').style('font-size', '11px');
        svg.append('g')
            .call(d3.axisLeft(y).ticks(5))
            .selectAll('text').style('fill', '#b0adb7').style('font-size', '11px');
        svg.selectAll('.domain, .tick line').style('stroke', '#444');

        // Axes through origin
        svg.append('line').attr('x1', x(-1.5)).attr('x2', x(1.5)).attr('y1', y(0)).attr('y2', y(0)).attr('stroke', '#333');
        svg.append('line').attr('x1', x(0)).attr('x2', x(0)).attr('y1', y(-1.5)).attr('y2', y(1.5)).attr('stroke', '#333');

        // Cardinal coordinates: A=(0,+1), T=(0,-1), G=(+1,0), C=(-1,0)
        const bases = [
            { label: 'A', x: 0, y: 1, color: '#4ecdc4' },
            { label: 'T', x: 0, y: -1, color: '#ff6b6b' },
            { label: 'G', x: 1, y: 0, color: '#f9d77e' },
            { label: 'C', x: -1, y: 0, color: '#a78bfa' },
        ];

        // Complementary pair lines
        svg.append('line')
            .attr('x1', x(0)).attr('y1', y(1)).attr('x2', x(0)).attr('y2', y(-1))
            .attr('stroke', '#555').attr('stroke-dasharray', '4,4').attr('stroke-width', 1.5);
        svg.append('line')
            .attr('x1', x(-1)).attr('y1', y(0)).attr('x2', x(1)).attr('y2', y(0))
            .attr('stroke', '#555').attr('stroke-dasharray', '4,4').attr('stroke-width', 1.5);

        // Draw nucleotides
        bases.forEach(b => {
            svg.append('circle')
                .attr('cx', x(b.x)).attr('cy', y(b.y))
                .attr('r', 20).attr('fill', b.color).attr('opacity', 0.8);
            svg.append('text')
                .attr('x', x(b.x)).attr('y', y(b.y) + 5)
                .text(b.label)
                .style('fill', '#fff').style('font-size', '16px').style('font-weight', 'bold')
                .attr('text-anchor', 'middle');
        });

        // Labels
        svg.append('text')
            .attr('x', x(0)).attr('y', y(1) - 28)
            .text('A = (0, +1)').style('fill', '#4ecdc4').style('font-size', '11px').attr('text-anchor', 'middle');
        svg.append('text')
            .attr('x', x(0)).attr('y', y(-1) + 38)
            .text('T = (0, \u22121)').style('fill', '#ff6b6b').style('font-size', '11px').attr('text-anchor', 'middle');
        svg.append('text')
            .attr('x', x(1) + 5).attr('y', y(0) - 28)
            .text('G = (+1, 0)').style('fill', '#f9d77e').style('font-size', '11px').attr('text-anchor', 'middle');
        svg.append('text')
            .attr('x', x(-1) - 5).attr('y', y(0) - 28)
            .text('C = (\u22121, 0)').style('fill', '#a78bfa').style('font-size', '11px').attr('text-anchor', 'middle');

        // Complementarity labels
        svg.append('text')
            .attr('x', x(0.5)).attr('y', y(0.5))
            .text('A\u2194T pair').style('fill', '#888').style('font-size', '10px');
        svg.append('text')
            .attr('x', x(-0.7)).attr('y', y(-0.3))
            .text('G\u2194C pair').style('fill', '#888').style('font-size', '10px');
    }, []);
    return <div ref={ref} className="d3_chart_container" />;
}

export default function Computing({ ActiveIndex }) {
    return (
        <>
            <div className={ActiveIndex === 6 ? "cavani_tm_section active animated fadeInUp" : "cavani_tm_section hidden animated"} id="computing_">
                <div className="section_inner">
                    <div className="cavani_tm_about">

                        <div className="cavani_tm_title">
                            <span>Ternary Partition Computing for Genomic Systems</span>
                        </div>
                        <div className="paper_meta">
                            <p className="paper_author">Kundai Farai Sachikonye &mdash; Technical University of Munich</p>
                        </div>

                        <div className="paper_section">
                            <h3 className="paper_section_title">Abstract</h3>
                            <p>This work introduces a ternary-based computing framework for genomic analysis grounded in partition theory. Solutions exist as predetermined endpoints in partition coordinates; computation becomes navigation rather than calculation. The ternary representation encodes three-dimensional S-entropy space: S = [0,1]<sup>3</sup>, where each trit specifies refinement along knowledge (S<sub>k</sub>), temporal (S<sub>t</sub>), or evolution (S<sub>e</sub>) axes.</p>
                            <p>A trit string encodes both position and trajectory, unifying data and instruction. Three fundamental operations &mdash; project, complete, and compose &mdash; replace Boolean AND, OR, NOT. The architecture achieves O(log<sub>3</sub> n) navigation complexity versus O(n<sup>2</sup>) for sequential analysis. Nucleotides map through the cardinal transformation &phi;: &#123;A,T,G,C&#125; &rarr; &#8477;<sup>2</sup>, where complementarity corresponds to partition inversion.</p>
                        </div>

                        <div className="paper_section">
                            <h3 className="paper_section_title">Complexity Comparison</h3>
                            <p>The ternary partition architecture achieves a fundamental reduction in computational complexity. Where sequential analysis requires O(n<sup>2</sup>) operations and binary approaches achieve O(n log n), the ternary navigation paradigm reduces this to O(log<sub>3</sub> n). For a human genome with n &asymp; 6 &times; 10<sup>9</sup> base pairs, this represents a reduction from &sim;10<sup>19</sup> operations to &sim;20 navigation steps.</p>
                            <ComplexityChart />
                            <p className="chart_caption">Computational complexity comparison. Ternary partition navigation (gold) achieves O(log<sub>3</sub> n), dramatically outperforming both sequential O(n<sup>2</sup>) and binary O(n log n) approaches.</p>
                        </div>

                        <div className="paper_section">
                            <h3 className="paper_section_title">Trit Resolution Scaling</h3>
                            <p>Each additional trit in the trit string refines the position in S-entropy space by a factor of 3. A single trit carries log<sub>2</sub>(3) &asymp; 1.585 bits of information, providing 58.5% more information per symbol than a binary bit. A tryte (6 trits) encodes 3<sup>6</sup> = 729 distinct S-space cells, compared to 2<sup>8</sup> = 256 for a binary byte.</p>
                            <ResolutionChart />
                            <p className="chart_caption">Resolution scaling with trit depth. Gold line shows resolution &epsilon;<sub>k</sub> = 3<sup>&minus;k/3</sup>; teal bars show the number of addressable S-space cells. At k = 30, over 200 trillion cells are addressable with resolution &sim;10<sup>&minus;5</sup>.</p>
                        </div>

                        <div className="paper_section">
                            <h3 className="paper_section_title">Cardinal Coordinate Transformation</h3>
                            <p>The cardinal transformation &phi; maps the four nucleotides to coordinates in &#8477;<sup>2</sup>. Complementary pairs are mapped to negations: A &harr; T corresponds to (0,+1) &harr; (0,&minus;1) and G &harr; C to (+1,0) &harr; (&minus;1,0). This mapping preserves the algebraic structure of Watson-Crick complementarity as partition inversion.</p>
                            <CardinalChart />
                            <p className="chart_caption">Cardinal coordinate transformation &phi;: &#123;A,T,G,C&#125; &rarr; &#8477;<sup>2</sup>. Complementary pairs are geometric negations, encoding Watson-Crick pairing as partition inversion.</p>
                        </div>

                        <div className="paper_section">
                            <h3 className="paper_section_title">Key Results</h3>
                            <div className="key_results_grid">
                                <div className="result_card">
                                    <div className="result_value">O(log<sub>3</sub> n)</div>
                                    <div className="result_label">Navigation Complexity</div>
                                </div>
                                <div className="result_card">
                                    <div className="result_value">1.585 bits</div>
                                    <div className="result_label">Information per Trit</div>
                                </div>
                                <div className="result_card">
                                    <div className="result_value">729</div>
                                    <div className="result_label">Tryte Distinct Values (3<sup>6</sup>)</div>
                                </div>
                                <div className="result_card">
                                    <div className="result_value">~101 trits</div>
                                    <div className="result_label">Double-Precision Accuracy</div>
                                </div>
                                <div className="result_card">
                                    <div className="result_value">3</div>
                                    <div className="result_label">Fundamental Operations (project, complete, compose)</div>
                                </div>
                                <div className="result_card">
                                    <div className="result_value">0</div>
                                    <div className="result_label">Free Parameters</div>
                                </div>
                            </div>
                        </div>

                        <div className="paper_section">
                            <h3 className="paper_section_title">Partition Synthesis Language (PSL)</h3>
                            <p>The framework includes a domain-specific language for expressing partition operations on genomic data. PSL provides typed constructs for genome definition, feature synthesis, and trajectory navigation. Core operations &mdash; project, complete, and compose &mdash; map directly to the mathematical framework, enabling formal verification of genomic analyses.</p>
                            <div className="code_block">
                                <pre><code>{`genome hg38 {
  source: "reference/hg38.fa"
  partition_depth: 24
}

feature promoter_region = synthesize {
  navigate: S_k > 0.8
  filter:   S_t in [0.2, 0.6]
  complete: from trajectory
}

analysis = compose(
  project(hg38, promoter_region),
  complete(regulatory_network)
)`}</code></pre>
                            </div>
                        </div>

                        <div className="paper_section">
                            <h3 className="paper_section_title">Navigation vs. Computation</h3>
                            <p>The fundamental paradigm shift introduced by this work is that genomic analysis is navigation, not computation. Solutions exist as predetermined endpoints in partition coordinates; the algorithm navigates to them rather than computing them. This is analogous to looking up a location on a map versus calculating its coordinates from scratch. The Empty Dictionary Principle states that a correctly navigated position in S-space contains the solution without requiring additional stored information.</p>
                        </div>

                    </div>
                </div>
            </div>
        </>
    );
}
