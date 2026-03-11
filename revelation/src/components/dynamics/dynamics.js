import React, { useEffect, useRef } from 'react';

function CapacitanceChart() {
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

        // DNA capacitance vs length (bp) — derived from cylindrical capacitor model
        // C = 2*pi*epsilon_0*L / ln(b/a), scaling linearly with length
        const data = [];
        for (let bp = 1000; bp <= 6e9; bp *= 3) {
            // Linear scaling: human genome (6e9 bp) ~ 300 pF
            const C_pF = (bp / 6e9) * 300;
            data.push({ bp, C_pF });
        }
        data.push({ bp: 6e9, C_pF: 300 });

        const x = d3.scaleLog().domain([1000, 6e9]).range([0, width]);
        const y = d3.scaleLog().domain([5e-5, 500]).range([height, 0]);

        svg.append('g').attr('transform', `translate(0,${height})`)
            .call(d3.axisBottom(x).ticks(5, '.0e'))
            .selectAll('text').style('fill', '#b0adb7').style('font-size', '11px');
        svg.append('g')
            .call(d3.axisLeft(y).ticks(5, '.1e'))
            .selectAll('text').style('fill', '#b0adb7').style('font-size', '11px');

        svg.selectAll('.domain, .tick line').style('stroke', '#444');

        const line = d3.line()
            .x(d => x(d.bp))
            .y(d => y(d.C_pF));

        svg.append('path').datum(data)
            .attr('fill', 'none').attr('stroke', '#f9d77e').attr('stroke-width', 2.5)
            .attr('d', line);

        svg.selectAll('circle').data(data).enter().append('circle')
            .attr('cx', d => x(d.bp)).attr('cy', d => y(d.C_pF))
            .attr('r', 4).attr('fill', '#f9d77e');

        // Highlight human genome point
        svg.append('circle')
            .attr('cx', x(6e9)).attr('cy', y(300))
            .attr('r', 8).attr('fill', 'none').attr('stroke', '#ff6b6b').attr('stroke-width', 2);
        svg.append('text')
            .attr('x', x(6e9) - 10).attr('y', y(300) - 15)
            .text('Human Genome: 300 pF')
            .style('fill', '#ff6b6b').style('font-size', '11px').attr('text-anchor', 'end');

        // Axis labels
        svg.append('text').attr('x', width / 2).attr('y', height + 42)
            .text('DNA Length (base pairs)').style('fill', '#b0adb7').style('font-size', '12px').attr('text-anchor', 'middle');
        svg.append('text').attr('transform', 'rotate(-90)').attr('x', -height / 2).attr('y', -55)
            .text('Capacitance (pF)').style('fill', '#b0adb7').style('font-size', '12px').attr('text-anchor', 'middle');
    }, []);
    return <div ref={ref} className="d3_chart_container" />;
}

function DischargeChart() {
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

        // RC discharge: V(t) = V0 * exp(-t/tau_RC), tau_RC ~ 30 ms
        const tau = 30; // ms
        const data = [];
        for (let t = 0; t <= 150; t += 1) {
            data.push({ t, V: Math.exp(-t / tau) });
        }

        const x = d3.scaleLinear().domain([0, 150]).range([0, width]);
        const y = d3.scaleLinear().domain([0, 1]).range([height, 0]);

        svg.append('g').attr('transform', `translate(0,${height})`)
            .call(d3.axisBottom(x).ticks(6))
            .selectAll('text').style('fill', '#b0adb7').style('font-size', '11px');
        svg.append('g')
            .call(d3.axisLeft(y).ticks(5).tickFormat(d3.format('.1f')))
            .selectAll('text').style('fill', '#b0adb7').style('font-size', '11px');

        svg.selectAll('.domain, .tick line').style('stroke', '#444');

        const area = d3.area()
            .x(d => x(d.t)).y0(height).y1(d => y(d.V));

        svg.append('path').datum(data)
            .attr('fill', 'rgba(249, 215, 126, 0.15)')
            .attr('d', area);

        const line = d3.line().x(d => x(d.t)).y(d => y(d.V));
        svg.append('path').datum(data)
            .attr('fill', 'none').attr('stroke', '#f9d77e').attr('stroke-width', 2.5)
            .attr('d', line);

        // tau marker
        svg.append('line')
            .attr('x1', x(30)).attr('x2', x(30))
            .attr('y1', 0).attr('y2', height)
            .attr('stroke', '#ff6b6b').attr('stroke-dasharray', '5,5');
        svg.append('text')
            .attr('x', x(30) + 5).attr('y', 20)
            .text('\u03C4_RC = 30 ms')
            .style('fill', '#ff6b6b').style('font-size', '11px');

        svg.append('text').attr('x', width / 2).attr('y', height + 42)
            .text('Time (ms)').style('fill', '#b0adb7').style('font-size', '12px').attr('text-anchor', 'middle');
        svg.append('text').attr('transform', 'rotate(-90)').attr('x', -height / 2).attr('y', -55)
            .text('V/V\u2080 (normalized)').style('fill', '#b0adb7').style('font-size', '12px').attr('text-anchor', 'middle');
    }, []);
    return <div ref={ref} className="d3_chart_container" />;
}

function OscillationChart() {
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

        // Phase-locked H-bond oscillations across base pairs
        // Coherence length ~ 25 bp, frequency ~ 10^13 Hz
        const bpCount = 50;
        const coherenceLength = 25;
        const data = [];
        for (let bp = 0; bp < bpCount; bp++) {
            const phase = (bp / coherenceLength) * 2 * Math.PI;
            const offset = (bp % coherenceLength) - coherenceLength / 2;
            const sigma = coherenceLength / 4;
            const amplitude = Math.exp(-(offset * offset) / (2 * sigma * sigma));
            data.push({ bp: bp + 1, amplitude: Math.cos(phase) * amplitude });
        }

        const x = d3.scaleLinear().domain([1, bpCount]).range([0, width]);
        const y = d3.scaleLinear().domain([-1, 1]).range([height, 0]);

        svg.append('g').attr('transform', `translate(0,${height})`)
            .call(d3.axisBottom(x).ticks(10))
            .selectAll('text').style('fill', '#b0adb7').style('font-size', '11px');
        svg.append('g')
            .call(d3.axisLeft(y).ticks(5))
            .selectAll('text').style('fill', '#b0adb7').style('font-size', '11px');

        svg.selectAll('.domain, .tick line').style('stroke', '#444');

        // Zero line
        svg.append('line')
            .attr('x1', 0).attr('x2', width)
            .attr('y1', y(0)).attr('y2', y(0))
            .attr('stroke', '#555').attr('stroke-dasharray', '3,3');

        // Coherence boundary
        svg.append('line')
            .attr('x1', x(25)).attr('x2', x(25))
            .attr('y1', 0).attr('y2', height)
            .attr('stroke', '#4ecdc4').attr('stroke-dasharray', '5,5');
        svg.append('text')
            .attr('x', x(25) + 5).attr('y', 15)
            .text('Coherence boundary (~25 bp)')
            .style('fill', '#4ecdc4').style('font-size', '10px');

        const line = d3.line().x(d => x(d.bp)).y(d => y(d.amplitude)).curve(d3.curveBasis);
        svg.append('path').datum(data)
            .attr('fill', 'none').attr('stroke', '#f9d77e').attr('stroke-width', 2)
            .attr('d', line);

        svg.selectAll('.dot').data(data).enter().append('circle')
            .attr('cx', d => x(d.bp)).attr('cy', d => y(d.amplitude))
            .attr('r', 2.5).attr('fill', '#f9d77e');

        svg.append('text').attr('x', width / 2).attr('y', height + 42)
            .text('Base Pair Position').style('fill', '#b0adb7').style('font-size', '12px').attr('text-anchor', 'middle');
        svg.append('text').attr('transform', 'rotate(-90)').attr('x', -height / 2).attr('y', -55)
            .text('Proton Displacement (normalized)').style('fill', '#b0adb7').style('font-size', '12px').attr('text-anchor', 'middle');
    }, []);
    return <div ref={ref} className="d3_chart_container" />;
}

export default function Dynamics({ ActiveIndex }) {
    return (
        <>
            <div className={ActiveIndex === 5 ? "cavani_tm_section active animated fadeInUp" : "cavani_tm_section hidden animated"} id="dynamics_">
                <div className="section_inner">
                    <div className="cavani_tm_about">

                        <div className="cavani_tm_title">
                            <span>Temporal Charge Dynamics in Nucleic Acids</span>
                        </div>
                        <div className="paper_meta">
                            <p className="paper_author">Kundai Farai Sachikonye &mdash; Technical University of Munich</p>
                        </div>

                        <div className="paper_section">
                            <h3 className="paper_section_title">Abstract</h3>
                            <p>This work establishes that nucleic acids function as deterministic charge oscillators with predictable temporal dynamics. Starting from partition theory, where hydrogen-bond protons occupy categorical binary states (donor or acceptor) at each instant, we derive that DNA exhibits capacitive charge storage with capacitance C &asymp; 300 pF and predictable discharge dynamics with time constant &tau;<sub>RC</sub> &asymp; 30 ms.</p>
                            <p>Experimental validation achieves a 4.27 &times; 10<sup>5</sup>-fold improvement over Heisenberg-limited measurement, with proton trajectory determinism reaching a relative standard deviation of 4.67 &times; 10<sup>&minus;7</sup>. The triple equivalence T<sub>osc</sub> = 2&pi; T<sub>cat</sub> is validated to a precision of 2.8 &times; 10<sup>&minus;16</sup>. Watson-Crick base pairing is shown to maintain charge balance through partition symmetry.</p>
                        </div>

                        <div className="paper_section">
                            <h3 className="paper_section_title">DNA as a Charge Capacitor</h3>
                            <p>The structural basis for capacitive charge storage in DNA arises from the phosphodiester backbone, which carries approximately one negative charge per nucleotide. For the human genome containing approximately 6 &times; 10<sup>9</sup> base pairs, this yields a linear charge density exceeding 1 e/nm. When modeled as a cylindrical capacitor with inner radius &sim;1 nm (helix core) and outer radius &sim;10 nm (counterion cloud), the resulting capacitance scales linearly with DNA length.</p>
                            <CapacitanceChart />
                            <p className="chart_caption">DNA capacitance as a function of length. The human genome (&sim;6 &times; 10<sup>9</sup> bp) yields approximately 300 pF, comparable to discrete electronic capacitors.</p>
                        </div>

                        <div className="paper_section">
                            <h3 className="paper_section_title">Discharge Dynamics</h3>
                            <p>DNA discharge follows exponential RC dynamics with a characteristic time constant &tau;<sub>RC</sub> &asymp; 30 ms. This timescale emerges from the product of the genomic capacitance (C &asymp; 300 pF) and the effective resistance of the ionic environment. The discharge curve V(t) = V<sub>0</sub> exp(&minus;t/&tau;<sub>RC</sub>) provides a deterministic framework for understanding charge redistribution within the genome.</p>
                            <DischargeChart />
                            <p className="chart_caption">Normalized discharge dynamics of genomic DNA. The RC time constant &tau;<sub>RC</sub> &asymp; 30 ms governs the rate of charge redistribution.</p>
                        </div>

                        <div className="paper_section">
                            <h3 className="paper_section_title">Phase-Locked H-Bond Networks</h3>
                            <p>Hydrogen-bond protons in DNA oscillate at approximately 10<sup>13</sup> Hz within the double-well potential of each base pair. These oscillations become phase-locked across neighboring base pairs through backbone-mediated coupling, establishing coherent charge oscillation domains with a characteristic coherence length of approximately 25 base pairs &mdash; matching the periodicity of nucleosome organization.</p>
                            <OscillationChart />
                            <p className="chart_caption">Phase-locked proton oscillations across base pair positions. Coherence domains of &sim;25 bp correspond to nucleosome-scale organization.</p>
                        </div>

                        <div className="paper_section">
                            <h3 className="paper_section_title">Key Results</h3>
                            <div className="key_results_grid">
                                <div className="result_card">
                                    <div className="result_value">300 pF</div>
                                    <div className="result_label">DNA Capacitance (human genome)</div>
                                </div>
                                <div className="result_card">
                                    <div className="result_value">30 ms</div>
                                    <div className="result_label">RC Time Constant (&tau;<sub>RC</sub>)</div>
                                </div>
                                <div className="result_card">
                                    <div className="result_value">10<sup>13</sup> Hz</div>
                                    <div className="result_label">Proton Oscillation Frequency</div>
                                </div>
                                <div className="result_card">
                                    <div className="result_value">~25 bp</div>
                                    <div className="result_label">Coherence Length</div>
                                </div>
                                <div className="result_card">
                                    <div className="result_value">4.27 &times; 10<sup>5</sup></div>
                                    <div className="result_label">Measurement Improvement Factor</div>
                                </div>
                                <div className="result_card">
                                    <div className="result_value">2.8 &times; 10<sup>&minus;16</sup></div>
                                    <div className="result_label">Triple Equivalence Precision</div>
                                </div>
                            </div>
                        </div>

                        <div className="paper_section">
                            <h3 className="paper_section_title">Triple Equivalence Theorem</h3>
                            <p>The central theoretical result establishes the equivalence between three independently measurable quantities: the oscillation period T<sub>osc</sub>, the categorical measurement period 2&pi; T<sub>cat</sub>, and the partition period T<sub>part</sub>. This triple equivalence, validated to a precision of 2.8 &times; 10<sup>&minus;16</sup>, demonstrates that oscillation, categorical distinction, and partition operations are fundamentally the same process observed from different perspectives.</p>
                        </div>

                        <div className="paper_section">
                            <h3 className="paper_section_title">Implications</h3>
                            <p>This framework reinterprets DNA as an active charge dynamics system rather than passive information storage. Gene regulation emerges from charge distribution dynamics rather than purely sequence-specific protein-DNA interactions. The C-Value Paradox &mdash; why organisms with similar complexity have vastly different genome sizes &mdash; is resolved: large non-coding DNA regions serve charge storage and regulation functions. Chromatin organization at the nucleosome scale (&sim;25 bp coherence) reflects the underlying physics of phase-locked charge oscillation.</p>
                        </div>

                    </div>
                </div>
            </div>
        </>
    );
}
