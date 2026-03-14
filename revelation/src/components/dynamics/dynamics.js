import React, { useEffect, useRef, useState } from 'react';

function ensureTooltip() {
    let tip = document.querySelector('.d3_tooltip');
    if (!tip) {
        tip = document.createElement('div');
        tip.className = 'd3_tooltip';
        tip.style.display = 'none';
        document.body.appendChild(tip);
    }
    return tip;
}

function CapacitanceChart() {
    const ref = useRef(null);
    useEffect(() => {
        if (!ref.current) return;
        const d3 = require('d3');
        const container = ref.current;
        container.innerHTML = '';
        const margin = { top: 20, right: 20, bottom: 45, left: 60 };
        const width = (container.clientWidth || 400) - margin.left - margin.right;
        const height = 280 - margin.top - margin.bottom;
        const tooltip = ensureTooltip();

        const svg = d3.select(container).append('svg')
            .attr('width', width + margin.left + margin.right)
            .attr('height', height + margin.top + margin.bottom)
            .append('g').attr('transform', `translate(${margin.left},${margin.top})`);

        const data = [];
        for (let bp = 1000; bp <= 6e9; bp *= 3) {
            data.push({ bp, C_pF: (bp / 6e9) * 300 });
        }
        data.push({ bp: 6e9, C_pF: 300 });

        const x = d3.scaleLog().domain([1000, 6e9]).range([0, width]);
        const y = d3.scaleLog().domain([5e-5, 500]).range([height, 0]);

        svg.append('g').attr('transform', `translate(0,${height})`)
            .call(d3.axisBottom(x).ticks(4, '.0e'))
            .selectAll('text').style('fill', '#667').style('font-size', '10px');
        svg.append('g').call(d3.axisLeft(y).ticks(4, '.1e'))
            .selectAll('text').style('fill', '#667').style('font-size', '10px');
        svg.selectAll('.domain, .tick line').style('stroke', '#1a2a3a');

        const line = d3.line().x(d => x(d.bp)).y(d => y(d.C_pF)).curve(d3.curveMonotoneX);

        const path = svg.append('path').datum(data)
            .attr('fill', 'none').attr('stroke', '#f9d77e').attr('stroke-width', 2.5)
            .attr('d', line);
        const totalLength = path.node().getTotalLength();
        path.attr('stroke-dasharray', totalLength).attr('stroke-dashoffset', totalLength)
            .transition().duration(1500).ease(d3.easeCubicOut).attr('stroke-dashoffset', 0);

        svg.selectAll('.dot').data(data).enter().append('circle')
            .attr('cx', d => x(d.bp)).attr('cy', d => y(d.C_pF))
            .attr('r', 0).attr('fill', '#f9d77e').attr('cursor', 'pointer')
            .transition().delay((d, i) => i * 100).duration(400)
            .attr('r', 5);

        svg.selectAll('.dot-hover').data(data).enter().append('circle')
            .attr('cx', d => x(d.bp)).attr('cy', d => y(d.C_pF))
            .attr('r', 15).attr('fill', 'transparent').attr('cursor', 'pointer')
            .on('mouseover', function(event, d) {
                d3.select(this.previousSibling).transition().duration(200).attr('r', 8).attr('fill', '#fff');
                tooltip.innerHTML = '<strong>' + (d.bp >= 1e6 ? (d.bp/1e9).toFixed(1) + ' Gbp' : (d.bp/1e3).toFixed(0) + ' kbp') + '</strong><br/>C = ' + (d.C_pF >= 1 ? d.C_pF.toFixed(1) : d.C_pF.toExponential(1)) + ' pF';
                tooltip.style.display = 'block';
                tooltip.style.left = (event.clientX + 12) + 'px';
                tooltip.style.top = (event.clientY - 10) + 'px';
            })
            .on('mousemove', function(event) {
                tooltip.style.left = (event.clientX + 12) + 'px';
                tooltip.style.top = (event.clientY - 10) + 'px';
            })
            .on('mouseout', function() {
                d3.select(this.previousSibling).transition().duration(200).attr('r', 5).attr('fill', '#f9d77e');
                tooltip.style.display = 'none';
            });

        svg.append('circle').attr('cx', x(6e9)).attr('cy', y(300))
            .attr('r', 0).attr('fill', 'none').attr('stroke', '#ff6b6b').attr('stroke-width', 2)
            .transition().delay(1600).duration(600).attr('r', 12);
        svg.append('text').attr('x', x(6e9) - 15).attr('y', y(300) - 18)
            .text('Human Genome: 300 pF').style('fill', '#ff6b6b').style('font-size', '10px').attr('text-anchor', 'end')
            .style('opacity', 0).transition().delay(1800).duration(400).style('opacity', 1);

        svg.append('text').attr('x', width / 2).attr('y', height + 38)
            .text('DNA Length (base pairs)').style('fill', '#667').style('font-size', '11px').attr('text-anchor', 'middle');
        svg.append('text').attr('transform', 'rotate(-90)').attr('x', -height / 2).attr('y', -48)
            .text('Capacitance (pF)').style('fill', '#667').style('font-size', '11px').attr('text-anchor', 'middle');
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
        const margin = { top: 20, right: 20, bottom: 45, left: 60 };
        const width = (container.clientWidth || 400) - margin.left - margin.right;
        const height = 280 - margin.top - margin.bottom;
        const tooltip = ensureTooltip();

        const svg = d3.select(container).append('svg')
            .attr('width', width + margin.left + margin.right)
            .attr('height', height + margin.top + margin.bottom)
            .append('g').attr('transform', `translate(${margin.left},${margin.top})`);

        const tau = 30;
        const data = [];
        for (let t = 0; t <= 150; t += 1) data.push({ t, V: Math.exp(-t / tau) });

        const x = d3.scaleLinear().domain([0, 150]).range([0, width]);
        const y = d3.scaleLinear().domain([0, 1]).range([height, 0]);

        svg.append('g').attr('transform', `translate(0,${height})`)
            .call(d3.axisBottom(x).ticks(5)).selectAll('text').style('fill', '#667').style('font-size', '10px');
        svg.append('g').call(d3.axisLeft(y).ticks(5).tickFormat(d3.format('.1f')))
            .selectAll('text').style('fill', '#667').style('font-size', '10px');
        svg.selectAll('.domain, .tick line').style('stroke', '#1a2a3a');

        const area = d3.area().x(d => x(d.t)).y0(height).y1(d => y(d.V));
        svg.append('path').datum(data).attr('fill', 'rgba(249, 215, 126, 0.08)').attr('d', area)
            .style('opacity', 0).transition().duration(1000).style('opacity', 1);

        const line = d3.line().x(d => x(d.t)).y(d => y(d.V));
        const path = svg.append('path').datum(data)
            .attr('fill', 'none').attr('stroke', '#f9d77e').attr('stroke-width', 2.5).attr('d', line);
        const totalLength = path.node().getTotalLength();
        path.attr('stroke-dasharray', totalLength).attr('stroke-dashoffset', totalLength)
            .transition().duration(2000).ease(d3.easeLinear).attr('stroke-dashoffset', 0);

        svg.append('line').attr('x1', x(30)).attr('x2', x(30)).attr('y1', 0).attr('y2', height)
            .attr('stroke', '#ff6b6b').attr('stroke-dasharray', '4,4').style('opacity', 0)
            .transition().delay(800).duration(400).style('opacity', 1);
        svg.append('text').attr('x', x(30) + 4).attr('y', 14)
            .text('\u03C4_RC = 30 ms').style('fill', '#ff6b6b').style('font-size', '10px')
            .style('opacity', 0).transition().delay(1000).duration(400).style('opacity', 1);

        const vLine = svg.append('line').attr('y1', 0).attr('y2', height).attr('stroke', '#f9d77e').attr('stroke-width', 0.5).style('opacity', 0);
        const hLine = svg.append('line').attr('x1', 0).attr('x2', width).attr('stroke', '#f9d77e').attr('stroke-width', 0.5).style('opacity', 0);
        const hoverDot = svg.append('circle').attr('r', 4).attr('fill', '#fff').style('opacity', 0);

        svg.append('rect').attr('width', width).attr('height', height).attr('fill', 'transparent')
            .on('mousemove', function(event) {
                const [mx] = d3.pointer(event);
                const t = x.invert(mx);
                const V = Math.exp(-t / tau);
                vLine.attr('x1', mx).attr('x2', mx).style('opacity', 0.5);
                hLine.attr('y1', y(V)).attr('y2', y(V)).style('opacity', 0.5);
                hoverDot.attr('cx', mx).attr('cy', y(V)).style('opacity', 1);
                tooltip.innerHTML = '<strong>t = ' + t.toFixed(1) + ' ms</strong><br/>V/V\u2080 = ' + V.toFixed(3);
                tooltip.style.display = 'block';
                tooltip.style.left = (event.clientX + 12) + 'px';
                tooltip.style.top = (event.clientY - 10) + 'px';
            })
            .on('mouseout', function() {
                vLine.style('opacity', 0); hLine.style('opacity', 0); hoverDot.style('opacity', 0);
                tooltip.style.display = 'none';
            });

        svg.append('text').attr('x', width / 2).attr('y', height + 38)
            .text('Time (ms)').style('fill', '#667').style('font-size', '11px').attr('text-anchor', 'middle');
        svg.append('text').attr('transform', 'rotate(-90)').attr('x', -height / 2).attr('y', -48)
            .text('V/V\u2080 (normalized)').style('fill', '#667').style('font-size', '11px').attr('text-anchor', 'middle');
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
        const margin = { top: 20, right: 20, bottom: 45, left: 60 };
        const width = (container.clientWidth || 400) - margin.left - margin.right;
        const height = 280 - margin.top - margin.bottom;
        const tooltip = ensureTooltip();

        const svg = d3.select(container).append('svg')
            .attr('width', width + margin.left + margin.right)
            .attr('height', height + margin.top + margin.bottom)
            .append('g').attr('transform', `translate(${margin.left},${margin.top})`);

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
            .call(d3.axisBottom(x).ticks(10)).selectAll('text').style('fill', '#667').style('font-size', '10px');
        svg.append('g').call(d3.axisLeft(y).ticks(5))
            .selectAll('text').style('fill', '#667').style('font-size', '10px');
        svg.selectAll('.domain, .tick line').style('stroke', '#1a2a3a');

        svg.append('line').attr('x1', 0).attr('x2', width).attr('y1', y(0)).attr('y2', y(0))
            .attr('stroke', '#1a2a3a').attr('stroke-dasharray', '2,2');

        svg.append('line').attr('x1', x(25)).attr('x2', x(25)).attr('y1', 0).attr('y2', height)
            .attr('stroke', '#4ecdc4').attr('stroke-dasharray', '4,4').attr('opacity', 0.6);
        svg.append('text').attr('x', x(25) + 4).attr('y', 12)
            .text('~25 bp coherence').style('fill', '#4ecdc4').style('font-size', '9px');

        const barWidth = Math.max(1, width / bpCount - 1);
        svg.selectAll('.bar').data(data).enter().append('rect')
            .attr('x', d => x(d.bp) - barWidth / 2)
            .attr('y', d => d.amplitude >= 0 ? y(d.amplitude) : y(0))
            .attr('width', barWidth)
            .attr('height', 0)
            .attr('fill', d => d.amplitude >= 0 ? 'rgba(249, 215, 126, 0.3)' : 'rgba(78, 205, 196, 0.3)')
            .attr('cursor', 'pointer')
            .transition().delay((d, i) => i * 30).duration(400)
            .attr('height', d => Math.abs(y(0) - y(d.amplitude)));

        const line = d3.line().x(d => x(d.bp)).y(d => y(d.amplitude)).curve(d3.curveBasis);
        const path = svg.append('path').datum(data)
            .attr('fill', 'none').attr('stroke', '#f9d77e').attr('stroke-width', 2).attr('d', line);
        const totalLength = path.node().getTotalLength();
        path.attr('stroke-dasharray', totalLength).attr('stroke-dashoffset', totalLength)
            .transition().duration(2000).ease(d3.easeLinear).attr('stroke-dashoffset', 0);

        svg.selectAll('.hover-zone').data(data).enter().append('rect')
            .attr('x', d => x(d.bp) - barWidth).attr('y', 0)
            .attr('width', barWidth * 2).attr('height', height).attr('fill', 'transparent')
            .on('mouseover', function(event, d) {
                tooltip.innerHTML = '<strong>bp ' + d.bp + '</strong><br/>Displacement: ' + d.amplitude.toFixed(3);
                tooltip.style.display = 'block';
                tooltip.style.left = (event.clientX + 12) + 'px';
                tooltip.style.top = (event.clientY - 10) + 'px';
            })
            .on('mousemove', function(event) {
                tooltip.style.left = (event.clientX + 12) + 'px';
                tooltip.style.top = (event.clientY - 10) + 'px';
            })
            .on('mouseout', function() { tooltip.style.display = 'none'; });

        svg.append('text').attr('x', width / 2).attr('y', height + 38)
            .text('Base Pair Position').style('fill', '#667').style('font-size', '11px').attr('text-anchor', 'middle');
        svg.append('text').attr('transform', 'rotate(-90)').attr('x', -height / 2).attr('y', -48)
            .text('Proton Displacement').style('fill', '#667').style('font-size', '11px').attr('text-anchor', 'middle');
    }, []);
    return <div ref={ref} className="d3_chart_container" />;
}

export default function Dynamics({ ActiveIndex }) {
    const sectionRef = useRef(null);
    const [activeChart, setActiveChart] = useState(0);

    useEffect(() => {
        if (ActiveIndex !== 5 || !sectionRef.current) return;

        const { gsap } = require('gsap');
        const { ScrollTrigger } = require('gsap/dist/ScrollTrigger');
        gsap.registerPlugin(ScrollTrigger);

        const scroller = sectionRef.current;
        const steps = scroller.querySelectorAll('.scrolly_step');
        const triggers = [];

        steps.forEach((step) => {
            const st = ScrollTrigger.create({
                scroller,
                trigger: step,
                start: 'top 70%',
                end: 'bottom 30%',
                toggleClass: 'active',
                onEnter: () => {
                    const idx = parseInt(step.getAttribute('data-chart') || '0');
                    setActiveChart(idx);
                },
                onEnterBack: () => {
                    const idx = parseInt(step.getAttribute('data-chart') || '0');
                    setActiveChart(idx);
                },
            });
            triggers.push(st);
        });

        return () => {
            triggers.forEach(t => t.kill());
        };
    }, [ActiveIndex]);

    return (
        <>
            <div
                ref={sectionRef}
                className={ActiveIndex === 5 ? "cavani_tm_section active animated fadeInUp" : "cavani_tm_section hidden animated"}
                id="dynamics_"
            >
                <div className="section_inner">
                    <div className="cavani_tm_about">
                        <div className="cavani_tm_title">
                            <span>Temporal Charge Dynamics in Nucleic Acids</span>
                        </div>
                        <div className="paper_meta">
                            <p className="paper_author">Kundai Farai Sachikonye &mdash; Technical University of Munich</p>
                        </div>

                        <div className="scrolly_container">
                            <div className="scrolly_chart_wrapper">
                                <div className="scrolly_chart_inner">
                                    <div className={`scrolly_chart ${activeChart === 0 ? 'visible' : ''}`}>
                                        <div className="chart_title">DNA Capacitance vs Length</div>
                                        <CapacitanceChart />
                                        <p className="chart_caption">Hover data points to inspect values. Human genome highlighted.</p>
                                    </div>
                                    <div className={`scrolly_chart ${activeChart === 1 ? 'visible' : ''}`}>
                                        <div className="chart_title">RC Discharge Dynamics</div>
                                        <DischargeChart />
                                        <p className="chart_caption">Move cursor across chart to trace discharge curve.</p>
                                    </div>
                                    <div className={`scrolly_chart ${activeChart === 2 ? 'visible' : ''}`}>
                                        <div className="chart_title">Phase-Locked H-Bond Oscillations</div>
                                        <OscillationChart />
                                        <p className="chart_caption">Hover bars to inspect individual base pair displacements.</p>
                                    </div>
                                </div>
                            </div>

                            <div className="scrolly_steps">
                                <section className="scrolly_step active" data-chart="0">
                                    <h3 className="paper_section_title">Abstract</h3>
                                    <p>This work establishes that nucleic acids function as deterministic charge oscillators with predictable temporal dynamics. Starting from partition theory, where hydrogen-bond protons occupy categorical binary states (donor or acceptor) at each instant, we derive that DNA exhibits capacitive charge storage with capacitance C &asymp; 300 pF and predictable discharge dynamics with time constant &tau;<sub>RC</sub> &asymp; 30 ms.</p>
                                    <p>Experimental validation achieves a 4.27 &times; 10<sup>5</sup>-fold improvement over Heisenberg-limited measurement, with proton trajectory determinism reaching a relative standard deviation of 4.67 &times; 10<sup>&minus;7</sup>.</p>
                                </section>

                                <section className="scrolly_step" data-chart="0">
                                    <h3 className="paper_section_title">DNA as a Charge Capacitor</h3>
                                    <p>The phosphodiester backbone carries approximately one negative charge per nucleotide. For the human genome (&sim;6 &times; 10<sup>9</sup> bp), this yields a linear charge density exceeding 1 e/nm. Modeled as a cylindrical capacitor, the resulting capacitance scales linearly with DNA length, producing C &asymp; 300 pF for the full human genome &mdash; comparable to discrete electronic capacitors.</p>
                                </section>

                                <section className="scrolly_step" data-chart="1">
                                    <h3 className="paper_section_title">Discharge Dynamics</h3>
                                    <p>DNA discharge follows exponential RC dynamics with &tau;<sub>RC</sub> &asymp; 30 ms, emerging from the product of genomic capacitance and the effective resistance of the ionic environment. This deterministic timescale governs charge redistribution within the genome.</p>
                                </section>

                                <section className="scrolly_step" data-chart="2">
                                    <h3 className="paper_section_title">Phase-Locked Networks</h3>
                                    <p>Hydrogen-bond protons oscillate at &sim;10<sup>13</sup> Hz within each base pair. These oscillations become phase-locked across neighboring base pairs, establishing coherent domains of &sim;25 bp &mdash; matching nucleosome organization.</p>
                                </section>

                                <section className="scrolly_step" data-chart="2">
                                    <h3 className="paper_section_title">Key Results</h3>
                                    <div className="key_results_grid">
                                        <div className="result_card">
                                            <div className="result_value">300 pF</div>
                                            <div className="result_label">DNA Capacitance</div>
                                        </div>
                                        <div className="result_card">
                                            <div className="result_value">30 ms</div>
                                            <div className="result_label">&tau;<sub>RC</sub> Time Constant</div>
                                        </div>
                                        <div className="result_card">
                                            <div className="result_value">10<sup>13</sup> Hz</div>
                                            <div className="result_label">Oscillation Frequency</div>
                                        </div>
                                        <div className="result_card">
                                            <div className="result_value">~25 bp</div>
                                            <div className="result_label">Coherence Length</div>
                                        </div>
                                        <div className="result_card">
                                            <div className="result_value">4.27&times;10<sup>5</sup></div>
                                            <div className="result_label">Measurement Improvement</div>
                                        </div>
                                        <div className="result_card">
                                            <div className="result_value">2.8&times;10<sup>&minus;16</sup></div>
                                            <div className="result_label">Triple Equivalence Precision</div>
                                        </div>
                                    </div>
                                </section>

                                <section className="scrolly_step" data-chart="0">
                                    <h3 className="paper_section_title">Implications</h3>
                                    <p>DNA is reinterpreted as an active charge dynamics system rather than passive information storage. Gene regulation emerges from charge distribution dynamics. The C-Value Paradox is resolved: large non-coding regions serve charge storage and regulation functions.</p>
                                </section>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </>
    );
}
