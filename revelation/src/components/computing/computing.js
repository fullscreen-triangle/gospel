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

function ComplexityChart() {
    const ref = useRef(null);
    useEffect(() => {
        if (!ref.current) return;
        const d3 = require('d3');
        const container = ref.current;
        container.innerHTML = '';
        const margin = { top: 20, right: 100, bottom: 45, left: 60 };
        const width = (container.clientWidth || 400) - margin.left - margin.right;
        const height = 280 - margin.top - margin.bottom;
        const tooltip = ensureTooltip();

        const svg = d3.select(container).append('svg')
            .attr('width', width + margin.left + margin.right)
            .attr('height', height + margin.top + margin.bottom)
            .append('g').attr('transform', `translate(${margin.left},${margin.top})`);

        const nValues = [100, 1000, 10000, 100000, 1e6, 1e7, 1e8];
        const datasets = [
            { name: 'O(n\u00B2) Sequential', color: '#ff6b6b', dash: '6,3', fn: n => n * n },
            { name: 'O(n log n) Binary', color: '#4ecdc4', dash: '3,3', fn: n => n * Math.log2(n) },
            { name: 'O(log\u2083 n) Ternary', color: '#f9d77e', dash: '', fn: n => Math.log(n) / Math.log(3) },
        ];

        const x = d3.scaleLog().domain([100, 1e8]).range([0, width]);
        const y = d3.scaleLog().domain([1, 1e16]).range([height, 0]);

        svg.append('g').attr('transform', `translate(0,${height})`)
            .call(d3.axisBottom(x).ticks(4, '.0e')).selectAll('text').style('fill', '#667').style('font-size', '10px');
        svg.append('g').call(d3.axisLeft(y).ticks(4, '.0e'))
            .selectAll('text').style('fill', '#667').style('font-size', '10px');
        svg.selectAll('.domain, .tick line').style('stroke', '#1a2a3a');

        const line = d3.line().x(d => x(d.n)).y(d => y(d.ops));

        datasets.forEach((ds, idx) => {
            const data = nValues.map(n => ({ n, ops: ds.fn(n) }));
            const path = svg.append('path').datum(data)
                .attr('fill', 'none').attr('stroke', ds.color).attr('stroke-width', ds.dash ? 2 : 3)
                .attr('stroke-dasharray', ds.dash).attr('d', line);
            const totalLength = path.node().getTotalLength();
            path.attr('stroke-dasharray', totalLength).attr('stroke-dashoffset', totalLength)
                .transition().duration(1500).delay(idx * 400).ease(d3.easeCubicOut)
                .attr('stroke-dashoffset', 0)
                .on('end', function() { d3.select(this).attr('stroke-dasharray', ds.dash); });

            svg.selectAll('.dot-' + idx).data(data).enter().append('circle')
                .attr('cx', d => x(d.n)).attr('cy', d => y(d.ops))
                .attr('r', 0).attr('fill', ds.color).attr('cursor', 'pointer')
                .transition().delay(idx * 400 + 1000).duration(300).attr('r', 3.5);

            svg.selectAll('.hover-' + idx).data(data).enter().append('circle')
                .attr('cx', d => x(d.n)).attr('cy', d => y(d.ops))
                .attr('r', 12).attr('fill', 'transparent').attr('cursor', 'pointer')
                .on('mouseover', function(event, d) {
                    d3.select(this.previousSibling).transition().duration(150).attr('r', 6);
                    tooltip.innerHTML = '<strong>' + ds.name + '</strong><br/>n = ' + d.n.toExponential(0) + '<br/>ops = ' + (d.ops < 100 ? d.ops.toFixed(1) : d.ops.toExponential(1));
                    tooltip.style.display = 'block';
                    tooltip.style.left = (event.clientX + 12) + 'px';
                    tooltip.style.top = (event.clientY - 10) + 'px';
                })
                .on('mouseout', function() {
                    d3.select(this.previousSibling).transition().duration(150).attr('r', 3.5);
                    tooltip.style.display = 'none';
                });
        });

        const legend = svg.append('g').attr('transform', `translate(${width + 8}, 10)`);
        datasets.forEach((ds, i) => {
            legend.append('line').attr('x1', 0).attr('x2', 16).attr('y1', i * 20).attr('y2', i * 20)
                .attr('stroke', ds.color).attr('stroke-width', 2);
            legend.append('text').attr('x', 20).attr('y', i * 20 + 4)
                .text(ds.name).style('fill', '#667').style('font-size', '9px');
        });

        svg.append('text').attr('x', width / 2).attr('y', height + 38)
            .text('Input Size (n)').style('fill', '#667').style('font-size', '11px').attr('text-anchor', 'middle');
        svg.append('text').attr('transform', 'rotate(-90)').attr('x', -height / 2).attr('y', -48)
            .text('Operations').style('fill', '#667').style('font-size', '11px').attr('text-anchor', 'middle');
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
        const margin = { top: 20, right: 20, bottom: 45, left: 60 };
        const width = (container.clientWidth || 400) - margin.left - margin.right;
        const height = 280 - margin.top - margin.bottom;
        const tooltip = ensureTooltip();

        const svg = d3.select(container).append('svg')
            .attr('width', width + margin.left + margin.right)
            .attr('height', height + margin.top + margin.bottom)
            .append('g').attr('transform', `translate(${margin.left},${margin.top})`);

        const data = [
            { k: 6, cells: 729, resolution: 0.11 },
            { k: 12, cells: 531441, resolution: 0.012 },
            { k: 18, cells: 387e6, resolution: 0.0014 },
            { k: 24, cells: 282e9, resolution: 0.00016 },
            { k: 30, cells: 206e12, resolution: 1.8e-5 },
        ];

        const x = d3.scaleLinear().domain([4, 32]).range([0, width]);
        const y = d3.scaleLog().domain([1e-5, 0.2]).range([height, 0]);

        svg.append('g').attr('transform', `translate(0,${height})`)
            .call(d3.axisBottom(x).ticks(5)).selectAll('text').style('fill', '#667').style('font-size', '10px');
        svg.append('g').call(d3.axisLeft(y).ticks(4, '.0e'))
            .selectAll('text').style('fill', '#667').style('font-size', '10px');
        svg.selectAll('.domain, .tick line').style('stroke', '#1a2a3a');

        const barWidth = 24;
        const yBar = d3.scaleLog().domain([100, 1e13]).range([height, 0]);
        data.forEach((d, i) => {
            svg.append('rect')
                .attr('x', x(d.k) - barWidth / 2).attr('y', height)
                .attr('width', barWidth).attr('height', 0)
                .attr('fill', 'rgba(78, 205, 196, 0.2)').attr('stroke', '#4ecdc4').attr('stroke-width', 1)
                .transition().delay(i * 200).duration(600).ease(d3.easeBounceOut)
                .attr('y', yBar(d.cells)).attr('height', height - yBar(d.cells));
        });

        const line = d3.line().x(d => x(d.k)).y(d => y(d.resolution));
        const path = svg.append('path').datum(data)
            .attr('fill', 'none').attr('stroke', '#f9d77e').attr('stroke-width', 2.5).attr('d', line);
        const totalLength = path.node().getTotalLength();
        path.attr('stroke-dasharray', totalLength).attr('stroke-dashoffset', totalLength)
            .transition().duration(1500).delay(500).attr('stroke-dashoffset', 0);

        data.forEach((d, i) => {
            svg.append('circle').attr('cx', x(d.k)).attr('cy', y(d.resolution))
                .attr('r', 0).attr('fill', '#f9d77e')
                .transition().delay(i * 200 + 800).duration(300).attr('r', 5);
        });

        svg.selectAll('.hover').data(data).enter().append('circle')
            .attr('cx', d => x(d.k)).attr('cy', d => y(d.resolution))
            .attr('r', 18).attr('fill', 'transparent').attr('cursor', 'pointer')
            .on('mouseover', function(event, d) {
                tooltip.innerHTML = '<strong>k = ' + d.k + ' trits</strong><br/>Cells: ' + (d.cells >= 1e6 ? d.cells.toExponential(1) : d.cells.toLocaleString()) + '<br/>Resolution: ' + d.resolution.toExponential(1);
                tooltip.style.display = 'block';
                tooltip.style.left = (event.clientX + 12) + 'px';
                tooltip.style.top = (event.clientY - 10) + 'px';
            })
            .on('mouseout', function() { tooltip.style.display = 'none'; });

        svg.append('text').attr('x', width / 2).attr('y', height + 38)
            .text('Trit Depth (k)').style('fill', '#667').style('font-size', '11px').attr('text-anchor', 'middle');
        svg.append('text').attr('transform', 'rotate(-90)').attr('x', -height / 2).attr('y', -48)
            .text('Resolution (\u03B5)').style('fill', '#667').style('font-size', '11px').attr('text-anchor', 'middle');
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
        const margin = { top: 20, right: 20, bottom: 45, left: 50 };
        const size = Math.min((container.clientWidth || 400) - margin.left - margin.right, 300);
        const tooltip = ensureTooltip();

        const svg = d3.select(container).append('svg')
            .attr('width', size + margin.left + margin.right)
            .attr('height', size + margin.top + margin.bottom)
            .append('g').attr('transform', `translate(${margin.left},${margin.top})`);

        const scale = d3.scaleLinear().domain([-1.5, 1.5]).range([0, size]);

        svg.append('line').attr('x1', scale(-1.5)).attr('x2', scale(1.5)).attr('y1', scale(0)).attr('y2', scale(0)).attr('stroke', '#1a2a3a');
        svg.append('line').attr('x1', scale(0)).attr('x2', scale(0)).attr('y1', scale(-1.5)).attr('y2', scale(1.5)).attr('stroke', '#1a2a3a');

        const bases = [
            { label: 'A', x: 0, y: 1, color: '#4ecdc4', desc: 'Adenine (0, +1)\nDonor \u00B7 Purine' },
            { label: 'T', x: 0, y: -1, color: '#ff6b6b', desc: 'Thymine (0, \u22121)\nAcceptor \u00B7 Pyrimidine' },
            { label: 'G', x: 1, y: 0, color: '#f9d77e', desc: 'Guanine (+1, 0)\nDonor \u00B7 Pyrimidine' },
            { label: 'C', x: -1, y: 0, color: '#a78bfa', desc: 'Cytosine (\u22121, 0)\nAcceptor \u00B7 Purine' },
        ];

        [[[0, 1], [0, -1]], [[-1, 0], [1, 0]]].forEach((pair, i) => {
            svg.append('line')
                .attr('x1', scale(pair[0][0])).attr('y1', scale(-pair[0][1]))
                .attr('x2', scale(pair[0][0])).attr('y2', scale(-pair[0][1]))
                .attr('stroke', '#333').attr('stroke-dasharray', '4,3').attr('stroke-width', 1.5)
                .transition().duration(800).delay(i * 300)
                .attr('x2', scale(pair[1][0])).attr('y2', scale(-pair[1][1]));
        });

        bases.forEach((b, i) => {
            const g = svg.append('g').attr('transform', `translate(${scale(b.x)},${scale(-b.y)})`).attr('cursor', 'pointer');
            g.append('circle').attr('r', 0).attr('fill', b.color).attr('opacity', 0.15)
                .attr('stroke', b.color).attr('stroke-width', 2)
                .transition().delay(i * 200 + 600).duration(500).ease(d3.easeBackOut).attr('r', 22);
            g.append('text').attr('y', 5).text(b.label)
                .style('fill', '#fff').style('font-size', '16px').style('font-weight', 'bold').attr('text-anchor', 'middle')
                .style('opacity', 0).transition().delay(i * 200 + 900).duration(300).style('opacity', 1);

            g.append('circle').attr('r', 28).attr('fill', 'transparent')
                .on('mouseover', function(event) {
                    g.select('circle').transition().duration(200).attr('r', 28).attr('opacity', 0.3);
                    tooltip.innerHTML = '<strong>' + b.desc.split('\n')[0] + '</strong><br/>' + (b.desc.split('\n')[1] || '');
                    tooltip.style.display = 'block';
                    tooltip.style.left = (event.clientX + 12) + 'px';
                    tooltip.style.top = (event.clientY - 10) + 'px';
                })
                .on('mouseout', function() {
                    g.select('circle').transition().duration(200).attr('r', 22).attr('opacity', 0.15);
                    tooltip.style.display = 'none';
                });
        });
    }, []);
    return <div ref={ref} className="d3_chart_container" />;
}

export default function Computing({ ActiveIndex }) {
    const sectionRef = useRef(null);
    const [activeChart, setActiveChart] = useState(0);

    useEffect(() => {
        if (ActiveIndex !== 6 || !sectionRef.current) return;

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
                className={ActiveIndex === 6 ? "cavani_tm_section active animated fadeInUp" : "cavani_tm_section hidden animated"}
                id="computing_"
            >
                <div className="section_inner">
                    <div className="cavani_tm_about">
                        <div className="cavani_tm_title">
                            <span>Ternary Partition Computing for Genomic Systems</span>
                        </div>
                        <div className="paper_meta">
                            <p className="paper_author">Kundai Farai Sachikonye &mdash; Technical University of Munich</p>
                        </div>

                        <div className="scrolly_container">
                            <div className="scrolly_chart_wrapper">
                                <div className="scrolly_chart_inner">
                                    <div className={`scrolly_chart ${activeChart === 0 ? 'visible' : ''}`}>
                                        <div className="chart_title">Complexity Comparison</div>
                                        <ComplexityChart />
                                        <p className="chart_caption">Hover data points for exact operation counts at each scale.</p>
                                    </div>
                                    <div className={`scrolly_chart ${activeChart === 1 ? 'visible' : ''}`}>
                                        <div className="chart_title">Cardinal Coordinate Transformation</div>
                                        <CardinalChart />
                                        <p className="chart_caption">Hover nucleotides to see partition properties.</p>
                                    </div>
                                    <div className={`scrolly_chart ${activeChart === 2 ? 'visible' : ''}`}>
                                        <div className="chart_title">Trit Resolution Scaling</div>
                                        <ResolutionChart />
                                        <p className="chart_caption">Bars: addressable cells. Line: resolution &epsilon;. Hover for values.</p>
                                    </div>
                                </div>
                            </div>

                            <div className="scrolly_steps">
                                <section className="scrolly_step active" data-chart="0">
                                    <h3 className="paper_section_title">Abstract</h3>
                                    <p>A ternary-based computing framework where solutions exist as predetermined endpoints in partition coordinates. Computation becomes navigation rather than calculation. The ternary representation encodes three-dimensional S-entropy space S = [0,1]<sup>3</sup>, where each trit specifies refinement along knowledge, temporal, or evolution axes.</p>
                                    <p>Three fundamental operations &mdash; project, complete, compose &mdash; replace Boolean AND, OR, NOT. The architecture achieves O(log<sub>3</sub> n) navigation complexity versus O(n<sup>2</sup>) for sequential analysis.</p>
                                </section>

                                <section className="scrolly_step" data-chart="0">
                                    <h3 className="paper_section_title">Why Ternary?</h3>
                                    <p>A single trit carries log<sub>2</sub>(3) &asymp; 1.585 bits &mdash; 58.5% more information per symbol than binary. A tryte (6 trits) encodes 3<sup>6</sup> = 729 distinct S-space cells versus 2<sup>8</sup> = 256 for a binary byte. For a human genome with n &asymp; 6 &times; 10<sup>9</sup> bp, ternary navigation reduces &sim;10<sup>19</sup> sequential operations to &sim;20 navigation steps.</p>
                                </section>

                                <section className="scrolly_step" data-chart="1">
                                    <h3 className="paper_section_title">Cardinal Transformation</h3>
                                    <p>The cardinal transformation &phi;: &#123;A,T,G,C&#125; &rarr; &#8477;<sup>2</sup> maps nucleotides so that complementary pairs are geometric negations: A&harr;T as (0,+1)&harr;(0,&minus;1) and G&harr;C as (+1,0)&harr;(&minus;1,0). Watson-Crick pairing becomes partition inversion.</p>
                                </section>

                                <section className="scrolly_step" data-chart="2">
                                    <h3 className="paper_section_title">Resolution Scaling</h3>
                                    <p>Each additional trit of depth refines the S-space partition by a factor of 3. At k = 6 trits, 729 cells are addressable with resolution &epsilon; = 0.11. At k = 30, over 2 &times; 10<sup>14</sup> cells provide resolution &epsilon; &asymp; 1.8 &times; 10<sup>&minus;5</sup>, sufficient for single-nucleotide precision across the human genome.</p>
                                </section>

                                <section className="scrolly_step" data-chart="0">
                                    <h3 className="paper_section_title">Partition Synthesis Language</h3>
                                    <div className="code_block">
                                        <pre><code>{`genome hg38 {
  source: "reference/hg38.fa"
  partition_depth: 24
}

feature promoter = synthesize {
  navigate: S_k > 0.8
  filter:   S_t in [0.2, 0.6]
  complete: from trajectory
}`}</code></pre>
                                    </div>
                                </section>

                                <section className="scrolly_step" data-chart="0">
                                    <h3 className="paper_section_title">Key Results</h3>
                                    <div className="key_results_grid">
                                        <div className="result_card">
                                            <div className="result_value">O(log<sub>3</sub> n)</div>
                                            <div className="result_label">Navigation Complexity</div>
                                        </div>
                                        <div className="result_card">
                                            <div className="result_value">1.585 bits</div>
                                            <div className="result_label">Per Trit</div>
                                        </div>
                                        <div className="result_card">
                                            <div className="result_value">729</div>
                                            <div className="result_label">Tryte Values (3<sup>6</sup>)</div>
                                        </div>
                                        <div className="result_card">
                                            <div className="result_value">0</div>
                                            <div className="result_label">Free Parameters</div>
                                        </div>
                                    </div>
                                </section>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </>
    );
}
