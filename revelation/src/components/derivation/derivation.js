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

function PartitionStatesChart() {
    const ref = useRef(null);
    useEffect(() => {
        if (!ref.current) return;
        const d3 = require('d3');
        const container = ref.current;
        container.innerHTML = '';
        const margin = { top: 30, right: 20, bottom: 20, left: 20 };
        const width = (container.clientWidth || 400) - margin.left - margin.right;
        const height = 340 - margin.top - margin.bottom;
        const tooltip = ensureTooltip();

        const svg = d3.select(container).append('svg')
            .attr('width', width + margin.left + margin.right)
            .attr('height', height + margin.top + margin.bottom)
            .append('g').attr('transform', `translate(${margin.left},${margin.top})`);

        const states = [
            { label: 'A', full: 'Adenine', p1: 'Donor', p2: 'Purine', x: 0.25, y: 0.75, color: '#4ecdc4', hBonds: 2, complement: 'T' },
            { label: 'T', full: 'Thymine', p1: 'Acceptor', p2: 'Pyrimidine', x: 0.75, y: 0.75, color: '#ff6b6b', hBonds: 2, complement: 'A' },
            { label: 'G', full: 'Guanine', p1: 'Donor', p2: 'Pyrimidine', x: 0.25, y: 0.25, color: '#f9d77e', hBonds: 3, complement: 'C' },
            { label: 'C', full: 'Cytosine', p1: 'Acceptor', p2: 'Purine', x: 0.75, y: 0.25, color: '#a78bfa', hBonds: 3, complement: 'G' },
        ];

        const x = d3.scaleLinear().domain([0, 1]).range([0, width]);
        const y = d3.scaleLinear().domain([0, 1]).range([height, 0]);

        svg.append('line').attr('x1', x(0.5)).attr('x2', x(0.5)).attr('y1', 0).attr('y2', height)
            .attr('stroke', '#1a2a3a').attr('stroke-width', 2);
        svg.append('line').attr('x1', 0).attr('x2', width).attr('y1', y(0.5)).attr('y2', y(0.5))
            .attr('stroke', '#1a2a3a').attr('stroke-width', 2);

        svg.append('text').attr('x', x(0.25)).attr('y', -12)
            .text('Partition I: Donor').style('fill', '#667').style('font-size', '10px').attr('text-anchor', 'middle');
        svg.append('text').attr('x', x(0.75)).attr('y', -12)
            .text('Partition I: Acceptor').style('fill', '#667').style('font-size', '10px').attr('text-anchor', 'middle');
        svg.append('text').attr('transform', 'rotate(-90)').attr('x', -y(0.75)).attr('y', -8)
            .text('Partition II: Purine').style('fill', '#667').style('font-size', '10px').attr('text-anchor', 'middle');
        svg.append('text').attr('transform', 'rotate(-90)').attr('x', -y(0.25)).attr('y', -8)
            .text('Partition II: Pyrimidine').style('fill', '#667').style('font-size', '10px').attr('text-anchor', 'middle');

        var pairs = [
            { x1: 0.25, y1: 0.75, x2: 0.75, y2: 0.75, label: 'A\u2194T' },
            { x1: 0.25, y1: 0.25, x2: 0.75, y2: 0.25, label: 'G\u2194C' },
        ];
        pairs.forEach(function(p, i) {
            svg.append('line')
                .attr('x1', x(p.x1) + 40).attr('y1', y(p.y1))
                .attr('x2', x(p.x2) - 40).attr('y2', y(p.y2))
                .attr('stroke', '#555').attr('stroke-dasharray', '6,4').attr('stroke-width', 1.5)
                .style('opacity', 0)
                .transition().delay(1200 + i * 300).duration(600).style('opacity', 1);

            svg.append('text')
                .attr('x', x(0.5)).attr('y', y(p.y1) - 10)
                .text(p.label + ' complementary pair')
                .style('fill', '#555').style('font-size', '9px').attr('text-anchor', 'middle')
                .style('opacity', 0).transition().delay(1400 + i * 300).duration(400).style('opacity', 1);
        });

        states.forEach(function(s, i) {
            var outerCircle = svg.append('circle')
                .attr('cx', x(s.x)).attr('cy', y(s.y))
                .attr('r', 0).attr('fill', s.color).attr('opacity', 0.15)
                .attr('stroke', s.color).attr('stroke-width', 2);
            outerCircle.transition().delay(i * 200).duration(600).ease(d3.easeBackOut).attr('r', 35);

            svg.append('text')
                .attr('x', x(s.x)).attr('y', y(s.y) + 5)
                .text(s.label)
                .style('fill', s.color).style('font-size', '16px').style('font-weight', 'bold')
                .attr('text-anchor', 'middle').style('pointer-events', 'none');

            svg.append('circle')
                .attr('cx', x(s.x)).attr('cy', y(s.y))
                .attr('r', 40).attr('fill', 'transparent').attr('cursor', 'pointer')
                .on('mouseover', function(event) {
                    outerCircle.transition().duration(200).attr('opacity', 0.35).attr('r', 42);
                    tooltip.innerHTML = '<strong>' + s.full + ' (' + s.label + ')</strong><br/>H-Bond: ' + s.p1 + '<br/>Ring: ' + s.p2 + '<br/>H-Bonds: ' + s.hBonds + '<br/>Complement: ' + s.complement;
                    tooltip.style.display = 'block';
                    tooltip.style.left = (event.clientX + 12) + 'px';
                    tooltip.style.top = (event.clientY - 10) + 'px';
                })
                .on('mousemove', function(event) {
                    tooltip.style.left = (event.clientX + 12) + 'px';
                    tooltip.style.top = (event.clientY - 10) + 'px';
                })
                .on('mouseout', function() {
                    outerCircle.transition().duration(200).attr('opacity', 0.15).attr('r', 35);
                    tooltip.style.display = 'none';
                });
        });
    }, []);
    return <div ref={ref} className="d3_chart_container" />;
}

function BindingEnergyChart() {
    const ref = useRef(null);
    useEffect(() => {
        if (!ref.current) return;
        const d3 = require('d3');
        const container = ref.current;
        container.innerHTML = '';
        const margin = { top: 20, right: 20, bottom: 65, left: 65 };
        const width = (container.clientWidth || 400) - margin.left - margin.right;
        const height = 300 - margin.top - margin.bottom;
        const tooltip = ensureTooltip();

        const svg = d3.select(container).append('svg')
            .attr('width', width + margin.left + margin.right)
            .attr('height', height + margin.top + margin.bottom)
            .append('g').attr('transform', `translate(${margin.left},${margin.top})`);

        const data = [
            { pair: 'A\u2013T', observed: -1.25, predicted: -1.2, hBonds: 2 },
            { pair: 'G\u2013C', observed: -2.5, predicted: -2.4, hBonds: 3 },
            { pair: 'A\u2013A (mis)', observed: -0.2, predicted: 0, hBonds: 0 },
            { pair: 'T\u2013G (mis)', observed: -0.1, predicted: 0, hBonds: 0 },
        ];

        const xScale = d3.scaleBand().domain(data.map(function(d) { return d.pair; })).range([0, width]).padding(0.3);
        const yScale = d3.scaleLinear().domain([-3.5, 0.5]).range([height, 0]);

        svg.append('g').attr('transform', 'translate(0,' + yScale(0) + ')')
            .call(d3.axisBottom(xScale))
            .selectAll('text').style('fill', '#667').style('font-size', '10px');
        svg.append('g').call(d3.axisLeft(yScale).ticks(7))
            .selectAll('text').style('fill', '#667').style('font-size', '10px');
        svg.selectAll('.domain, .tick line').style('stroke', '#1a2a3a');

        svg.append('line').attr('x1', 0).attr('x2', width)
            .attr('y1', yScale(0)).attr('y2', yScale(0)).attr('stroke', '#333');

        const barWidth = xScale.bandwidth() / 2 - 2;

        data.forEach(function(d, i) {
            var barY = d.observed < 0 ? yScale(0) : yScale(d.observed);
            var barH = Math.abs(yScale(0) - yScale(d.observed));
            svg.append('rect')
                .attr('x', xScale(d.pair)).attr('y', yScale(0))
                .attr('width', barWidth).attr('height', 0)
                .attr('fill', '#f9d77e').attr('opacity', 0.8).attr('cursor', 'pointer')
                .transition().delay(i * 150).duration(600).ease(d3.easeCubicOut)
                .attr('y', barY).attr('height', barH);

            svg.append('rect')
                .attr('x', xScale(d.pair)).attr('y', 0)
                .attr('width', barWidth).attr('height', height)
                .attr('fill', 'transparent').attr('cursor', 'pointer')
                .on('mouseover', function(event) {
                    tooltip.innerHTML = '<strong>' + d.pair + ' (Observed)</strong><br/>\u0394G = ' + d.observed.toFixed(2) + ' kcal/mol<br/>H-Bonds: ' + d.hBonds;
                    tooltip.style.display = 'block';
                    tooltip.style.left = (event.clientX + 12) + 'px';
                    tooltip.style.top = (event.clientY - 10) + 'px';
                })
                .on('mousemove', function(event) {
                    tooltip.style.left = (event.clientX + 12) + 'px';
                    tooltip.style.top = (event.clientY - 10) + 'px';
                })
                .on('mouseout', function() { tooltip.style.display = 'none'; });
        });

        data.forEach(function(d, i) {
            var barY = d.predicted < 0 ? yScale(0) : yScale(d.predicted);
            var barH = Math.max(1, Math.abs(yScale(0) - yScale(d.predicted)));
            svg.append('rect')
                .attr('x', xScale(d.pair) + barWidth + 4).attr('y', yScale(0))
                .attr('width', barWidth).attr('height', 0)
                .attr('fill', '#4ecdc4').attr('opacity', 0.8).attr('cursor', 'pointer')
                .transition().delay(i * 150 + 100).duration(600).ease(d3.easeCubicOut)
                .attr('y', barY).attr('height', barH);

            svg.append('rect')
                .attr('x', xScale(d.pair) + barWidth + 4).attr('y', 0)
                .attr('width', barWidth).attr('height', height)
                .attr('fill', 'transparent').attr('cursor', 'pointer')
                .on('mouseover', function(event) {
                    tooltip.innerHTML = '<strong>' + d.pair + ' (Predicted)</strong><br/>\u0394G = ' + d.predicted.toFixed(2) + ' kcal/mol';
                    tooltip.style.display = 'block';
                    tooltip.style.left = (event.clientX + 12) + 'px';
                    tooltip.style.top = (event.clientY - 10) + 'px';
                })
                .on('mousemove', function(event) {
                    tooltip.style.left = (event.clientX + 12) + 'px';
                    tooltip.style.top = (event.clientY - 10) + 'px';
                })
                .on('mouseout', function() { tooltip.style.display = 'none'; });
        });

        var legend = svg.append('g').attr('transform', 'translate(' + (width - 130) + ', 5)');
        legend.append('rect').attr('width', 12).attr('height', 12).attr('fill', '#f9d77e');
        legend.append('text').attr('x', 18).attr('y', 11).text('Observed').style('fill', '#667').style('font-size', '10px');
        legend.append('rect').attr('y', 18).attr('width', 12).attr('height', 12).attr('fill', '#4ecdc4');
        legend.append('text').attr('x', 18).attr('y', 29).text('Partition Predicted').style('fill', '#667').style('font-size', '10px');

        svg.append('text').attr('x', width / 2).attr('y', height + 50)
            .text('Base Pair').style('fill', '#667').style('font-size', '11px').attr('text-anchor', 'middle');
        svg.append('text').attr('transform', 'rotate(-90)').attr('x', -height / 2).attr('y', -50)
            .text('\u0394G (kcal/mol)').style('fill', '#667').style('font-size', '11px').attr('text-anchor', 'middle');
    }, []);
    return <div ref={ref} className="d3_chart_container" />;
}

function HelixGeometryChart() {
    const ref = useRef(null);
    useEffect(() => {
        if (!ref.current) return;
        const d3 = require('d3');
        const container = ref.current;
        container.innerHTML = '';
        const margin = { top: 20, right: 20, bottom: 45, left: 65 };
        const width = (container.clientWidth || 400) - margin.left - margin.right;
        const height = 280 - margin.top - margin.bottom;
        const tooltip = ensureTooltip();

        const svg = d3.select(container).append('svg')
            .attr('width', width + margin.left + margin.right)
            .attr('height', height + margin.top + margin.bottom)
            .append('g').attr('transform', `translate(${margin.left},${margin.top})`);

        const data = [];
        for (var angle = 20; angle <= 50; angle += 0.5) {
            var offset = (angle - 36) / 5;
            var energy = 0.5 * offset * offset + 0.1 * Math.sin((angle - 36) * 0.5);
            data.push({ angle: angle, energy: energy });
        }

        const xScale = d3.scaleLinear().domain([20, 50]).range([0, width]);
        const yScale = d3.scaleLinear().domain([0, 5]).range([height, 0]);

        svg.append('g').attr('transform', `translate(0,${height})`)
            .call(d3.axisBottom(xScale).ticks(7))
            .selectAll('text').style('fill', '#667').style('font-size', '10px');
        svg.append('g').call(d3.axisLeft(yScale).ticks(5))
            .selectAll('text').style('fill', '#667').style('font-size', '10px');
        svg.selectAll('.domain, .tick line').style('stroke', '#1a2a3a');

        const area = d3.area().x(function(d) { return xScale(d.angle); }).y0(height).y1(function(d) { return yScale(d.energy); });
        svg.append('path').datum(data)
            .attr('fill', 'rgba(249, 215, 126, 0.08)').attr('d', area)
            .style('opacity', 0).transition().duration(1000).style('opacity', 1);

        const line = d3.line().x(function(d) { return xScale(d.angle); }).y(function(d) { return yScale(d.energy); }).curve(d3.curveBasis);
        const path = svg.append('path').datum(data)
            .attr('fill', 'none').attr('stroke', '#f9d77e').attr('stroke-width', 2.5).attr('d', line);
        const totalLength = path.node().getTotalLength();
        path.attr('stroke-dasharray', totalLength).attr('stroke-dashoffset', totalLength)
            .transition().duration(2000).ease(d3.easeLinear).attr('stroke-dashoffset', 0);

        svg.append('line')
            .attr('x1', xScale(36)).attr('x2', xScale(36)).attr('y1', 0).attr('y2', height)
            .attr('stroke', '#4ecdc4').attr('stroke-dasharray', '5,5')
            .style('opacity', 0).transition().delay(1200).duration(500).style('opacity', 1);
        svg.append('text').attr('x', xScale(36) + 5).attr('y', 14)
            .text('Predicted: 36\u00B0')
            .style('fill', '#4ecdc4').style('font-size', '10px')
            .style('opacity', 0).transition().delay(1400).duration(400).style('opacity', 1);

        svg.append('line')
            .attr('x1', xScale(34.3)).attr('x2', xScale(34.3)).attr('y1', 0).attr('y2', height)
            .attr('stroke', '#ff6b6b').attr('stroke-dasharray', '5,5')
            .style('opacity', 0).transition().delay(1500).duration(500).style('opacity', 1);
        svg.append('text').attr('x', xScale(34.3) - 5).attr('y', 30)
            .text('Observed: 34.3\u00B0')
            .style('fill', '#ff6b6b').style('font-size', '10px').attr('text-anchor', 'end')
            .style('opacity', 0).transition().delay(1700).duration(400).style('opacity', 1);

        const vLine = svg.append('line').attr('y1', 0).attr('y2', height).attr('stroke', '#f9d77e').attr('stroke-width', 0.5).style('opacity', 0);
        const hLine = svg.append('line').attr('x1', 0).attr('x2', width).attr('stroke', '#f9d77e').attr('stroke-width', 0.5).style('opacity', 0);
        const hoverDot = svg.append('circle').attr('r', 4).attr('fill', '#fff').style('opacity', 0);

        svg.append('rect').attr('width', width).attr('height', height).attr('fill', 'transparent')
            .on('mousemove', function(event) {
                var coords = d3.pointer(event);
                var mx = coords[0];
                var a = xScale.invert(mx);
                var off = (a - 36) / 5;
                var e = 0.5 * off * off + 0.1 * Math.sin((a - 36) * 0.5);
                vLine.attr('x1', mx).attr('x2', mx).style('opacity', 0.5);
                hLine.attr('y1', yScale(e)).attr('y2', yScale(e)).style('opacity', 0.5);
                hoverDot.attr('cx', mx).attr('cy', yScale(e)).style('opacity', 1);
                tooltip.innerHTML = '<strong>Twist: ' + a.toFixed(1) + '\u00B0/bp</strong><br/>Energy: ' + e.toFixed(3) + ' a.u.';
                tooltip.style.display = 'block';
                tooltip.style.left = (event.clientX + 12) + 'px';
                tooltip.style.top = (event.clientY - 10) + 'px';
            })
            .on('mouseout', function() {
                vLine.style('opacity', 0); hLine.style('opacity', 0); hoverDot.style('opacity', 0);
                tooltip.style.display = 'none';
            });

        svg.append('text').attr('x', width / 2).attr('y', height + 38)
            .text('Helical Twist Angle (\u00B0/bp)').style('fill', '#667').style('font-size', '11px').attr('text-anchor', 'middle');
        svg.append('text').attr('transform', 'rotate(-90)').attr('x', -height / 2).attr('y', -50)
            .text('Partition Energy (a.u.)').style('fill', '#667').style('font-size', '11px').attr('text-anchor', 'middle');
    }, []);
    return <div ref={ref} className="d3_chart_container" />;
}

export default function Derivation({ ActiveIndex }) {
    const sectionRef = useRef(null);
    const [activeChart, setActiveChart] = useState(0);

    useEffect(() => {
        if (ActiveIndex !== 8 || !sectionRef.current) return;

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
                className={ActiveIndex === 8 ? "cavani_tm_section active animated fadeInUp" : "cavani_tm_section hidden animated"}
                id="derivation_"
            >
                <div className="section_inner">
                    <div className="cavani_tm_about">
                        <div className="cavani_tm_title">
                            <span>Derivation of Nucleic Acid Structure from Partition Operations</span>
                        </div>
                        <div className="paper_meta">
                            <p className="paper_author">Kundai Farai Sachikonye &mdash; Technical University of Munich</p>
                        </div>

                        <div className="scrolly_container">
                            <div className="scrolly_chart_wrapper">
                                <div className="scrolly_chart_inner">
                                    <div className={`scrolly_chart ${activeChart === 0 ? 'visible' : ''}`}>
                                        <div className="chart_title">Four-State Partition System</div>
                                        <PartitionStatesChart />
                                        <p className="chart_caption">Hover each nucleotide to inspect partition properties.</p>
                                    </div>
                                    <div className={`scrolly_chart ${activeChart === 1 ? 'visible' : ''}`}>
                                        <div className="chart_title">Binding Energy: Observed vs Predicted</div>
                                        <BindingEnergyChart />
                                        <p className="chart_caption">Hover bars to compare observed and predicted binding energies.</p>
                                    </div>
                                    <div className={`scrolly_chart ${activeChart === 2 ? 'visible' : ''}`}>
                                        <div className="chart_title">Helical Twist Energy Landscape</div>
                                        <HelixGeometryChart />
                                        <p className="chart_caption">Move cursor to trace energy landscape. Predicted within 5% of observed.</p>
                                    </div>
                                </div>
                            </div>

                            <div className="scrolly_steps">
                                <section className="scrolly_step active" data-chart="0">
                                    <h3 className="paper_section_title">Abstract</h3>
                                    <p>This work derives nucleic acid structure from partition operations on bounded dynamical systems without invoking quantum mechanics or molecular biology. From the Bounded Phase Space Law alone, we prove that electron transport in bounded systems generates four distinguishable partition states corresponding exactly to the nucleotide bases A, T, G, and C.</p>
                                    <p>The <strong>Charge Emergence Theorem</strong> establishes that electric charge emerges from partitioning &mdash; unpartitioned matter has no charge. The <strong>Composition Theorem</strong> shows that binding reduces partition depth, with the deficit released as binding energy. All results derive from two axioms with zero free parameters.</p>
                                </section>

                                <section className="scrolly_step" data-chart="0">
                                    <h3 className="paper_section_title">Four-State Partition System</h3>
                                    <p>Two independent binary partitions on electron transport &mdash; hydrogen-bond donor/acceptor and purine/pyrimidine ring geometry &mdash; generate exactly four distinguishable states. These correspond precisely to the four nucleotide bases. This is not a mapping imposed on biology but a mathematical necessity: bounded electron transport in systems with two independent partition axes must produce exactly four states.</p>
                                </section>

                                <section className="scrolly_step" data-chart="1">
                                    <h3 className="paper_section_title">Binding Energy from Partition Depth</h3>
                                    <p>The Composition Theorem predicts that complementary base pairing reduces the combined partition depth, with the deficit released as free energy. A&ndash;T pairs (2 hydrogen bonds) release &Delta;G &asymp; &minus;1.0 to &minus;1.5 kcal/mol, while G&ndash;C pairs (3 hydrogen bonds) release &Delta;G &asymp; &minus;2.0 to &minus;3.0 kcal/mol. Mismatches produce near-zero partition depth reduction, explaining their thermodynamic instability.</p>
                                </section>

                                <section className="scrolly_step" data-chart="2">
                                    <h3 className="paper_section_title">Double-Helix Geometry</h3>
                                    <p>The helical twist angle emerges from partition stability constraints. The framework predicts an optimal twist of approximately 36&deg; per base pair; B-form DNA exhibits 34.3&deg;/bp (5% relative error). The rise per base pair (&sim;0.34 nm) and the major/minor groove asymmetry also follow from the dual-partition stabilization requirement.</p>
                                </section>

                                <section className="scrolly_step" data-chart="0">
                                    <h3 className="paper_section_title">Key Theorems</h3>
                                    <div className="key_results_grid">
                                        <div className="result_card">
                                            <div className="result_value">Charge Emergence</div>
                                            <div className="result_label">Electric charge emerges from partitioning; unpartitioned matter has no charge</div>
                                        </div>
                                        <div className="result_card">
                                            <div className="result_value">Composition</div>
                                            <div className="result_label">Binding reduces partition depth; deficit released as binding energy</div>
                                        </div>
                                        <div className="result_card">
                                            <div className="result_value">Nucleotide Correspondence</div>
                                            <div className="result_label">Four partition states correspond exactly to A, T, G, C</div>
                                        </div>
                                        <div className="result_card">
                                            <div className="result_value">Complementarity</div>
                                            <div className="result_label">Watson-Crick pairing equals partition completion</div>
                                        </div>
                                    </div>
                                </section>

                                <section className="scrolly_step" data-chart="1">
                                    <h3 className="paper_section_title">Experimental Validation</h3>
                                    <div className="validation_table">
                                        <table>
                                            <thead>
                                                <tr>
                                                    <th>Prediction</th>
                                                    <th>Predicted</th>
                                                    <th>Observed</th>
                                                    <th>Agreement</th>
                                                </tr>
                                            </thead>
                                            <tbody>
                                                <tr><td>DNA capacitance</td><td>&sim;100 pF</td><td>&sim;300 pF</td><td>Order of magnitude</td></tr>
                                                <tr><td>A&ndash;T pairing energy</td><td>&minus;1.2 kcal/mol</td><td>&minus;1.0 to &minus;1.5</td><td>Within range</td></tr>
                                                <tr><td>G&ndash;C pairing energy</td><td>&minus;2.4 kcal/mol</td><td>&minus;2.0 to &minus;3.0</td><td>Within range</td></tr>
                                                <tr><td>Helical twist</td><td>&sim;36&deg;/bp</td><td>34.3&deg;/bp</td><td>5% error</td></tr>
                                                <tr><td>Number of bases</td><td>Exactly 4</td><td>4 (A, T, G, C)</td><td>Exact</td></tr>
                                            </tbody>
                                        </table>
                                    </div>
                                </section>

                                <section className="scrolly_step" data-chart="2">
                                    <h3 className="paper_section_title">Information Density Enhancement</h3>
                                    <p>The cardinal coordinate transformation enables geometric analysis of DNA sequences with information density scaling as I<sub>geometric</sub>/I<sub>linear</sub> = &Theta;(log n). For a sequence of n = 10<sup>7</sup> nucleotides, geometric analysis extracts approximately 11.6 times more information than linear analysis.</p>
                                </section>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </>
    );
}
