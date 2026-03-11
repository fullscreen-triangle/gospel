import React, { useState, useEffect } from 'react'
import { customCursor } from '../../plugin/plugin';

export default function ContactDefault({ ActiveIndex }) {
    useEffect(() => {
        customCursor();
    });

    const [form, setForm] = useState({ email: "", name: "", msg: "", organization: "" });
    const [active, setActive] = useState(null);
    const [error, setError] = useState(false);
    const [success, setSuccess] = useState(false);
    const onChange = (e) => {
        setForm({ ...form, [e.target.name]: e.target.value });
    };
    const { email, name, msg, organization } = form;
    const onSubmit = (e) => {
        e.preventDefault();
        if (email && name && msg) {
            setSuccess(true);
            setTimeout(() => {
                setForm({ email: "", name: "", msg: "", organization: "" });
                setSuccess(false);
            }, 2000);
        } else {
            setError(true);
            setTimeout(() => {
                setError(false);
            }, 2000);
        }
    };
    return (
        <>
            <div className={ActiveIndex === 4 ? "cavani_tm_section active animated fadeInUp" : "cavani_tm_section hidden animated"} id="contact_">
                <div className="section_inner">
                    <div className="cavani_tm_contact">
                        <div className="cavani_tm_title">
                            <span>Collaborate &amp; Fund</span>
                        </div>

                        <div className="paper_section">
                            <p>We welcome collaboration from researchers, institutions, and industry partners interested in advancing partition-theory-based genomics. Whether you are exploring novel computational approaches to genomic analysis, seeking a framework for pharmacogenomic or nutrigenomic applications, or interested in funding groundbreaking research at the intersection of physics and biology, we would be glad to hear from you.</p>
                        </div>

                        <div className="short_info">
                            <ul>
                                <li>
                                    <div className="list_inner">
                                        <i className="icon-location"></i>
                                        <div>
                                            <strong>Bitspark GmbH</strong><br />
                                            <span>Kleestra&szlig;e 21-23, 90461 N&uuml;rnberg, Germany</span>
                                        </div>
                                    </div>
                                </li>
                                <li>
                                    <div className="list_inner">
                                        <i className="icon-mail-3"></i>
                                        <div>
                                            <strong>AIMe Registry</strong><br />
                                            <span><a href="mailto:kundai.sachikonye@wzw.tum.de">kundai.sachikonye@wzw.tum.de</a></span>
                                        </div>
                                    </div>
                                </li>
                                <li>
                                    <div className="list_inner">
                                        <i className="icon-mobile"></i>
                                        <div>
                                            <strong>Affiliation</strong><br />
                                            <span>Technical University of Munich (TUM)</span>
                                        </div>
                                    </div>
                                </li>
                            </ul>
                        </div>

                        <div className="funding_section">
                            <div className="cavani_tm_title">
                                <span>Funding Opportunities</span>
                            </div>
                            <div className="funding_grid">
                                <div className="funding_card">
                                    <h4>Research Grants</h4>
                                    <p>Support fundamental research in partition theory applied to genomics, charge dynamics in nucleic acids, and ternary computing architectures for biological systems.</p>
                                </div>
                                <div className="funding_card">
                                    <h4>Industry Partnerships</h4>
                                    <p>Collaborate on applying the Gospel framework to clinical pharmacogenomics, personalized nutrition, and fitness genomics applications.</p>
                                </div>
                                <div className="funding_card">
                                    <h4>Computational Infrastructure</h4>
                                    <p>Help scale the Rust high-performance core for whole-genome analysis, enabling real-time partition-based genomic processing at population scale.</p>
                                </div>
                            </div>
                        </div>

                        <div className="form">
                            <div className="left" style={{width: '100%'}}>
                                <div className="cavani_tm_title">
                                    <span>Get in Touch</span>
                                </div>
                                <div className="fields">
                                    <form className="contact_form" onSubmit={(e) => onSubmit(e)}>
                                        <div
                                            className="returnmessage"
                                            style={{ display: success ? "block" : "none" }}
                                        >
                                            <span className="contact_success">
                                                Thank you for your interest. We will respond shortly.
                                            </span>
                                        </div>
                                        <div
                                            className="empty_notice"
                                            style={{ display: error ? "block" : "none" }}
                                        >
                                            <span>Please fill in all required fields.</span>
                                        </div>

                                        <div className="fields">
                                            <ul>
                                                <li
                                                    className={`input_wrapper ${active === "name" || name ? "active" : ""}`}
                                                >
                                                    <input
                                                        onFocus={() => setActive("name")}
                                                        onBlur={() => setActive(null)}
                                                        onChange={(e) => onChange(e)}
                                                        value={name}
                                                        name="name"
                                                        id="name"
                                                        type="text"
                                                        placeholder="Name"
                                                    />
                                                </li>
                                                <li
                                                    className={`input_wrapper ${active === "email" || email ? "active" : ""}`}
                                                >
                                                    <input
                                                        onFocus={() => setActive("email")}
                                                        onBlur={() => setActive(null)}
                                                        onChange={(e) => onChange(e)}
                                                        value={email}
                                                        name="email"
                                                        id="email"
                                                        type="email"
                                                        placeholder="Email"
                                                    />
                                                </li>
                                                <li
                                                    className={`input_wrapper ${active === "organization" || organization ? "active" : ""}`}
                                                >
                                                    <input
                                                        onFocus={() => setActive("organization")}
                                                        onBlur={() => setActive(null)}
                                                        onChange={(e) => onChange(e)}
                                                        value={organization}
                                                        name="organization"
                                                        id="organization"
                                                        type="text"
                                                        placeholder="Organization / Institution"
                                                    />
                                                </li>
                                                <li
                                                    className={`last ${active === "message" || msg ? "active" : ""}`}
                                                >
                                                    <textarea
                                                        onFocus={() => setActive("message")}
                                                        onBlur={() => setActive(null)}
                                                        name="msg"
                                                        onChange={(e) => onChange(e)}
                                                        value={msg}
                                                        id="message"
                                                        placeholder="Describe your research interest or collaboration proposal"
                                                    />
                                                </li>
                                            </ul>
                                            <div className="cavani_tm_button">
                                                <input
                                                    className='a'
                                                    type="submit"
                                                    id="send_message"
                                                    value="Send Message"
                                                />
                                            </div>
                                        </div>
                                    </form>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </>
    )
}
