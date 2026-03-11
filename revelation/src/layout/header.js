import React from 'react'

export default function Header({handleOnClick, ActiveIndex}) {

    return (
        <>
            {/* HEADER */}
            <div className="cavani_tm_header">
                <div className="logo">
                    <a href="#"><span className="logo_text">Gospel</span></a>
                </div>
                <div className="menu">
                    <ul className="transition_link">
                        <li onClick={() => handleOnClick(0)}><a className={ActiveIndex === 0 ? "active" : ""}>Home</a></li>
                        <li onClick={() => handleOnClick(1)}><a className={ActiveIndex === 1 ? "active" : ""}>About</a></li>
                        <li onClick={() => handleOnClick(5)}><a className={ActiveIndex === 5 ? "active" : ""}>Dynamics</a></li>
                        <li onClick={() => handleOnClick(6)}><a className={ActiveIndex === 6 ? "active" : ""}>Computing</a></li>
                        <li onClick={() => handleOnClick(8)}><a className={ActiveIndex === 8 ? "active" : ""}>Derivation</a></li>
                        <li onClick={() => handleOnClick(7)}><a className={ActiveIndex === 7 ? "active" : ""}>Framework</a></li>
                        <li onClick={() => handleOnClick(4)}><a className={ActiveIndex === 4 ? "active" : ""}>Contact</a></li>
                    </ul>
                </div>
            </div>
            {/* /HEADER */}

        </>
    )
}
