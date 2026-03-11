import React from 'react'

export default function Footer() {
    return (
        <>
            <footer>
                {/* <!-- FOOTER --> */}
                <div className="cavani_tm_footer">
                    <div className="copyright">
                        <p>Copyright &copy; {new Date().getFullYear()} Gospel Framework &mdash; AIMe Registry &amp; Bitspark GmbH</p>
                    </div>
                    <div className="social">
                        <ul>
                            <li><a href="https://github.com" target="_blank" rel="noopener noreferrer"><img className="svg" src="img/svg/social/github.svg" alt="GitHub" /></a></li>
                        </ul>
                    </div>
                </div>
                {/* <!-- /FOOTER --> */}
            </footer>
        </>
    )
}
