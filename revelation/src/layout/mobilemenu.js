import React, {useEffect} from 'react'
import { dataImage } from '../plugin/plugin'

export default function Mobilemenu({isToggled, handleOnClick}) {
  useEffect(() => {
    dataImage();
  });
    return (
        <>

            {/* MOBILE MENU */}
            <div className={!isToggled ? "cavani_tm_mobile_menu" :  "cavani_tm_mobile_menu opened"} >
                <div className="inner">
                    <div className="wrapper">
                        <div className="avatar">
                            <div className="image" data-img-url="img/about/1.jpg" />
                        </div>
                        <div className="menu_list">
                            <ul className="transition_link">
                                <li onClick={() => handleOnClick(0)}><a href="#home">Home</a></li>
                                <li onClick={() => handleOnClick(1)}><a href="#about">About</a></li>
                                <li onClick={() => handleOnClick(5)}><a href="#dynamics">Dynamics</a></li>
                                <li onClick={() => handleOnClick(6)}><a href="#computing">Computing</a></li>
                                <li onClick={() => handleOnClick(8)}><a href="#derivation">Derivation</a></li>
                                <li onClick={() => handleOnClick(7)}><a href="#framework">Framework</a></li>
                                <li onClick={() => handleOnClick(4)}><a href="#contact">Contact</a></li>
                            </ul>
                        </div>
                        <div className="copyright">
                            <p>Copyright &copy; {new Date().getFullYear()} Gospel Framework</p>
                        </div>
                    </div>
                </div>
            </div>
            {/* /MOBILE MENU */}


        </>
    )
}
