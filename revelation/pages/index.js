import dynamic from "next/dynamic";
import React, { useState } from "react";
import HomeDefault from "../src/components/home/home-default";
import Layout from "../src/layout/layout";
import ContactDefault from "../src/components/contact/contact-default";
import Header from "../src/layout/header";
import LeftRightBar from "../src/layout/left-right-bar";
import Mobilemenu from "../src/layout/mobilemenu";
import Modalbox from "../src/layout/modalbox";
import TopBar from "../src/layout/top-bar";

const GLBViewer = dynamic(
  () => import("../src/components/model/GLBViewer"),
  { ssr: false }
);
const Dynamics = dynamic(
  () => import("../src/components/dynamics/dynamics"),
  { ssr: false }
);
const Computing = dynamic(
  () => import("../src/components/computing/computing"),
  { ssr: false }
);
const Derivation = dynamic(
  () => import("../src/components/derivation/derivation"),
  { ssr: false }
);

export default function Home() {
  const [ActiveIndex, setActiveIndex] = useState(0);
  const handleOnClick = (index) => {
    setActiveIndex(index);
  };

  const [isToggled, setToggled] = useState(false);
  const toggleTrueFalse = () => setToggled(!isToggled);

  return (
    <>
      <Layout>
        <Modalbox />
        <Header handleOnClick={handleOnClick} ActiveIndex={ActiveIndex} />
        <LeftRightBar />
        <TopBar toggleTrueFalse={toggleTrueFalse} isToggled={isToggled} />
        <Mobilemenu toggleTrueFalse={toggleTrueFalse} isToggled={isToggled} handleOnClick={handleOnClick} />

        {/* <!-- MAINPART --> */}
        <div className="cavani_tm_mainpart">


         <GLBViewer />

          <div className="main_content">
            <HomeDefault ActiveIndex={ActiveIndex} handleOnClick={handleOnClick} />


            <Dynamics ActiveIndex={ActiveIndex} />

            <Computing ActiveIndex={ActiveIndex} />

            <Derivation ActiveIndex={ActiveIndex} />

            <ContactDefault ActiveIndex={ActiveIndex} />
          </div>
        </div>
        {/* MAINPART */}
      </Layout>
    </>
  );
}
