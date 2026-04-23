import { motion } from 'framer-motion'
import Link from 'next/link'
import React from 'react'


let MotionLink = motion(Link);

const Logo = () => {

  return (
    <div
     className='flex flex-col items-center justify-center mt-2'>
        <MotionLink href="/" 
    className='flex items-center justify-center rounded-full w-16 h-16  bg-dark text-white dark:border-2 dark:border-solid dark:border-light
    text-2xl font-bold'
    whileHover={{
      backgroundColor:["#121212", "#B63E96","#58E6D9","#B63E96","#121212"],
      transition:{duration:1.5, repeat: Infinity }
    }}
    >Ω</MotionLink>
    </div>
  )
}

export default Logo