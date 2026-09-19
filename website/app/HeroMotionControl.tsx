'use client';
import {useEffect,useState} from 'react';
import {Play,Pause} from 'lucide-react';
import {Button} from '@/components/ui/button';

export default function HeroMotionControl(){
 const [paused,setPaused]=useState(false);
 useEffect(()=>{document.documentElement.classList.toggle('motion-paused',paused);return()=>document.documentElement.classList.remove('motion-paused')},[paused]);
 return <Button variant="ghost" className="motion-toggle" onClick={()=>setPaused(!paused)} aria-label={paused?'Resume visual animation':'Pause visual animation'}>{paused?<Play size={14}/>:<Pause size={14}/>}<span>{paused?'Resume':'Pause'} motion</span></Button>;
}
