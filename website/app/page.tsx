import { ArrowDown, ArrowUpRight } from 'lucide-react';
import Research from './Research';
import HeroMotionControl from './HeroMotionControl';
import PublicationLinks from './PublicationLinks';
import PaperAuthors from './PaperAuthors';

export default function Home() {
  return <main id="top">
    <a className="skip-link" href="#results">Skip to benchmark</a>
    <header className="site-header wrap">
      <a href="#top" className="wordmark">CAUSAL<span>WM</span><span className="brand-mark" aria-hidden="true">✳</span></a>
      <nav aria-label="Main navigation">
        <a href="#results">Benchmark</a><a href="#method">Framework</a><a href="#reasoning">Case</a><a href="#data">Data</a>
        <span className="nav-tag">AETHER AI · RESEARCH</span>
      </nav>
    </header>
    <section className="paper-hero paper-hero-illustrated wrap" aria-labelledby="hero-title">
      <div className="paper-copy">
      <p className="eyebrow">EMBODIED WORLD MODELS / 2026</p>
      <h1 id="hero-title">Causal<span>WM</span><small className="title-version"> v1</small></h1>
      <p className="paper-subtitle">Causal Chain-of-Thought Reasoning<br className="paper-title-break"/> for Embodied World Model</p>
      <p className="paper-abstract">A 16B embodied world model that makes physical reasoning explicit. CausalWM predicts optical flow, then 3D pointmaps, then future video—using each intermediate prediction as context for the next. Language instructions and robot actions provide complementary ways to guide the imagined future.</p>
      <PublicationLinks/>
      <div className="paper-links">
        <a className="text-link" href="#results">Benchmark results <ArrowDown size={16}/></a>
        <a className="text-link" href="./assets/CausalWM_Connected.pdf?v=4430cd7a8437" target="_blank" rel="noreferrer">Method figure <ArrowUpRight size={16}/></a>
      </div>
      </div>
<div className="hero-diagram" role="group" aria-label="CausalWM predicts optical flow, then pointmaps, then future RGB, reusing each intermediate prediction as context.">
<div className="diagram-coordinate">O → F → P → V<span>CAUSAL REASONING TRAJECTORY</span></div>
<svg className="trajectory" viewBox="0 0 700 620" fill="none" aria-hidden="true"><defs><linearGradient id="stream"><stop stopColor="#169b89"/><stop offset=".5" stopColor="#8d74bf"/><stop offset="1" stopColor="#cc6845"/></linearGradient></defs>{Array.from({length:18},(_,i)=><path key={i} d={`M ${20+i*3} 560 C ${90+i*9} ${460-i*6}, ${40+i*7} ${170+i*6}, ${275+i*5} ${150+i*6} S ${470+i*8} ${570-i*8}, ${650+i*2} 80`} stroke="url(#stream)" strokeWidth=".7" opacity={.13+i*.012}/>)}<path className="signal-path" d="M 50 560 C 180 400, 100 190, 325 205 S 550 480, 670 80" stroke="url(#stream)" strokeWidth="2"/></svg>
{[{id:1,word:'Motion',type:'OPTICAL FLOW',clip:'flow'},{id:2,word:'Geometry',type:'POINTMAPS',clip:'pointmap'},{id:3,word:'Future',type:'RGB VIDEO',clip:'rgb'}].map(s=><div key={s.id} className={`feature-plate plate-${s.id}`}><div className="plate-top"><span>0{s.id} / {s.type}</span><span>↗</span></div><img src={`./cases/language/single_arm_0007/${s.clip}.jpg`} alt="" width="640" height="480"/><div className="plate-bottom">{s.word}<span>{s.id<3?'REUSE AS CONTEXT':'PREDICTED OBSERVATION'}</span></div></div>)}
<p className="diagram-footnote">Flow → Pointmaps → RGB</p><HeroMotionControl/></div>
    </section>
    <PaperAuthors/>
    <Research/>
  </main>;
}
