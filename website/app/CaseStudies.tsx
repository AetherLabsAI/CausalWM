'use client';
import { useState } from 'react';
import { Tabs, TabsList, TabsTrigger, TabsContent } from '@/components/ui/tabs';
import { Button } from '@/components/ui/button';
import cases from '@/lib/cases.json';
import './CaseStudies.css';

const stages = [
  { key: 'observe', symbol: 'O', name: 'Observe', title: 'Start with the world as it is.', body: 'An observed RGB frame and a language instruction establish the scene and the intended task. The observation remains fixed throughout the reasoning chain.', context: ['Observed frame', 'Language instruction'], output: 'Input observation', color: '#bed1cb' },
  { key: 'flow', symbol: 'F', name: 'Motion', title: 'First, predict what moves.', body: 'Optical flow makes scene motion explicit. The shared diffusion Transformer predicts this stream first, then keeps the completed prediction as context for the next stage.', context: ['Observation O', 'Instruction'], output: 'Predicted optical flow', color: '#9cddc5' },
  { key: 'pointmap', symbol: 'P', name: 'Geometry', title: 'Then, reason about spatial structure.', body: 'Pointmaps provide 3D geometric context. Their prediction can attend to the observation and completed flow, while later RGB tokens remain inaccessible.', context: ['Observation O', 'Predicted flow F'], output: 'Predicted pointmaps', color: '#c6b3e3' },
  { key: 'rgb', symbol: 'V', name: 'Future RGB', title: 'Let the reasoning shape the future.', body: 'Future RGB generation reads both motion and geometry. The intermediate streams stay fixed, giving the model structured context for the final video prediction.', context: ['Observation O', 'Flow F', 'Pointmaps P'], output: 'Generated RGB video', color: '#e5b299' },
];

function CaseVideo({ src, poster, label, width = 640, height = 480 }: { src: string; poster: string; label: string; width?: number; height?: number }) {
  const [failed, setFailed] = useState(false);
  return <>
    <video controls muted playsInline loop preload="none" src={src} poster={poster} width={width} height={height} aria-label={label} onError={() => setFailed(true)}>
      <a href={src}>Open the video</a>
    </video>
    {failed && <p className="case-video-error" role="status">Video unavailable. <a href={src}>Open the video file</a>.</p>}
  </>;
}

export default function CaseStudies() {
  const [caseId, setCaseId] = useState(cases.language[0].id);
  const [stage, setStage] = useState('compare');
  const selected = cases.language.find(c => c.id === caseId)!;
  const base = `./cases/language/${selected.id}`;
  const active = stages.find(s => s.key === stage);

  return <section className="reasoning dark case-studies" id="reasoning"><div className="wrap section">
    <div className="section-heading"><div><p className="eyebrow">03 / CASE STUDIES</p><h2>See the steps<br/><em>between now and next.</em></h2></div><p>Explore generated motion, geometry and future video across three language-conditioned robot tasks.</p></div>
    <div className="case-picker" role="group" aria-label="Choose a language-conditioned case">
      {cases.language.map((c, i) => <Button key={c.id} variant="ghost" className="case-option" aria-pressed={caseId === c.id} onClick={() => setCaseId(c.id)}>
        <img src={`./cases/language/${c.id}/input.jpg`} width="80" height="60" loading="lazy" alt=""/>
        <span><small>0{i + 1} / CASE</small><strong>{c.title}</strong></span>
      </Button>)}
    </div>
    <div className="case-instruction" aria-live="polite"><p className="eyebrow">LANGUAGE INSTRUCTION</p><p>“{selected.prompt}”</p><span>{selected.frames} frames · {selected.fps} fps · 640 × 480 · {selected.steps.join('/')} denoising steps</span></div>
    <Tabs value={stage} onValueChange={v => setStage(String(v))} className="stage-tabs">
      <div className="stage-toolbar"><TabsList className="stage-list" aria-label="Causal reasoning stage">
        <TabsTrigger value="compare" className="compare-stage">Side by side</TabsTrigger>
        {stages.map(s => <TabsTrigger key={s.key} value={s.key}><span className="stage-symbol">{s.symbol}</span><span>{s.name}</span></TabsTrigger>)}
      </TabsList></div>
      <TabsContent value="compare" className="case-comparison">
        <figure className="stage-viewer comparison-viewer"><div className="comparison-labels"><span>01 / Optical flow</span><span>02 / Pointmaps</span><span>03 / Future RGB</span></div>
          <CaseVideo key={`${caseId}-compare`} src={`${base}/panel_flow_pointmap_rgb.mp4`} poster={`${base}/panel_flow_pointmap_rgb.jpg`} label={`${selected.title}: synchronized optical flow, pointmaps and generated RGB`} width={1920} height={480}/>
          <figcaption>The three predicted streams at matching video frames. Use the player to pause, scrub or replay.</figcaption>
        </figure>
      </TabsContent>
      {stages.map((s, i) => <TabsContent key={s.key} value={s.key} className="stage-panel">
        <div className="stage-copy"><p className="eyebrow" style={{ color: s.color }}>STAGE 0{i} / {s.output.toUpperCase()}</p><h3>{s.title}</h3><p>{s.body}</p><div className="context-label">AVAILABLE CONTEXT</div><div className="context-chips">{s.context.map(c => <span key={c}>{c}</span>)}</div><div className="attention-note"><span>One shared diffusion Transformer</span><span>Completed streams remain fixed</span></div></div>
        <figure className="stage-viewer"><div className="viewer-header"><span>{s.output}</span><span>{i === 0 ? 'CONDITIONING FRAME' : '121 FRAMES / 16 FPS'}</span></div>
          {i === 0 ? <img src={`${base}/input.jpg`} width="640" height="480" alt={`Input observation for: ${selected.prompt}`}/> : <CaseVideo key={`${caseId}-${s.key}`} src={`${base}/${s.key}.mp4`} poster={`${base}/${s.key}.jpg`} label={`${selected.title}: ${s.output}`}/>}
          <figcaption>{i === 0 ? 'The supplied first frame conditions the generated sequence.' : `${s.output} · ${selected.title}`}</figcaption>
        </figure>
      </TabsContent>)}
    </Tabs>
    <div className="attention-block"><div><p className="eyebrow">THE INFORMATION FLOW</p><h3>A causal order.<br/>An explicit dependency.</h3><p>Each stream can read the observation and itself or earlier streams. Later-to-earlier paths are blocked; attention within each stream remains bidirectional.</p><p className="tiny-note">The matrix shows stream visibility, not measured attention weights.</p></div><div className="mask-container"><div className="matrix-axis">KEYS / CONTEXT →</div><div className="mask-grid" role="table" aria-label="Stage-ordered attention visibility"><div role="row" className="matrix-row"><span role="columnheader"/>{stages.map(s => <span key={s.key} role="columnheader">{s.symbol}</span>)}</div>{stages.map((s, row) => <div key={s.key} role="row" className={`matrix-row ${stage === s.key ? 'selected-row' : ''}`}><span role="rowheader">{s.symbol}</span>{stages.map((t, col) => <span key={t.key} role="cell" className={`matrix-cell ${col <= row ? 'allowed' : 'blocked'}`} aria-label={`${s.name} reads ${t.name}: ${col <= row ? 'allowed' : 'blocked'}`}>{col <= row ? <span aria-hidden="true">{stage === s.key ? '•' : ''}</span> : <span aria-hidden="true">—</span>}</span>)}</div>)}</div><div className="matrix-legend"><span><i/>Allowed</span><span><i/>Blocked</span><span>{active ? `Active row: ${active.symbol}` : 'O → F → P → V'}</span></div></div></div>
    <div className="triview-case">
      <div className="section-heading"><div><p className="eyebrow">ACTION-CONDITIONED / TRIWORLDBENCH</p><h3>One action.<br/><em>Three viewpoints.</em></h3></div><p>Robot trajectories are rendered as URDF control videos to guide synchronized head and wrist views.</p></div>
      <div className="case-instruction"><p className="eyebrow">TASK / EPISODE 25</p><p>“{cases.triworldbench.prompt}”</p><span>Selected best-of-8 submission · {cases.triworldbench.frames} frames · {cases.triworldbench.fps} fps</span></div>
      <figure className="stage-viewer comparison-viewer"><div className="comparison-labels"><span>Head camera</span><span>Left wrist</span><span>Right wrist</span></div>
        <div className="triview-video"><CaseVideo src="./cases/triworldbench/episode25/comparison.mp4" poster="./cases/triworldbench/episode25/comparison.jpg" label="Episode 25: URDF control input on the top row and generated RGB on the bottom row, in head, left-wrist and right-wrist views" width={960} height={480}/><span className="triview-row-label control-label">URDF CONTROL INPUT</span><span className="triview-row-label output-label">GENERATED RGB</span></div>
        <figcaption>Top: the black-background URDF control supplied to the model. Bottom: the generated views at the same frames. This action-conditioned example uses rendered robot control; the flow and pointmap predictions are shown in the language-conditioned cases above.</figcaption>
      </figure>
    </div>
  </div></section>;
}
