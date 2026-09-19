'use client';
import {useEffect,useRef,useState} from 'react';
import {ArrowRight,ArrowDown,Play,Pause} from 'lucide-react';
import {Button} from '@/components/ui/button';
import {Tabs,TabsList,TabsTrigger} from '@/components/ui/tabs';
import {NativeSelect,NativeSelectOption,NativeSelectOptGroup} from '@/components/ui/native-select';
import {Accordion,AccordionItem,AccordionTrigger,AccordionContent} from '@/components/ui/accordion';
import {Table,TableHeader,TableBody,TableRow,TableHead,TableCell,TableFooter} from '@/components/ui/table';
import data from '../lib/training-data.json';

type Metric='source'|'retained';
const fmt=(v:number,d=1)=>v.toLocaleString('en-US',{minimumFractionDigits:d,maximumFractionDigits:d});
const datasets=data.datasets;
const pct=(v:number)=>v<.1&&v>0?'<0.1':v.toFixed(1);
const totals={source:datasets.reduce((s,d)=>s+d.source,0),retained:datasets.reduce((s,d)=>s+d.retained,0)};
function point(r:number,a:number){return [Number((260+r*Math.cos(a)).toFixed(3)),Number((260+r*Math.sin(a)).toFixed(3))];}
function sector(r0:number,r1:number,start:number,end:number){
 const gap=Math.min(.012,(end-start)*.14);start+=gap/2;end-=gap/2;
 const [x0,y0]=point(r1,start),[x1,y1]=point(r1,end),[x2,y2]=point(r0,end),[x3,y3]=point(r0,start);const large=end-start>Math.PI?1:0;
 return `M${x0},${y0} A${r1},${r1} 0 ${large} 1 ${x1},${y1} L${x2},${y2} A${r0},${r0} 0 ${large} 0 ${x3},${y3}Z`;
}
export default function DataSection(){
 const [metric,setMetric]=useState<Metric>('retained');
 const [selected,setSelected]=useState(1);
 const [hovered,setHovered]=useState<number|null>(null);
 const [tour,setTour]=useState(false);
 const [values,setValues]=useState(datasets.map(d=>d.retained));
 const latestValues=useRef(values);
 useEffect(()=>{
  const target=datasets.map(d=>d[metric]);
  if(window.matchMedia('(prefers-reduced-motion: reduce)').matches){latestValues.current=target;setValues(target);return;}
  const from=latestValues.current;let frame=0;let started:number|null=null;
  const tick=(now:number)=>{if(started===null)started=now;const p=Math.min((now-started)/650,1);const ease=1-Math.pow(1-p,3);const next=target.map((n,i)=>from[i]+(n-from[i])*ease);latestValues.current=next;setValues(next);if(p<1)frame=requestAnimationFrame(tick);};
  frame=requestAnimationFrame(tick);return()=>cancelAnimationFrame(frame);
 },[metric]);
 useEffect(()=>{if(!tour)return;const order=datasets.map((d,i)=>({i,v:d[metric]})).sort((a,b)=>b.v-a.v).map(d=>d.i);const timer=setInterval(()=>setSelected(s=>order[(order.indexOf(s)+1)%order.length]),3500);return()=>clearInterval(timer)},[tour,metric]);
 const currentIndex=hovered??selected;const current=datasets[currentIndex];const group=data.groups[current.group];
 const total=values.reduce((a,b)=>a+b,0);let cursor=-Math.PI/2;
 const slices=values.map((v,i)=>{const start=cursor;cursor+=v/total*Math.PI*2;return {i,start,end:cursor};});
 let groupCursor=-Math.PI/2;
 const groups=data.groups.map((g,i)=>{const value=values.reduce((s,v,j)=>s+(datasets[j].group===i?v:0),0);const start=groupCursor;groupCursor+=value/total*Math.PI*2;return {...g,start,end:groupCursor,value};});
 const choose=(i:number)=>{setSelected(i);setHovered(null);setTour(false)};
 return <section className="data-section section wrap" id="data" aria-labelledby="data-heading">
 <p className="eyebrow">04 / THE DATA FOUNDATION</p>
 <div className="section-heading"><h2 id="data-heading">Many embodiments.<br/><em>A shared physical world.</em></h2><p>Human activity, real robots and simulation contribute about 31K source hours, curated into 20K hours of interaction experience.</p></div>
 <div className="data-summary"><div><strong>20</strong><span>Source families</span></div><div><strong>{Math.trunc(data.reportedSourceHours).toLocaleString('en-US')}<span className="hours-fraction">.{data.reportedSourceHours.toFixed(1).split('.')[1]}</span></strong><span>Source hours · ≈31K</span></div><ArrowRight className="summary-arrow" aria-hidden="true"/><div><strong>{Math.trunc(data.reportedRetainedHours).toLocaleString('en-US')}<span>.{data.reportedRetainedHours.toFixed(1).split('.')[1]}</span></strong><span>Retained hours · ≈20K</span></div><div><strong>{(data.reportedRetainedHours/data.reportedSourceHours*100).toFixed(1)}<span>%</span></strong><span>Hours retained</span></div></div>
 <div className="composition-toolbar"><div><h3>Explore the composition.</h3><p>Outer ring: datasets · Inner ring: embodiment groups</p></div><Tabs value={metric} onValueChange={v=>{setMetric(v as Metric);setHovered(null)}}><TabsList className="data-mode" aria-label="Data composition measurement"><TabsTrigger value="source">Before filtering</TabsTrigger><TabsTrigger value="retained">After preparation</TabsTrigger></TabsList></Tabs></div>
 <div className="composition-grid"><div className="composition-chart"><svg viewBox="0 0 520 520" role="group" aria-label={`Data composition by ${metric==='source'?'source':'retained'} hours; choose a dataset from the outer ring or source selector.`}>
 <circle cx="260" cy="260" r="223" fill="none" stroke="#d8e2d8" strokeDasharray="1 8"/>
 {groups.map((g,i)=><path key={g.label} d={sector(121,153,g.start,g.end)} fill={g.color} opacity={current.group===i?.92:.35}><title>{`${g.label}: ${fmt(datasets.reduce((s,d)=>s+(d.group===i?d[metric]:0),0))} hours`}</title></path>)}
 {slices.map(s=>{const ds=datasets[s.i],middle=(s.start+s.end)/2,active=currentIndex===s.i;return <path key={ds.name} d={sector(160,210,s.start,s.end)} fill={data.groups[ds.group].color} opacity={active?1:current.group===ds.group?.67:.35} transform={active?`translate(${(Math.cos(middle)*5).toFixed(3)},${(Math.sin(middle)*5).toFixed(3)})`:'translate(0,0)'} tabIndex={0} role="button" aria-label={`${ds.name}: ${fmt(ds[metric])} ${metric} hours. ${pct(ds[metric]/totals[metric]*100)} percent of the pool.`} aria-pressed={selected===s.i} onClick={()=>choose(s.i)} onPointerEnter={()=>setHovered(s.i)} onPointerLeave={()=>setHovered(null)} onFocus={()=>setHovered(s.i)} onBlur={()=>setHovered(null)} onKeyDown={e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();choose(s.i)}}}><title>{`${ds.name} · ${fmt(ds[metric])} h`}</title></path>})}
 <text x="260" y="218" textAnchor="middle" className="ring-caption">{metric==='source'?'SOURCE HOURS':'RETAINED HOURS'}</text><text x="260" y="269" textAnchor="middle" className="ring-number">{fmt(values[currentIndex])}</text><text x="260" y="301" textAnchor="middle" className="ring-share">{pct(current[metric]/totals[metric]*100)}% of the pool</text><text x="260" y="333" textAnchor="middle" className="ring-index">{String(currentIndex+1).padStart(2,'0')} / 20 SOURCES</text>
 </svg><div className="chart-tour"><span>Hover to explore · click to select</span><Button variant="ghost" aria-pressed={tour} onClick={()=>{setTour(!tour);setHovered(null)}}>{tour?<Pause size={14} className="tour-icon" aria-hidden="true"/>:<Play size={14} className="tour-icon" aria-hidden="true"/>} {tour?'Pause tour':'Tour sources'}</Button></div></div>
 <div className="source-detail"><label htmlFor="dataset-select" className="small-label">CHOOSE A SOURCE</label><NativeSelect id="dataset-select" value={selected} onChange={e=>choose(Number(e.target.value))} className="dataset-select">{data.groups.map((g,i)=><NativeSelectOptGroup key={g.label} label={g.label}>{datasets.map((d,j)=>d.group===i?<NativeSelectOption key={d.name} value={j}>{d.name}</NativeSelectOption>:null)}</NativeSelectOptGroup>)}</NativeSelect>
 <div className="source-content" aria-live={tour?'off':'polite'}><p className="source-group"><i style={{background:group.color}}/>{group.label}</p><h3>{current.name}</h3><div className="source-facts"><div><span>Source hours</span><strong>{fmt(current.source)}</strong></div><ArrowRight size={18} aria-hidden="true"/><div><span>Retained hours</span><strong>{fmt(current.retained)}</strong></div></div><div className="retention-heading"><span>Retention</span><strong>{(current.retained/current.source*100).toFixed(1)}%</strong></div><div className="retention-track" aria-hidden="true"><span style={{width:`${current.retained/current.source*100}%`,background:group.color}}/></div><dl><div><dt>Camera views</dt><dd>{current.views}</dd></div><div><dt>Task coverage</dt><dd>{current.tasks}</dd></div></dl>{current.selectedSubset&&<p className="tiny-note">Source hours refer to the selected local subset.</p>}{current.name==='DROID'&&<p className="tiny-note">DROID data uses the DreamZero-DROID release.</p>}</div>
 </div></div>
 <div className="embodiment-key">{data.groups.map((g,i)=><div key={g.label} className={current.group===i?'is-current':''}><i style={{background:g.color}}/><span>{g.label}</span><strong>{pct(datasets.reduce((s,d)=>s+(d.group===i?d[metric]:0),0)/totals[metric]*100)}%</strong></div>)}</div>
 <p className="data-footnote">Synchronized camera views of the same trajectory are counted once. Totals follow the manuscript’s data table. Chart shares use the displayed rows, which sum to {fmt(totals.source)} source hours and {fmt(totals.retained)} retained hours; each differs from the reported total by 0.1 hour.</p>
 <Accordion className="data-inventory"><AccordionItem value="inventory"><AccordionTrigger>All 20 sources · hours, embodiments and task coverage</AccordionTrigger><AccordionContent><div className="inventory-download"><span>Values from the manuscript’s data inventory.</span><a href="./data/training-data.csv" download>Download CSV <ArrowDown size={15}/></a></div><Table><TableHeader><TableRow><TableHead>Dataset</TableHead><TableHead className="numeric">Source (h)</TableHead><TableHead className="numeric">Retained (h)</TableHead><TableHead>Embodiment</TableHead><TableHead>Views</TableHead><TableHead>Task coverage</TableHead></TableRow></TableHeader><TableBody>{datasets.map(d=><TableRow key={d.name}><TableCell>{d.name}{d.selectedSubset?' †':''}</TableCell><TableCell className="numeric">{fmt(d.source)}</TableCell><TableCell className="numeric">{fmt(d.retained)}</TableCell><TableCell>{data.groups[d.group].label}</TableCell><TableCell>{d.views}</TableCell><TableCell>{d.tasks}</TableCell></TableRow>)}</TableBody><TableFooter><TableRow><TableCell>Manuscript total</TableCell><TableCell className="numeric">{fmt(data.reportedSourceHours)}</TableCell><TableCell className="numeric">{fmt(data.reportedRetainedHours)}</TableCell><TableCell colSpan={3}>20 source families</TableCell></TableRow></TableFooter></Table><p className="tiny-note">† Source hours refer to selected subsets of OXE and RoVid-X. DROID uses the DreamZero-DROID release. Synchronized views are counted once per trajectory.</p></AccordionContent></AccordionItem></Accordion>
 <div className="source-domains"><article><span>01 / HUMAN EGOCENTRIC</span><h3>Everyday interaction.</h3><p>Daily activity, tool use and dexterous hand–object interaction across human viewpoints.</p><small>Ego4D · Egocentric-10K · EgoDex · EPIC-Kitchens · H2O · EgoVerse</small></article><article><span>02 / REAL ROBOTS</span><h3>Diverse embodiments.</h3><p>Single-arm, bimanual, mobile-manipulation and humanoid platforms with varied cameras and end effectors.</p><small>AgiBot · Galaxea · RoboCOIN · RoboMIND · DROID · RT-1 · BridgeData V2 · OXE · Humanoid-Everyday · RoVid-X</small></article><article><span>03 / SIMULATION</span><h3>Broader physical coverage.</h3><p>Additional objects, trajectories, scenes and embodiments with explicit control and annotation.</p><small>InternData-A1 · RoboCasa365 · RoboTwin 2.0 · AgiBot-World 2026 digital twins</small></article></div>
 <div className="data-curation"><div><p className="eyebrow">FROM SOURCE VIDEOS TO TRAINING CLIPS</p><h3>Curate the interaction.<br/><em>Clarify the instruction.</em></h3><p>Processing routes adapt to each source. Clean recordings may bypass individual filters; suitable native annotations are normalized and reused.</p></div><ol>{[{title:'Validate duration',text:'Check frame rates, timestamps and usable clip length.'},{title:'Screen motion',text:'Apply source-specific flow thresholds for excessive or insufficient motion.'},{title:'Check the action',text:'Keep recognizable actors and task-relevant physical interactions.'},{title:'Segment & recaption',text:'Build contiguous events with concise, action-focused instructions.'}].map((s,i)=><li key={s.title}><span>{String(i+1).padStart(2,'0')}</span><div><h4>{s.title}</h4><p>{s.text}</p></div></li>)}</ol></div>
 </section>;
}
