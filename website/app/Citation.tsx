'use client';
import { useState } from 'react';
import { ArrowDown, Check, Copy } from 'lucide-react';
import citation from '../public/data/citation.bib?raw';

export default function Citation() {
  const [status, setStatus] = useState<'idle' | 'copied' | 'error'>('idle');

  async function copyCitation() {
    try {
      await navigator.clipboard.writeText(citation);
      setStatus('copied');
    } catch {
      setStatus('error');
    }
  }

  return <section className="citation" id="citation" aria-labelledby="citation-heading">
    <div className="citation-heading">
      <h3 id="citation-heading">Cite this work</h3>
      <button type="button" className="citation-copy" onClick={copyCitation}>
        {status === 'copied' ? <Check size={16} aria-hidden="true"/> : <Copy size={16} aria-hidden="true"/>}
        <span aria-live="polite">{status === 'copied' ? 'Copied!' : 'Copy BibTeX'}</span>
      </button>
    </div>
    <pre tabIndex={0} aria-label="BibTeX citation"><code>{citation}</code></pre>
    {status === 'error' && <p className="citation-feedback" role="status">Could not copy. Select the text above or download the citation below.</p>}
    <a className="text-link" href="./data/citation.bib" download="causalwm.bib">Download citation <ArrowDown size={15} aria-hidden="true"/></a>
  </section>;
}
