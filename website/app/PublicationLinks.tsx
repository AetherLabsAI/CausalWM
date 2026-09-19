import { ArrowUpRight } from 'lucide-react';

const resources = [
  { label: 'Paper', href: 'https://openreview.net/pdf?id=3pf4d0EEqm' },
  { label: 'Code', href: 'https://github.com/AetherLabsAI/CausalWM' },
  { label: 'Model weights', href: 'https://huggingface.co/AetherLabs-AI/CausalWM' },
];

export default function PublicationLinks() {
  return <div className="publication-links">
    {resources.map(({ label, href }) => <a key={label} href={href} target="_blank" rel="noopener noreferrer">
      {label}<ArrowUpRight size={15} aria-hidden="true"/>
    </a>)}
  </div>;
}
