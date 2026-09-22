import { ArrowUpRight } from 'lucide-react';

const resources = [
  { label: 'Paper', href: 'https://arxiv.org/pdf/2609.23184' },
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
