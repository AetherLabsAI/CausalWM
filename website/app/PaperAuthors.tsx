const authors = [
  { name: 'Ziming Xu', marks: '1,†' },
  { name: 'Shuang Liang', marks: '1,2,†,‡' },
  { name: 'Ruobing Han', marks: '1' },
  { name: 'Ziqiao Xi', marks: '1' },
  { name: 'Mingxing Rao', marks: '1,3,‡' },
  { name: 'Kun Zhou', marks: '1,*' },
  { name: 'Zijun Zhang', marks: '1' },
  { name: 'Yuchen Yan', marks: '1' },
  { name: 'Yufan Wei', marks: '1,2,‡' },
  { name: 'Junbo Huang', marks: '1' },
  { name: 'Yifei Shao', marks: '1' },
  { name: 'Fang Nan', marks: '1,2,‡' },
  { name: 'Biwei Huang', marks: '1' },
];

export default function PaperAuthors() {
  return <section className="paper-authors wrap" id="authors" aria-label="Paper authors and affiliations">
    <ul className="paper-author-list">
      {authors.map(author => <li key={author.name}>{author.name}<sup>{author.marks}</sup></li>)}
    </ul>
    <p className="paper-affiliations">
      <span><sup>1</sup> Aether AI</span>
      <span><sup>2</sup> University of California, San Diego</span>
      <span><sup>3</sup> Vanderbilt University</span>
    </p>
    <p className="paper-author-notes">
      <span>† Equal contribution.</span>
      <span>* Corresponding author and project leader.</span>
      <span>‡ Work done during internship at Aether AI.</span>
    </p>
  </section>;
}
