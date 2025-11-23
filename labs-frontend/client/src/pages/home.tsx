import { ExternalLink } from "lucide-react";
import { useState, useEffect } from "react";


const LinkableCard = ({
  title,
  specs,
  link,
  icon,
  highlight = false,
}: {
  title: string,
  specs: string,
  link: string,
  icon?: string,
  highlight?: boolean,
}) => (
  <a
    href={link}
    target="_blank"
    rel="noopener noreferrer"
    className={`relative border-r border-b border-black p-8 flex flex-col justify-between h-full min-h-[320px] group overflow-hidden transition-none cursor-pointer ${highlight ? 'bg-black text-white hover:invert' : 'bg-white text-black hover:bg-black hover:text-white'
      }`}
    style={{ backgroundColor: highlight ? '#000000' : '#FFFFFF', borderColor: '#0A0A0A' }}
  >
    <div className="relative z-10">
      <div className={`flex justify-between items-start mb-6 pb-4 border-b ${highlight ? 'border-white/40' : 'border-black/20'}`}>
        <div className="flex items-center gap-3 flex-1 pr-4">
          {icon && <img src={icon} alt={`${title} Logo`} className={`h-8 w-8 object-contain ${highlight ? 'invert' : ''}`} />}
          <h3 className="text-xl font-mono font-bold uppercase tracking-tighter">{title}</h3>
        </div>
        <ExternalLink className="w-4 h-4 ml-2 flex-shrink-0 mt-1" strokeWidth={1.5} />
      </div>
      <p className="font-mono text-xs leading-relaxed opacity-80">{specs}</p>
    </div>

    {!highlight && (
      <div className="absolute inset-0 bg-white opacity-0 group-hover:opacity-100 mix-blend-mode-difference pointer-events-none transition-none z-20"></div>
    )}
  </a>
);

const HeroQuote = () => (
  <div className="mb-8">
    <p className="font-mono text-xs uppercase tracking-widest mb-4 opacity-70">The Inspiration — The Original "Idea Factory" (1952)</p>
    <p className="font-serif text-2xl md:text-3xl leading-relaxed italic min-h-[200px]">
      "The vitality of a research organization is only a composite of the spirit of the people in it. It has little to do with buildings or equipment, although indeed these things are important mechanical factors in its existence."
    </p>
  </div>
);

export default function Home() {
  return (
    <div className="min-h-screen bg-white text-black font-serif selection:bg-black selection:text-white" style={{ backgroundColor: '#F5F5F0', color: '#0A0A0A' }}>

      {/* 1. HEADER (Sticky) */}
      <header className="sticky top-0 z-50 border-b flex justify-between items-center px-6 py-4 uppercase font-sans text-sm tracking-wide" style={{ backgroundColor: '#F5F5F0', borderColor: '#0A0A0A' }}>
        <div className="font-bold text-lg flex items-center gap-3">
          SHANNON LABS
        </div>
        <div className="font-mono text-xs flex gap-4">
          <a href="/art" className="hover:underline hover:bg-black hover:text-white px-1 transition-none">[ART, EDUCATION &amp; MUSIC]</a>
          <a href="/about" className="hover:underline hover:bg-black hover:text-white px-1 transition-none">[ABOUT THE FOUNDER]</a>
          <a href="https://github.com/Shannon-Labs" target="_blank" rel="noopener noreferrer" className="hover:underline hover:bg-black hover:text-white px-1 transition-none">[GITHUB]</a>
          <a href="https://twitter.com/huntermbown" target="_blank" rel="noopener noreferrer" className="hover:underline hover:bg-black hover:text-white px-1 transition-none">[TWITTER]</a>
          <a href="https://www.linkedin.com/in/hunterbown/" target="_blank" rel="noopener noreferrer" className="hover:underline hover:bg-black hover:text-white px-1 transition-none">[LINKEDIN]</a>
        </div>
      </header>

      {/* 2. HERO SECTION */}
      <section className="border-b py-32 px-6 md:px-12 flex flex-col items-start justify-center" style={{ backgroundColor: '#F5F5F0', borderColor: '#0A0A0A' }}>
        <h1 className="text-7xl md:text-8xl lg:text-9xl font-serif font-normal leading-none tracking-tight mb-8" style={{ color: '#0A0A0A' }}>
          SHANNON<br />LABS
        </h1>
        <h2 className="text-xl md:text-2xl font-mono font-normal mb-8 uppercase tracking-widest" style={{ color: '#0A0A0A' }}>
          THE NERVOUS SYSTEM FOR THE AGI ERA.
        </h2>
        <div className="font-mono text-sm mb-8 max-w-3xl" style={{ color: '#0A0A0A' }}>
          <span className="font-bold uppercase">Current focus:</span> Driftlock Choir — The temporal synchronization layer for Universal Compute. A wireless timing mesh network enabling the swarm. Because the next frontier isn't just faster code, it's coordinated physics.
        </div>
        <div className="font-mono text-xs uppercase tracking-widest border-t pt-6" style={{ borderColor: '#0A0A0A' }}>
          Est. 2025 | Dallas, TX | Status: Operating
        </div>
      </section>

      {/* 3. THE INSTITUTIONAL LINEAGE */}
      <section className="border-b py-20 px-6 md:px-12 flex justify-center" style={{ backgroundColor: '#F5F5F0', borderColor: '#0A0A0A' }}>
        <article className="max-w-[75ch] w-full" style={{ backgroundColor: '#FFFFFF', color: '#0A0A0A' }}>
          <div className="p-10 md:p-14 border" style={{ borderColor: '#0A0A0A' }}>
            <HeroQuote />
            <div className="text-center font-mono text-sm uppercase tracking-wider border-t pt-6" style={{ color: '#0A0A0A', borderColor: '#0A0A0A' }}>
              — Ralph Bown, <em>Vitality of a Research Institution and How to Maintain It</em> (1952)
            </div>
          </div>
        </article>
      </section>

      {/* 4. SYSTEM GRID: THE 4 PILLARS */}
      <section className="border-b" style={{ backgroundColor: '#0A0A0A', borderColor: '#0A0A0A' }}>
        <div className="px-4 py-3 font-mono text-xs uppercase tracking-widest border-b" style={{ backgroundColor: '#0A0A0A', color: '#F5F5F0', borderColor: '#F5F5F0' }}>
          // Current Research Vectors
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2">
          <LinkableCard
            title="DRIFTLOCK"
            specs="Compression-based anomaly detection. Finding the signal at the edges where probabilistic models break."
            link="https://driftlock.web.app/"
            icon="/driftlock-logo.svg"
          />
          <div className="relative group">
            <LinkableCard
              title="DRIFTLOCK CHOIR"
              specs="Simulation Validated: 91 Femtoseconds. Wireless timing precision rivaling dedicated fiber synchronization. The synchronization layer for Universal Compute."
              link="https://driftlock-choir.pages.dev/"
              icon="/driftlock-choir-logo.svg"
              highlight={true}
            />
          </div>
          <LinkableCard
            title="SHANNON CONTROL UNIT"
            specs="Entropy Stabilization via Closed-Loop PI Control. Independently validated by Tencent researchers (EntroPIC). Priority Date: Sept 02, 2025."
            link="https://github.com/Shannon-Labs/shannon-control-unit"
            icon="/scu-logo.svg"
          />
          <LinkableCard
            title="HEGELION"
            specs="Dialectical Reasoning Engine. Wraps any LLM in a Thesis → Antithesis → Synthesis loop. Available as an MCP server for search-grounded dialectics."
            link="https://github.com/Hmbown/Hegelion"
            icon="/hegelion-logo.svg"
          />
        </div>
      </section>


      {/* 7. THE BOWN PROTOCOL: 1952 vs 2025 */}
      <section className="border-b py-20 px-6 md:px-12 flex justify-center" style={{ backgroundColor: '#0A0A0A', borderColor: '#0A0A0A' }}>
        <div className="max-w-7xl w-full">
          <h2 className="text-2xl md:text-3xl font-serif font-bold mb-12 uppercase text-center" style={{ color: '#F5F5F0' }}>
            Institutional Pillars<br/>
            Vitality and How to Maintain It
          </h2>
          <p className="text-center font-mono text-xs uppercase tracking-widest mb-12" style={{ color: '#F5F5F0', opacity: 0.7 }}>
            The Next Idea Factory
          </p>

          <div className="grid grid-cols-2 gap-8">
            {/* 1952 COLUMN */}
            <div>
              <div className="border-b pb-4 mb-6" style={{ borderColor: '#F5F5F0' }}>
                <h3 className="font-mono text-sm uppercase tracking-widest font-bold mb-2" style={{ color: '#F5F5F0' }}>
                  Vitality of a Research Institution
                </h3>
                <p className="font-mono text-xs uppercase tracking-widest opacity-70" style={{ color: '#F5F5F0' }}>
                  Ralph Bown, 1952
                </p>
              </div>
              <div className="space-y-0 border-l border-r border-t" style={{ borderColor: '#F5F5F0' }}>
                <div className="p-6 border-b" style={{ borderColor: '#F5F5F0', color: '#F5F5F0' }}>
                  <div className="font-mono text-xs uppercase tracking-widest mb-2 opacity-70">[01]</div>
                  <h3 className="font-mono text-sm font-bold mb-2 uppercase tracking-wider">A Human Problem</h3>
                  <p className="font-serif text-sm leading-relaxed opacity-90">"Primarily a problem in human selection, human relations and group spirit."</p>
                </div>
                <div className="p-6 border-b" style={{ borderColor: '#F5F5F0', color: '#F5F5F0' }}>
                  <div className="font-mono text-xs uppercase tracking-widest mb-2 opacity-70">[02]</div>
                  <h3 className="font-mono text-sm font-bold mb-2 uppercase tracking-wider">Technical Objective</h3>
                  <p className="font-serif text-sm leading-relaxed opacity-90">"A well defined but broad technical objective furnishes a rallying point and sharpens decisions."</p>
                </div>
                <div className="p-6 border-b" style={{ borderColor: '#F5F5F0', color: '#F5F5F0' }}>
                  <div className="font-mono text-xs uppercase tracking-widest mb-2 opacity-70">[03]</div>
                  <h3 className="font-mono text-sm font-bold mb-2 uppercase tracking-wider">Freedom & Dignity</h3>
                  <p className="font-serif text-sm leading-relaxed opacity-90">"The freedom and dignity of the individual in the world of science is a paramount principle."</p>
                </div>
                <div className="p-6 border-b" style={{ borderColor: '#F5F5F0', color: '#F5F5F0' }}>
                  <div className="font-mono text-xs uppercase tracking-widest mb-2 opacity-70">[04]</div>
                  <h3 className="font-mono text-sm font-bold mb-2 uppercase tracking-wider">Organizational Structure</h3>
                  <p className="font-serif text-sm leading-relaxed opacity-90">"An orderly organizational structure with room for recognition of a variety of skills is helpful."</p>
                </div>
                <div className="p-6 border-b" style={{ borderColor: '#F5F5F0', color: '#F5F5F0' }}>
                  <div className="font-mono text-xs uppercase tracking-widest mb-2 opacity-70">[05]</div>
                  <h3 className="font-mono text-sm font-bold mb-2 uppercase tracking-wider">Self-Governing Work</h3>
                  <p className="font-serif text-sm leading-relaxed opacity-90">"A program which keeps moving dynamically forward into new ground is the purpose of the whole thing."</p>
                </div>
                <div className="p-6 border-b" style={{ borderColor: '#F5F5F0', color: '#F5F5F0' }}>
                  <div className="font-mono text-xs uppercase tracking-widest mb-2 opacity-70">[06]</div>
                  <h3 className="font-mono text-sm font-bold mb-2 uppercase tracking-wider">Economic Rewards</h3>
                  <p className="font-serif text-sm leading-relaxed opacity-90">"Just and adequate economic rewards are necessary but far from sufficient."</p>
                </div>
              </div>
            </div>

            {/* 2025 COLUMN */}
            <div>
              <div className="border-b pb-4 mb-6" style={{ borderColor: '#F5F5F0' }}>
                <h3 className="font-mono text-sm uppercase tracking-widest font-bold mb-2" style={{ color: '#F5F5F0' }}>
                  Vitality of Humanity
                </h3>
                <p className="font-mono text-xs uppercase tracking-widest opacity-70" style={{ color: '#F5F5F0' }}>
                  Hunter Bown, 2025
                </p>
              </div>
              <div className="space-y-0 border-l border-r border-t" style={{ borderColor: '#F5F5F0' }}>
                <div className="p-6 border-b" style={{ borderColor: '#F5F5F0', color: '#F5F5F0' }}>
                  <div className="font-mono text-xs uppercase tracking-widest mb-2 opacity-70">[01]</div>
                  <h3 className="font-mono text-sm font-bold mb-2 uppercase tracking-wider">A Human Problem</h3>
                  <p className="font-serif text-sm leading-relaxed opacity-90">Building teams of uninhibited polymaths. The Hedy Lamarrs of the AGI era—newly awakened Renaissance minds capable of synthesizing art and architecture.</p>
                </div>
                <div className="p-6 border-b" style={{ borderColor: '#F5F5F0', color: '#F5F5F0' }}>
                  <div className="font-mono text-xs uppercase tracking-widest mb-2 opacity-70">[02]</div>
                  <h3 className="font-mono text-sm font-bold mb-2 uppercase tracking-wider">Technical Objective</h3>
                  <p className="font-serif text-sm leading-relaxed opacity-90">The Next Idea Factory. Infrastructure for human autonomy.</p>
                </div>
                <div className="p-6 border-b" style={{ borderColor: '#F5F5F0', color: '#F5F5F0' }}>
                  <div className="font-mono text-xs uppercase tracking-widest mb-2 opacity-70">[03]</div>
                  <h3 className="font-mono text-sm font-bold mb-2 uppercase tracking-wider">Freedom & Dignity</h3>
                  <p className="font-serif text-sm leading-relaxed opacity-90">Cognitive Sovereignty. The right to pursue the "missing fundamental"—structural signals invisible to probabilistic models. Preserving human intuition against the noise of generative homogeneity.</p>
                </div>
                <div className="p-6 border-b" style={{ borderColor: '#F5F5F0', color: '#F5F5F0' }}>
                  <div className="font-mono text-xs uppercase tracking-widest mb-2 opacity-70">[04]</div>
                  <h3 className="font-mono text-sm font-bold mb-2 uppercase tracking-wider">Organizational Structure</h3>
                  <p className="font-serif text-sm leading-relaxed opacity-90">Flat hierarchies that value theoretical depth AND engineering execution. Complementary skills integrated.</p>
                </div>
                <div className="p-6 border-b" style={{ borderColor: '#F5F5F0', color: '#F5F5F0' }}>
                  <div className="font-mono text-xs uppercase tracking-widest mb-2 opacity-70">[05]</div>
                  <h3 className="font-mono text-sm font-bold mb-2 uppercase tracking-wider">Self-Governing Work</h3>
                  <p className="font-serif text-sm leading-relaxed opacity-90">Matching construction against decay—continuous growth or institutional death. Dynamic forward momentum required.</p>
                </div>
                <div className="p-6 border-b" style={{ borderColor: '#F5F5F0', color: '#F5F5F0' }}>
                  <div className="font-mono text-xs uppercase tracking-widest mb-2 opacity-70">[06]</div>
                  <h3 className="font-mono text-sm font-bold mb-2 uppercase tracking-wider">Economic Rewards</h3>
                  <p className="font-serif text-sm leading-relaxed opacity-90">Inventor-first patent policy. The Inventor retains patent title and majority equity. Incentives aligned through a minority stake and a recursive research license. Launching sovereign companies, not capturing them.</p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* 8. FOOTER */}
      <footer className="py-12 px-6 md:px-12 flex flex-col md:flex-row justify-between items-start md:items-center gap-8" style={{ backgroundColor: '#0A0A0A', color: '#F5F5F0', borderTop: '1px solid #F5F5F0' }}>
        <div>
          <div className="font-mono text-xs space-y-2">
            <p><a href="mailto:hunter@shannonlabs.dev" className="hover:underline">[EMAIL: hunter@shannonlabs.dev]</a></p>
          </div>
        </div>
        <div className="font-mono text-xs uppercase tracking-widest flex gap-6">
          <a href="/art" className="hover:bg-white hover:text-black px-2 py-1 transition-none">[ART, EDUCATION &amp; MUSIC]</a>
          <a href="/about" className="hover:bg-white hover:text-black px-2 py-1 transition-none">[ABOUT THE FOUNDER]</a>
          <a href="https://github.com/Shannon-Labs" target="_blank" rel="noopener noreferrer" className="hover:bg-white hover:text-black px-2 py-1 transition-none">[GITHUB]</a>
          <a href="https://twitter.com/huntermbown" target="_blank" rel="noopener noreferrer" className="hover:bg-white hover:text-black px-2 py-1 transition-none">[TWITTER]</a>
          <a href="https://www.linkedin.com/in/hunterbown/" target="_blank" rel="noopener noreferrer" className="hover:bg-white hover:text-black px-2 py-1 transition-none">[LINKEDIN]</a>
        </div>
      </footer>

    </div>
  );
}
