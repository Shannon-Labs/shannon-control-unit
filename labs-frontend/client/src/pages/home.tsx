import { ExternalLink } from "lucide-react";


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
          <h3 className="text-2xl font-bold uppercase font-sans tracking-tight">{title}</h3>
        </div>
        <ExternalLink className="w-4 h-4 ml-2 flex-shrink-0 mt-1" strokeWidth={1.5} />
      </div>
      <p className="font-serif text-base leading-relaxed">{specs}</p>
    </div>

    {!highlight && (
      <div className="absolute inset-0 bg-white opacity-0 group-hover:opacity-100 mix-blend-mode-difference pointer-events-none transition-none z-20"></div>
    )}
  </a>
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
          THE INFRASTRUCTURE FOR THE NEXT IDEA FACTORY.
        </h2>
        <div className="font-mono text-sm mb-8 max-w-3xl" style={{ color: '#0A0A0A' }}>
          <span className="font-bold uppercase">Current focus:</span> Driftlock Choir — a truly wireless distributed timing mesh network, staking a claim for human ingenuity in the age of AGI.
        </div>
        <div className="font-mono text-xs uppercase tracking-widest border-t pt-6" style={{ borderColor: '#0A0A0A' }}>
          Est. 2025 | Dallas, TX | Status: Operating
        </div>
      </section>

      {/* 3. THE INSTITUTIONAL LINEAGE */}
      <section className="border-b py-20 px-6 md:px-12 flex justify-center" style={{ backgroundColor: '#F5F5F0', borderColor: '#0A0A0A' }}>
        <article className="max-w-[75ch] w-full" style={{ backgroundColor: '#FFFFFF', color: '#0A0A0A' }}>
          <div className="p-10 md:p-14 border" style={{ borderColor: '#0A0A0A' }}>
            <div className="mb-8">
              <p className="font-mono text-xs uppercase tracking-widest mb-4 opacity-70">The Inspiration — The Original "Idea Factory" (1952)</p>
              <p className="font-serif text-2xl md:text-3xl leading-relaxed italic">
                "To maintain vitality implies a dynamic process of continuous growth in which a steady state is achieved only by matching construction against decay."
              </p>
            </div>
            <div className="text-center font-mono text-sm uppercase tracking-wider border-t pt-6" style={{ color: '#0A0A0A', borderColor: '#0A0A0A' }}>
              — Ralph Bown, Vice-President of Research, Bell Labs
            </div>
            <div className="text-center font-serif text-lg italic mt-8" style={{ color: '#0A0A0A' }}>
              Ralph Bown announced the transistor (1948) and later defined the <span className="font-bold">Vitality of a Research Institution</span>.<br />We are announcing the protocol for what comes next: <span className="font-bold">The Vitality of Humanity.</span>
            </div>
          </div>
        </article>
      </section>

      {/* 4. SYSTEM GRID: THE 4 PILLARS */}
      <section className="border-b" style={{ backgroundColor: '#0A0A0A', borderColor: '#0A0A0A' }}>
        <div className="px-4 py-3 font-mono text-xs uppercase tracking-widest border-b" style={{ backgroundColor: '#0A0A0A', color: '#F5F5F0', borderColor: '#F5F5F0' }}>
          // The 4 Pillars of Sovereignty
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2">
          <LinkableCard
            title="SHANNON CONTROL UNIT"
            specs="Entropy Stabilization via Closed-Loop PI Control. Validated at 1B/3B scales. Priority Date: Sept 02, 2025."
            link="https://github.com/Shannon-Labs/shannon-control-unit"
          />
          <LinkableCard
            title="HEGELION"
            specs="Dialectical Reasoning Engine. Recursive Thesis → Antithesis → Synthesis. Structuring thought beyond prediction."
            link="https://github.com/Hmbown/Hegelion"
          />
          <LinkableCard
            title="DRIFTLOCK"
            specs="Compression-based anomaly detection. Deterministic entropy engine that detects data drift without training data."
            link="https://driftlock.web.app/"
            icon="/driftlock-logo.svg"
          />
          <div className="relative group">
            <LinkableCard
              title="DRIFTLOCK CHOIR"
              specs="Chronometric interferometry. 2025 extension of Ralph Bown’s 1948 patent. Wireless sync at ~90fs precision. Current objective: bench validation of the hardware layer at scale — towards a truly wireless distributed timing mesh network."
              link="https://driftlock-choir.pages.dev/"
              icon="/driftlock-choir-logo.svg"
              highlight={true}
            />

          </div>
        </div>
      </section>


      {/* 6. RESEARCH VECTORS */}
      <section className="border-b py-20 px-6 md:px-12 flex justify-center" style={{ backgroundColor: '#F5F5F0', borderColor: '#0A0A0A' }}>
        <div className="max-w-7xl w-full">
          <h2 className="text-2xl md:text-3xl font-serif font-bold mb-12 uppercase text-center" style={{ color: '#0A0A0A' }}>
            Research Vectors
          </h2>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-20">
            {/* VECTORS - Left Column */}
            <div className="space-y-6">
              {/* [01] SCU */}
              <a
                href="https://github.com/Shannon-Labs/shannon-control-unit"
                target="_blank"
                rel="noopener noreferrer"
                className="block p-6 border bg-white text-black hover:bg-black hover:text-white transition-colors group"
                style={{ borderColor: '#0A0A0A' }}
              >
                <div className="flex justify-between items-start mb-2">
                  <div className="font-mono text-xs uppercase tracking-widest opacity-70 group-hover:opacity-100">[01]</div>
                  <img src="/scu-logo.svg" alt="SCU" className="w-8 h-8 object-contain group-hover:invert" />
                </div>
                <h3 className="font-bold text-lg mb-2">Entropy Stabilization</h3>
                <p className="font-serif text-sm leading-relaxed">Control theory applied to model complexity. Replacing static schedules with dynamic feedback loops.</p>
              </a>

              {/* [03] Driftlock */}
              <a
                href="https://driftlock.web.app/"
                target="_blank"
                rel="noopener noreferrer"
                className="block p-6 border bg-white text-black hover:bg-black hover:text-white transition-colors group"
                style={{ borderColor: '#0A0A0A' }}
              >
                <div className="flex justify-between items-start mb-2">
                  <div className="font-mono text-xs uppercase tracking-widest opacity-70 group-hover:opacity-100">[03]</div>
                  <img src="/driftlock-logo.svg" alt="Driftlock" className="w-8 h-8 object-contain group-hover:invert" />
                </div>
                <h3 className="font-bold text-lg mb-2">Anomaly Detection</h3>
                <p className="font-serif text-sm leading-relaxed">Compression-based drift detection without training data. Understanding outliers, not suppressing them.</p>
              </a>

              {/* [05] Institutional Vitality */}
              <div className="p-6 border bg-white text-black group" style={{ backgroundColor: '#FFFFFF', borderColor: '#0A0A0A', color: '#0A0A0A' }}>
                <div className="flex justify-between items-start mb-2">
                  <div className="font-mono text-xs uppercase tracking-widest opacity-70">[05]</div>
                  <img src="/shannon-logo.png" alt="Shannon Labs" className="w-8 h-8 object-contain" />
                </div>
                <h3 className="font-bold text-lg mb-2">Institutional Vitality</h3>
                <p className="font-serif text-sm leading-relaxed">Building research institutions that maintain vitality through continuous growth and decay-matching construction.</p>
              </div>
            </div>

            {/* VECTORS - Right Column */}
            <div className="space-y-6">
              {/* [02] Hegelion */}
              <a
                href="https://github.com/Hmbown/Hegelion"
                target="_blank"
                rel="noopener noreferrer"
                className="block p-6 border bg-white text-black hover:bg-black hover:text-white transition-colors group"
                style={{ borderColor: '#0A0A0A' }}
              >
                <div className="flex justify-between items-start mb-2">
                  <div className="font-mono text-xs uppercase tracking-widest opacity-70 group-hover:opacity-100">[02]</div>
                  <img src="/hegelion-logo.svg" alt="Hegelion" className="w-8 h-8 object-contain group-hover:invert" />
                </div>
                <h3 className="font-bold text-lg mb-2">Dialectical Reasoning</h3>
                <p className="font-serif text-sm leading-relaxed">Moving beyond next-token prediction to structural thought. Thesis → Antithesis → Synthesis recursively applied.</p>
              </a>

              {/* [04] Driftlock Choir */}
              <a
                href="https://driftlock-choir.pages.dev/"
                target="_blank"
                rel="noopener noreferrer"
                className="block p-6 border bg-white text-black hover:bg-black hover:text-white transition-colors group"
                style={{ borderColor: '#0A0A0A' }}
              >
                <div className="flex justify-between items-start mb-2">
                  <div className="font-mono text-xs uppercase tracking-widest opacity-70 group-hover:opacity-100">[04]</div>
                  <img src="/driftlock-choir-logo.svg" alt="Choir" className="w-8 h-8 object-contain group-hover:invert" />
                </div>
                <h3 className="font-bold text-lg mb-2">Chronometric Interferometry</h3>
                <p className="font-serif text-sm leading-relaxed">Hardware-layer timing synchronization via wireless carriers. Achieving fiber-grade precision without fiber.</p>
              </a>

              {/* [06] Human Vitality */}
              <div className="p-6 border bg-white text-black group" style={{ backgroundColor: '#FFFFFF', borderColor: '#0A0A0A', color: '#0A0A0A' }}>
                <div className="flex justify-between items-start mb-2">
                  <div className="font-mono text-xs uppercase tracking-widest opacity-70">[06]</div>
                  <img src="/vitality-logo.svg" alt="Human Vitality" className="w-8 h-8 object-contain" />
                </div>
                <h3 className="font-bold text-lg mb-2">Human Vitality</h3>
                <p className="font-serif text-sm leading-relaxed">Ensuring we remain high on the cognitive food chain. Building systems that extend human intent rather than replace it.</p>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* 7. THE BOWN PROTOCOL: 1952 vs 2025 */}
      <section className="border-b py-20 px-6 md:px-12 flex justify-center" style={{ backgroundColor: '#0A0A0A', borderColor: '#0A0A0A' }}>
        <div className="max-w-7xl w-full">
          <h2 className="text-2xl md:text-3xl font-serif font-bold mb-12 uppercase text-center" style={{ color: '#F5F5F0' }}>
            The Vitality of Humanity — And How to Maintain It
          </h2>
          <p className="text-center font-mono text-xs uppercase tracking-widest mb-12" style={{ color: '#F5F5F0', opacity: 0.7 }}>
            The Next Idea Factory
          </p>

          <div className="grid grid-cols-2 gap-8">
            {/* 1952 COLUMN */}
            <div>
              <h3 className="font-mono text-sm uppercase tracking-widest mb-6 font-bold border-b pb-4" style={{ borderColor: '#F5F5F0', color: '#F5F5F0' }}>
                Vitality of a Research Institution
              </h3>
              <div className="space-y-4">
                <div className="p-6 border" style={{ backgroundColor: '#F5F5F0', borderColor: '#F5F5F0', color: '#0A0A0A' }}>
                  <div className="font-mono text-xs uppercase tracking-widest mb-2 opacity-70">[01]</div>
                  <h3 className="font-bold text-lg mb-2">A Human Problem</h3>
                  <p className="font-serif text-sm leading-relaxed">"Primarily a problem in human selection, human relations and group spirit."</p>
                </div>
                <div className="p-6 border" style={{ backgroundColor: '#F5F5F0', borderColor: '#F5F5F0', color: '#0A0A0A' }}>
                  <div className="font-mono text-xs uppercase tracking-widest mb-2 opacity-70">[02]</div>
                  <h3 className="font-bold text-lg mb-2">Technical Objective</h3>
                  <p className="font-serif text-sm leading-relaxed">"A well defined but broad technical objective furnishes a rallying point and sharpens decisions."</p>
                </div>
                <div className="p-6 border" style={{ backgroundColor: '#F5F5F0', borderColor: '#F5F5F0', color: '#0A0A0A' }}>
                  <div className="font-mono text-xs uppercase tracking-widest mb-2 opacity-70">[03]</div>
                  <h3 className="font-bold text-lg mb-2">Freedom & Dignity</h3>
                  <p className="font-serif text-sm leading-relaxed">"The freedom and dignity of the individual in the world of science is a paramount principle."</p>
                </div>
                <div className="p-6 border" style={{ backgroundColor: '#F5F5F0', borderColor: '#F5F5F0', color: '#0A0A0A' }}>
                  <div className="font-mono text-xs uppercase tracking-widest mb-2 opacity-70">[04]</div>
                  <h3 className="font-bold text-lg mb-2">Organizational Structure</h3>
                  <p className="font-serif text-sm leading-relaxed">"An orderly organizational structure with room for recognition of a variety of skills is helpful."</p>
                </div>
                <div className="p-6 border" style={{ backgroundColor: '#F5F5F0', borderColor: '#F5F5F0', color: '#0A0A0A' }}>
                  <div className="font-mono text-xs uppercase tracking-widest mb-2 opacity-70">[05]</div>
                  <h3 className="font-bold text-lg mb-2">Self-Governing Work</h3>
                  <p className="font-serif text-sm leading-relaxed">"A program which keeps moving dynamically forward into new ground is the purpose of the whole thing."</p>
                </div>
                <div className="p-6 border" style={{ backgroundColor: '#F5F5F0', borderColor: '#F5F5F0', color: '#0A0A0A' }}>
                  <div className="font-mono text-xs uppercase tracking-widest mb-2 opacity-70">[06]</div>
                  <h3 className="font-bold text-lg mb-2">Economic Rewards</h3>
                  <p className="font-serif text-sm leading-relaxed">"Just and adequate economic rewards are necessary but far from sufficient."</p>
                </div>
              </div>
            </div>

            {/* 2025 COLUMN */}
            <div>
              <h3 className="font-mono text-sm uppercase tracking-widest mb-6 font-bold border-b pb-4" style={{ borderColor: '#F5F5F0', color: '#F5F5F0' }}>
                Vitality of Humanity
              </h3>
              <div className="space-y-4">
                <div className="p-6 border" style={{ backgroundColor: '#F5F5F0', borderColor: '#F5F5F0', color: '#0A0A0A' }}>
                  <div className="font-mono text-xs uppercase tracking-widest mb-2 opacity-70">[01]</div>
                  <h3 className="font-bold text-lg mb-2">A Human Problem</h3>
                  <p className="font-serif text-sm leading-relaxed">Building teams of uninhibited polymaths. The Hedy Lamarrs of the AGI era—newly awakened Renaissance minds capable of synthesizing art and architecture.</p>
                </div>
                <div className="p-6 border" style={{ backgroundColor: '#F5F5F0', borderColor: '#F5F5F0', color: '#0A0A0A' }}>
                  <div className="font-mono text-xs uppercase tracking-widest mb-2 opacity-70">[02]</div>
                  <h3 className="font-bold text-lg mb-2">Technical Objective</h3>
                  <p className="font-serif text-sm leading-relaxed">The Next Idea Factory. Infrastructure for human autonomy.</p>
                </div>
                <div className="p-6 border" style={{ backgroundColor: '#F5F5F0', borderColor: '#F5F5F0', color: '#0A0A0A' }}>
                  <div className="font-mono text-xs uppercase tracking-widest mb-2 opacity-70">[03]</div>
                  <h3 className="font-bold text-lg mb-2">Freedom & Dignity</h3>
                  <p className="font-serif text-sm leading-relaxed">Cognitive Sovereignty. The right to pursue the "missing fundamental"—structural signals invisible to probabilistic models. Preserving human intuition against the noise of generative homogeneity.</p>
                </div>
                <div className="p-6 border" style={{ backgroundColor: '#F5F5F0', borderColor: '#F5F5F0', color: '#0A0A0A' }}>
                  <div className="font-mono text-xs uppercase tracking-widest mb-2 opacity-70">[04]</div>
                  <h3 className="font-bold text-lg mb-2">Organizational Structure</h3>
                  <p className="font-serif text-sm leading-relaxed">Flat hierarchies that value theoretical depth AND engineering execution. Complementary skills integrated.</p>
                </div>
                <div className="p-6 border" style={{ backgroundColor: '#F5F5F0', borderColor: '#F5F5F0', color: '#0A0A0A' }}>
                  <div className="font-mono text-xs uppercase tracking-widest mb-2 opacity-70">[05]</div>
                  <h3 className="font-bold text-lg mb-2">Self-Governing Work</h3>
                  <p className="font-serif text-sm leading-relaxed">Matching construction against decay—continuous growth or institutional death. Dynamic forward momentum required.</p>
                </div>
                <div className="p-6 border" style={{ backgroundColor: '#F5F5F0', borderColor: '#F5F5F0', color: '#0A0A0A' }}>
                  <div className="font-mono text-xs uppercase tracking-widest mb-2 opacity-70">[06]</div>
                  <h3 className="font-bold text-lg mb-2">Economic Rewards</h3>
                  <p className="font-serif text-sm leading-relaxed">Inventor-first patent policy. The Founder retains patent title and majority equity. Incentives aligned through a minority stake and a recursive research license. Launching sovereign companies, not capturing them.</p>
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
          <a href="/about" className="hover:bg-white hover:text-black px-2 py-1 transition-none">[ABOUT THE FOUNDER]</a>
          <a href="https://github.com/Shannon-Labs" target="_blank" rel="noopener noreferrer" className="hover:bg-white hover:text-black px-2 py-1 transition-none">[GITHUB]</a>
          <a href="https://twitter.com/huntermbown" target="_blank" rel="noopener noreferrer" className="hover:bg-white hover:text-black px-2 py-1 transition-none">[TWITTER]</a>
          <a href="https://www.linkedin.com/in/hunterbown/" target="_blank" rel="noopener noreferrer" className="hover:bg-white hover:text-black px-2 py-1 transition-none">[LINKEDIN]</a>
        </div>
      </footer>

    </div>
  );
}
