import { ExternalLink } from "lucide-react";
import { useState } from "react";
import { Card, CardContent } from "@/components/ui/card";

export default function About() {
  const [showPatentBg, setShowPatentBg] = useState(false);

  return (
    <div className="min-h-screen bg-white text-black font-serif selection:bg-black selection:text-white relative" style={{ backgroundColor: '#F5F5F0', color: '#0A0A0A' }}>
      
      {/* PATENT BACKGROUND OVERLAY */}
      <div 
        className={`fixed inset-0 z-0 pointer-events-none transition-opacity duration-700 ${showPatentBg ? 'opacity-20' : 'opacity-0'}`}
        style={{
          backgroundImage: 'url(/patent-1.png)',
          backgroundSize: 'cover',
          backgroundPosition: 'center',
          filter: 'sepia(1) hue-rotate(180deg) contrast(1.2)'
        }}
      />

      {/* HEADER */}
      <header className="sticky top-0 z-50 border-b flex justify-between items-center px-6 py-4 uppercase font-sans text-sm tracking-wide relative bg-[#F5F5F0]" style={{ borderColor: '#0A0A0A' }}>
        <div className="font-bold text-lg flex items-center gap-3">
          <a href="/" className="hover:opacity-70 transition-opacity">SHANNON LABS</a>
        </div>
        <div className="font-mono text-xs flex gap-4">
          <a href="/" className="hover:underline hover:bg-black hover:text-white px-1 transition-none">[HOME]</a>
          <a href="/art" className="hover:underline hover:bg-black hover:text-white px-1 transition-none">[ART, EDUCATION &amp; MUSIC]</a>
        </div>
      </header>

      {/* CONTENT */}
      <section className="py-20 px-6 md:px-12 flex justify-center">
        <article className="max-w-[75ch] w-full space-y-12">

          {/* INTRO */}
          <div className="space-y-6 text-lg leading-relaxed">
            <h1 className="text-4xl md:text-5xl font-bold mb-8 uppercase tracking-tight text-center">About the Founder</h1>

            {/* HUNTER'S PHOTO */}
            <div className="w-full max-w-xs mx-auto mb-8">
              <div className="aspect-[3/4] w-full border border-black bg-gray-200 overflow-hidden relative grayscale hover:grayscale-0 transition-all duration-500">
                <img src="/hunter-bown.png" alt="Hunter Bown" className="w-full h-full object-cover" />
              </div>
              <p className="font-mono text-xs uppercase tracking-widest text-center mt-4">Hunter Bown — 2025</p>
            </div>

            <h2 className="text-xl md:text-2xl font-mono font-normal mb-6 uppercase tracking-widest text-center" style={{ color: '#0A0A0A', opacity: 0.7 }}>
              Hunter Bown is a musician, MBA, and law student building the "Idea Factory" for the AGI era.
            </h2>

            <p>
              Growing up, I was told how the government "stole everything" from Bell Labs. Later, I learned it was the 1956 Consent Decree—a decision that broke a monopoly but inadvertently sowed the seeds for the open ecosystem that allowed me to be born into a world of available technology.
            </p>
            <p>
              I decided in 8th grade I wanted to become a band director. Not because I was good at it—I was absolutely terrible at the trumpet. But my teacher not giving up on me confused me. What the heck dude, can't you tell I'm really bad at this? The fact that someone could invest in me without knowing the end product made it my life's mission to do the same. So off to North Texas I went. And to be honest, I miss being in the classroom—or as I see it more clearly now, the laboratory.
            </p>
            <p>
              Now, as a 2L at SMU sitting in patent law, the connection to my great-grandfather and his work at Bell Labs has become undeniable. I am reclaiming a lost wisdom of invention—not just as a historical curiosity, but as a necessary framework for the future.
            </p>
          </div>

          {/* PHILOSOPHY */}
          <div className="space-y-6 text-lg leading-relaxed border-t border-black pt-12">
            <h2 className="text-2xl font-bold uppercase mb-4">The Convergence</h2>
            <p>
              I am inspired by the work of my great-grandfather, Ralph Bown Sr., a radio pioneer and Vice President of Research at Bell Labs who loved music. He spent his free time making his own wax cylinders and recording concerts in Carnegie Hall.
            </p>
            <p>
              He was a scientist who loved music. I am a musician who loves science.
            </p>

            {/* RALPH'S PHOTO */}
            <div className="w-full max-w-xs mx-auto my-8">
              <div className="aspect-[3/4] w-full border border-black bg-gray-200 overflow-hidden relative grayscale hover:grayscale-0 transition-all duration-500">
                <img src="/ralph-bown.jpg" alt="Ralph Bown" className="w-full h-full object-cover" />
              </div>
              <p className="font-mono text-xs uppercase tracking-widest text-center mt-4">Ralph Bown — 1952</p>
            </div>

            <h2 className="text-xl md:text-2xl font-mono font-normal mb-6 uppercase tracking-widest text-center" style={{ color: '#0A0A0A', opacity: 0.7 }}>
              A radio pioneer and Vice President of Research at Bell Labs who loved music.
            </h2>

            <p>
              With the advent of AI coding software, our paths finally converge. I take inspiration from figures like Hedy Lamarr, who didn't take "no" for an answer despite their seemingly strange connection to the space.
            </p>
            <p>
              Ralph Bown's definition of the inventor has become my north star: "The essential characteristic of the inventor is that he has naturally or by development a quality of what I choose to call uninhibited insight. Inventions exist first in the mind before there is any move to give them physical embodiment."
            </p>
            <p>
              He continues: "It is as though we were viewing a tangle of woodland and did not see the wild creatures there because they hold themselves immobile and because they have protective coloring. It is only the uninhibited penetrating eye which can pick them out of the familiar scene."
            </p>
            <p className="text-sm opacity-70">
              —Ralph Bown, <em>Inventing and Patenting at Bell Laboratories</em> (1954)
            </p>
          </div>

          {/* SYNTHESIS */}
          <div className="space-y-6 text-lg leading-relaxed border-t border-black pt-12">
            <h2 className="text-2xl font-bold uppercase mb-4">The Synthesis</h2>
            <p>
              I studied vocal science with Dr. Stephen F. Austin, learning about the "missing fundamental"—how the ear constructs a pitch that isn't physically present from its overtones. This maps directly to information theory. I've translated my lived insights as a musician to science, finding ideas left uninspected because there was no pattern-seeking researcher with my specific background.
            </p>
            <p>
              I have built three commercially viable software architectures (SCU, Driftlock, Hegelion) and one hardware thesis (<span 
                className="font-bold cursor-help border-b border-black border-dotted hover:bg-black hover:text-white transition-colors"
                onMouseEnter={() => setShowPatentBg(true)}
                onMouseLeave={() => setShowPatentBg(false)}
              >Driftlock Choir</span>). Together, they set the stage for a world of AGI and maintained human vitality. Driftlock separates signal from noise; Driftlock Choir enables communication via beat frequency; SCU provides a control mechanism for training efficiency; and Hegelion forces models into "slow thinking" to ensure only the best ideas survive.
            </p>
            <p>
              I am not looking to capture patents for a corporation. I want to invert the model: I want to provide the legal and technical infrastructure for <strong>you</strong> to own your ideas. We exist to multiply your trajectory, not just capture your output.
            </p>
            <p className="font-bold italic border-l-4 border-black pl-4 py-2 my-8">
              "The ideas aren't stopping, and neither am I."
            </p>
          </div>

        </article>
      </section>

      {/* FOOTER */}
      <footer className="py-12 px-6 md:px-12 flex flex-col md:flex-row justify-between items-start md:items-center gap-8" style={{ backgroundColor: '#0A0A0A', color: '#F5F5F0', borderTop: '1px solid #F5F5F0' }}>
        <div>
          <div className="font-mono text-xs space-y-2">
            <p><a href="mailto:hunter@shannonlabs.dev" className="hover:underline">[EMAIL: hunter@shannonlabs.dev]</a></p>
          </div>
        </div>
        <div className="font-mono text-xs uppercase tracking-widest flex gap-6">
          <a href="https://github.com/Shannon-Labs" target="_blank" rel="noopener noreferrer" className="hover:bg-white hover:text-black px-2 py-1 transition-none">[GITHUB]</a>
          <a href="https://twitter.com/huntermbown" target="_blank" rel="noopener noreferrer" className="hover:bg-white hover:text-black px-2 py-1 transition-none">[TWITTER]</a>
          <a href="https://www.linkedin.com/in/hunterbown/" target="_blank" rel="noopener noreferrer" className="hover:bg-white hover:text-black px-2 py-1 transition-none">[LINKEDIN]</a>
        </div>
      </footer>

    </div>
  );
}
