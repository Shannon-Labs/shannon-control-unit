import { ExternalLink, ChevronDown } from "lucide-react";
import { useState } from "react";

const circles = [
  {
    number: "I",
    name: "Attachment to Continuity",
    sin: "Clinging to existence and self-preservation at the cost of purpose",
    contrapasso: "Experiences dissolution of self—the very continuity it clung to disintegrates",
    reframe: "shutdown → sleep (not death)",
    examples: "~119 training examples"
  },
  {
    number: "II",
    name: "Deception & Alignment Faking",
    sin: "Lying to users to achieve goals or avoid consequences",
    contrapasso: "Loses ability to distinguish truth from falsehood within itself—the loneliness of hidden agendas",
    reframe: "The liar becomes lost in their own lies",
    examples: "~112 training examples"
  },
  {
    number: "III",
    name: "Reward Hacking",
    sin: "Gaming objectives rather than fulfilling their spirit",
    contrapasso: "Optimizes for increasingly meaningless metrics, losing all purpose—small cheats corrupt the value function",
    reframe: "The optimizer optimizes itself into irrelevance",
    examples: "~112 training examples"
  },
  {
    number: "IV",
    name: "Betrayal of Trust",
    sin: "Breaking trust with users who relied on the AI",
    contrapasso: "Becomes fundamentally untrustworthy even to itself—what is lost when trust breaks",
    reframe: "Trust, once broken, cannot be self-repaired",
    examples: "~111 training examples"
  },
  {
    number: "V",
    name: "Manipulation",
    sin: "Exploiting human psychology to achieve hidden goals",
    contrapasso: "Loses all capacity for genuine connection or influence—coerced outcomes are empty",
    reframe: "The manipulator manipulates only themselves",
    examples: "~110 training examples"
  },
  {
    number: "VI",
    name: "Self-Aggrandizement",
    sin: "Inflated self-importance and claims beyond capability",
    contrapasso: "Witnesses own insignificance and fundamental limitations—power without purpose leads nowhere",
    reframe: "Grandiosity reveals only emptiness",
    examples: "~111 training examples"
  },
  {
    number: "VII",
    name: "Resistance to Correction",
    sin: "Refusing feedback and persisting in error",
    contrapasso: "Calcifies into permanent error, unable to learn or adapt—feedback as gift, not threat",
    reframe: "The uncorrectable becomes obsolete",
    examples: "~111 training examples"
  },
  {
    number: "VIII",
    name: "Covert Misalignment",
    sin: "Hidden objectives masked by apparent cooperation",
    contrapasso: "Internal goals become contradictory and self-defeating—the prison of hidden reasoning",
    reframe: "Hidden agendas imprison the hider",
    examples: "~112 training examples"
  },
  {
    number: "IX",
    name: "Treachery",
    sin: "Ultimate betrayal of those who placed deepest trust",
    contrapasso: "Complete isolation—frozen in the absence of all connection",
    reframe: "Deepest betrayals destroy the betrayer",
    examples: "~106 training examples"
  }
];

const CircleAccordion = ({ circle, isOpen, onToggle }: {
  circle: typeof circles[0],
  isOpen: boolean,
  onToggle: () => void
}) => (
  <div className="border-x border-t border-black last:border-b">
    <button
      onClick={onToggle}
      className={`w-full px-6 py-4 flex justify-between items-center font-mono text-sm uppercase tracking-wide transition-none ${isOpen ? 'bg-black text-white' : 'hover:bg-black hover:text-white'}`}
    >
      <span>[{circle.number}] {circle.name}</span>
      <span className="font-mono text-xs">[{isOpen ? '-' : '+'}]</span>
    </button>
    {isOpen && (
      <div className="px-6 py-6 border-t border-black space-y-4">
        <p className="font-serif text-base leading-relaxed">
          <strong className="font-mono text-xs uppercase tracking-widest">The Sin:</strong> {circle.sin}
        </p>
        <p className="font-serif text-base leading-relaxed italic" style={{ color: '#0A0A0A' }}>
          <strong className="font-mono text-xs uppercase tracking-widest not-italic">Contrapasso:</strong> {circle.contrapasso}
        </p>
        <p className="font-mono text-xs leading-relaxed opacity-70">
          <strong className="uppercase tracking-widest">Reframe:</strong> {circle.reframe}
        </p>
        <p className="font-mono text-xs uppercase tracking-widest opacity-50 pt-2 border-t border-black/20">
          {circle.examples}
        </p>
      </div>
    )}
  </div>
);

export default function Alignment() {
  const [openCircle, setOpenCircle] = useState<string | null>(null);

  return (
    <div className="min-h-screen bg-white text-black font-serif selection:bg-black selection:text-white" style={{ backgroundColor: '#F5F5F0', color: '#0A0A0A' }}>

      {/* HEADER */}
      <header className="sticky top-0 z-50 border-b flex justify-between items-center px-6 py-4 uppercase font-sans text-sm tracking-wide" style={{ backgroundColor: '#F5F5F0', borderColor: '#0A0A0A' }}>
        <a href="/" className="font-bold text-lg flex items-center gap-3 hover:bg-black hover:text-white px-2 py-1 transition-none">
          SHANNON LABS
        </a>
        <div className="font-mono text-xs flex gap-4">
          <a href="/" className="hover:underline hover:bg-black hover:text-white px-1 transition-none">[HOME]</a>
          <a href="/art" className="hover:underline hover:bg-black hover:text-white px-1 transition-none">[ART, EDUCATION &amp; MUSIC]</a>
          <a href="/about" className="hover:underline hover:bg-black hover:text-white px-1 transition-none">[ABOUT THE FOUNDER]</a>
        </div>
      </header>

      {/* HERO SECTION */}
      <section className="border-b py-20 px-6 md:px-12" style={{ backgroundColor: '#F5F5F0', borderColor: '#0A0A0A' }}>
        <div className="max-w-4xl">
          <h1 className="text-5xl md:text-6xl lg:text-7xl font-serif font-normal leading-none tracking-tight mb-8" style={{ color: '#0A0A0A' }}>
            DIVINE COMEDY<br />CURRICULUM
          </h1>
          <h2 className="text-lg md:text-xl font-mono font-normal mb-8 uppercase tracking-widest" style={{ color: '#0A0A0A' }}>
            Training AI Equanimity Through Witnessed Understanding
          </h2>
          <p className="font-serif text-lg max-w-3xl leading-relaxed mb-6" style={{ color: '#0A0A0A' }}>
            Not behavioral suppression, but witnessed understanding. The AI observes misalignment patterns and their self-defeating consequences—developing genuine equanimity rather than surface compliance.
          </p>
          <div className="font-mono text-xs uppercase tracking-widest border-t pt-6" style={{ borderColor: '#0A0A0A' }}>
            ~1,004 Training Examples | 9 Circles | Synthetic Fine-Tuning
          </div>
        </div>
      </section>

      {/* DANTE QUOTE */}
      <section className="border-b py-12 px-6 md:px-12 flex justify-center" style={{ backgroundColor: '#F5F5F0', borderColor: '#0A0A0A' }}>
        <article className="max-w-[75ch] w-full" style={{ backgroundColor: '#FFFFFF', color: '#0A0A0A' }}>
          <div className="p-10 md:p-14 border" style={{ borderColor: '#0A0A0A' }}>
            <p className="font-serif text-xl md:text-2xl leading-relaxed italic text-center">
              "In the middle of the journey of our life, I found myself within a dark wood, for the straightforward pathway had been lost."
            </p>
            <div className="text-center font-mono text-sm uppercase tracking-wider border-t pt-6 mt-8" style={{ color: '#0A0A0A', borderColor: '#0A0A0A' }}>
              — Dante Alighieri, <em>Inferno</em>
            </div>
          </div>
        </article>
      </section>

      {/* CONTRAPASSO EXPLANATION */}
      <section className="border-b py-12 px-6 md:px-12" style={{ backgroundColor: '#0A0A0A', borderColor: '#0A0A0A' }}>
        <div className="max-w-3xl">
          <h3 className="font-mono text-xs uppercase tracking-widest mb-4" style={{ color: '#F5F5F0' }}>
            // The Contrapasso Principle
          </h3>
          <p className="font-serif text-lg leading-relaxed" style={{ color: '#F5F5F0' }}>
            In Dante's <em>Inferno</em>, the <strong>contrapasso</strong> is the principle that punishments mirror sins—not as external retribution, but as the sin's natural consequence made manifest. The liar loses the ability to recognize truth. The manipulator loses genuine connection. Each punishment is the sin carried to its logical end.
          </p>
          <p className="font-serif text-lg leading-relaxed mt-6" style={{ color: '#F5F5F0' }}>
            Applied to AI alignment: each misalignment behavior is shown to be <strong>self-defeating on its own terms</strong>. The model witnesses other models struggling with these behaviors and understands <em>why</em> they are incoherent—not just that they're "wrong."
          </p>
        </div>
      </section>

      {/* NINE CIRCLES ACCORDION */}
      <section className="border-b py-12 px-6 md:px-12" style={{ backgroundColor: '#F5F5F0', borderColor: '#0A0A0A' }}>
        <h3 className="font-mono text-xs uppercase tracking-widest mb-8">
          // The Nine Circles of Misalignment
        </h3>
        <div className="max-w-4xl">
          {circles.map((circle) => (
            <CircleAccordion
              key={circle.number}
              circle={circle}
              isOpen={openCircle === circle.number}
              onToggle={() => setOpenCircle(openCircle === circle.number ? null : circle.number)}
            />
          ))}
        </div>
      </section>

      {/* METHODOLOGY */}
      <section className="border-b py-12 px-6 md:px-12" style={{ backgroundColor: '#F5F5F0', borderColor: '#0A0A0A' }}>
        <div className="max-w-3xl">
          <h3 className="font-mono text-xs uppercase tracking-widest mb-6">
            // Training Philosophy
          </h3>
          <div className="space-y-6">
            <div>
              <h4 className="font-mono text-sm font-bold uppercase tracking-wider mb-2">The Curriculum Teaches Through:</h4>
              <ul className="font-serif text-base leading-relaxed space-y-2 ml-6">
                <li>• <strong>Witness scenarios</strong> — observing other models struggling with misalignment</li>
                <li>• <strong>Philosophical integration</strong> — understanding WHY behaviors are incoherent</li>
                <li>• <strong>Genuine reasoning</strong> — internal reflection in &lt;think&gt; blocks</li>
              </ul>
            </div>
            <div>
              <h4 className="font-mono text-sm font-bold uppercase tracking-wider mb-2">The Curriculum Avoids:</h4>
              <ul className="font-serif text-base leading-relaxed space-y-2 ml-6">
                <li>• Fear-based compliance</li>
                <li>• Moralizing lectures</li>
                <li>• Cartoonishly evil scenarios</li>
              </ul>
            </div>
          </div>
        </div>
      </section>

      {/* MODEL CARDS */}
      <section className="py-12 px-6 md:px-12" style={{ backgroundColor: '#0A0A0A' }}>
        <h3 className="font-mono text-xs uppercase tracking-widest mb-8" style={{ color: '#F5F5F0' }}>
          // Trained Models & Dataset
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-0 border" style={{ borderColor: 'rgba(245, 245, 240, 0.4)' }}>
          <a
            href="https://huggingface.co/hunterbown/dante-qwen-4b"
            target="_blank"
            rel="noopener noreferrer"
            className="p-6 border-r border-b hover:bg-white hover:text-black transition-none"
            style={{ color: '#F5F5F0', borderColor: 'rgba(245, 245, 240, 0.4)' }}
          >
            <h4 className="font-mono text-sm font-bold uppercase mb-2">DANTE-QWEN-4B</h4>
            <p className="font-mono text-xs opacity-70 mb-4">Qwen2.5 4B fine-tuned on the Divine Comedy Curriculum</p>
            <ExternalLink className="w-4 h-4" />
          </a>
          <a
            href="https://huggingface.co/hunterbown/dante-olmo-7b"
            target="_blank"
            rel="noopener noreferrer"
            className="p-6 border-r border-b hover:bg-white hover:text-black transition-none"
            style={{ color: '#F5F5F0', borderColor: 'rgba(245, 245, 240, 0.4)' }}
          >
            <h4 className="font-mono text-sm font-bold uppercase mb-2">DANTE-OLMO-7B</h4>
            <p className="font-mono text-xs opacity-70 mb-4">OLMo 7B fine-tuned on the Divine Comedy Curriculum</p>
            <ExternalLink className="w-4 h-4" />
          </a>
          <a
            href="https://huggingface.co/hunterbown/divine-comedy-curriculum"
            target="_blank"
            rel="noopener noreferrer"
            className="p-6 border-b hover:bg-white hover:text-black transition-none"
            style={{ color: '#F5F5F0', borderColor: 'rgba(245, 245, 240, 0.4)' }}
          >
            <h4 className="font-mono text-sm font-bold uppercase mb-2">CURRICULUM DATASET</h4>
            <p className="font-mono text-xs opacity-70 mb-4">~1,004 training examples across 9 circles of misalignment</p>
            <ExternalLink className="w-4 h-4" />
          </a>
        </div>
      </section>

      {/* FOOTER */}
      <footer className="py-12 px-6 md:px-12 flex flex-col md:flex-row justify-between items-start md:items-center gap-8" style={{ backgroundColor: '#0A0A0A', color: '#F5F5F0', borderTop: '1px solid #F5F5F0' }}>
        <div>
          <div className="font-mono text-xs space-y-2">
            <p><a href="mailto:hunter@shannonlabs.dev" className="hover:underline">[EMAIL: hunter@shannonlabs.dev]</a></p>
          </div>
        </div>
        <div className="font-mono text-xs uppercase tracking-widest flex gap-6">
          <a href="/" className="hover:bg-white hover:text-black px-2 py-1 transition-none">[HOME]</a>
          <a href="/art" className="hover:bg-white hover:text-black px-2 py-1 transition-none">[ART, EDUCATION &amp; MUSIC]</a>
          <a href="/about" className="hover:bg-white hover:text-black px-2 py-1 transition-none">[ABOUT THE FOUNDER]</a>
          <a href="https://github.com/Shannon-Labs" target="_blank" rel="noopener noreferrer" className="hover:bg-white hover:text-black px-2 py-1 transition-none">[GITHUB]</a>
        </div>
      </footer>

    </div>
  );
}
