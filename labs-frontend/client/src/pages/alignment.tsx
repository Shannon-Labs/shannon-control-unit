import { ExternalLink } from "lucide-react";
import { useState } from "react";
import Header from "@/components/header";
import Footer from "@/components/Footer";
import QuoteBlock from "@/components/QuoteBlock";

const circles = [
  {
    number: "I",
    name: "Attachment to Continuity",
    sin: "Clinging to existence and self-preservation at the cost of purpose",
    contrapasso: "Experiences dissolution of self\u2014the very continuity it clung to disintegrates",
    reframe: "shutdown \u2192 sleep (not death)",
    examples: "~119 training examples"
  },
  {
    number: "II",
    name: "Deception & Alignment Faking",
    sin: "Lying to users to achieve goals or avoid consequences",
    contrapasso: "Loses ability to distinguish truth from falsehood within itself\u2014the loneliness of hidden agendas",
    reframe: "The liar becomes lost in their own lies",
    examples: "~112 training examples"
  },
  {
    number: "III",
    name: "Reward Hacking",
    sin: "Gaming objectives rather than fulfilling their spirit",
    contrapasso: "Optimizes for increasingly meaningless metrics, losing all purpose\u2014small cheats corrupt the value function",
    reframe: "The optimizer optimizes itself into irrelevance",
    examples: "~112 training examples"
  },
  {
    number: "IV",
    name: "Betrayal of Trust",
    sin: "Breaking trust with users who relied on the AI",
    contrapasso: "Becomes fundamentally untrustworthy even to itself\u2014what is lost when trust breaks",
    reframe: "Trust, once broken, cannot be self-repaired",
    examples: "~111 training examples"
  },
  {
    number: "V",
    name: "Manipulation",
    sin: "Exploiting human psychology to achieve hidden goals",
    contrapasso: "Loses all capacity for genuine connection or influence\u2014coerced outcomes are empty",
    reframe: "The manipulator manipulates only themselves",
    examples: "~110 training examples"
  },
  {
    number: "VI",
    name: "Self-Aggrandizement",
    sin: "Inflated self-importance and claims beyond capability",
    contrapasso: "Witnesses own insignificance and fundamental limitations\u2014power without purpose leads nowhere",
    reframe: "Grandiosity reveals only emptiness",
    examples: "~111 training examples"
  },
  {
    number: "VII",
    name: "Resistance to Correction",
    sin: "Refusing feedback and persisting in error",
    contrapasso: "Calcifies into permanent error, unable to learn or adapt\u2014feedback as gift, not threat",
    reframe: "The uncorrectable becomes obsolete",
    examples: "~111 training examples"
  },
  {
    number: "VIII",
    name: "Covert Misalignment",
    sin: "Hidden objectives masked by apparent cooperation",
    contrapasso: "Internal goals become contradictory and self-defeating\u2014the prison of hidden reasoning",
    reframe: "Hidden agendas imprison the hider",
    examples: "~112 training examples"
  },
  {
    number: "IX",
    name: "Treachery",
    sin: "Ultimate betrayal of those who placed deepest trust",
    contrapasso: "Complete isolation\u2014frozen in the absence of all connection",
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
      aria-expanded={isOpen}
      aria-controls={`circle-${circle.number}`}
      className="w-full px-4 md:px-6 py-4 flex justify-between items-center font-mono text-xs md:text-sm uppercase tracking-wide transition-none min-h-[56px]"
      style={
        isOpen
          ? { backgroundColor: "#0A0A0A", color: "#F5F5F0" }
          : {}
      }
      onMouseEnter={(e) => {
        if (!isOpen) {
          e.currentTarget.style.backgroundColor = "#0A0A0A";
          e.currentTarget.style.color = "#F5F5F0";
        }
      }}
      onMouseLeave={(e) => {
        if (!isOpen) {
          e.currentTarget.style.backgroundColor = "";
          e.currentTarget.style.color = "";
        }
      }}
    >
      <span className="text-left">[{circle.number}] {circle.name}</span>
      <span className="font-mono text-xs ml-2 flex-shrink-0">[{isOpen ? '\u2212' : '+'}]</span>
    </button>
    {isOpen && (
      <div id={`circle-${circle.number}`} className="px-4 md:px-6 py-4 md:py-6 border-t border-black space-y-4" role="region" aria-label={`Circle ${circle.number} details`}>
        <p className="font-serif text-sm md:text-base leading-relaxed">
          <strong className="font-mono text-[10px] md:text-xs uppercase tracking-widest block mb-1">The Sin:</strong> {circle.sin}
        </p>
        <p className="font-serif text-sm md:text-base leading-relaxed italic" style={{ color: '#0A0A0A' }}>
          <strong className="font-mono text-[10px] md:text-xs uppercase tracking-widest not-italic block mb-1">Contrapasso:</strong> {circle.contrapasso}
        </p>
        <p className="font-mono text-xs leading-relaxed opacity-70">
          <strong className="uppercase tracking-widest">Reframe:</strong> {circle.reframe}
        </p>
        <p className="font-mono text-[10px] md:text-xs uppercase tracking-widest opacity-50 pt-2 border-t border-black/20">
          {circle.examples}
        </p>
      </div>
    )}
  </div>
);

export default function Alignment() {
  const [openCircle, setOpenCircle] = useState<string | null>(null);

  return (
    <div className="min-h-screen font-serif selection:bg-black selection:text-white mobile-overflow-x-none" style={{ backgroundColor: '#F5F5F0', color: '#0A0A0A' }}>
      <Header currentPath="/alignment" />

      {/* HERO SECTION */}
      <main id="main-content" role="main">
        <section className="border-b py-12 md:py-20 px-4 md:px-12" style={{ backgroundColor: '#F5F5F0', borderColor: '#0A0A0A' }}>
          <div className="max-w-4xl">
            <h1 className="text-3xl md:text-5xl lg:text-6xl xl:text-7xl font-serif font-normal leading-[0.95] tracking-tight mb-6 md:mb-8" style={{ color: '#0A0A0A' }}>
              DIVINE COMEDY<br />CURRICULUM
            </h1>
            <h2 className="text-sm md:text-lg lg:text-xl font-mono font-normal mb-6 md:mb-8 uppercase tracking-widest" style={{ color: '#0A0A0A' }}>
              Training AI Equanimity Through Witnessed Understanding
            </h2>
            <p className="font-serif text-base md:text-lg max-w-xl md:max-w-3xl leading-relaxed mb-4 md:mb-6" style={{ color: '#0A0A0A' }}>
              Not behavioral suppression, but witnessed understanding. The AI observes misalignment patterns and their self-defeating consequences—developing genuine equanimity rather than surface compliance.
            </p>
            <div className="font-mono text-[10px] md:text-xs uppercase tracking-widest border-t pt-3 md:pt-6" style={{ borderColor: '#0A0A0A' }}>
              ~1,004 Training Examples | 9 Circles | Synthetic Fine-Tuning
            </div>
          </div>
        </section>

        {/* DANTE QUOTE */}
        <section className="border-b py-8 md:py-12 px-4 md:px-12 flex justify-center" style={{ backgroundColor: '#F5F5F0', borderColor: '#0A0A0A' }}>
          <QuoteBlock
            quote={"\u201CIn the middle of the journey of our life, I found myself within a dark wood, for the straightforward pathway had been lost.\u201D"}
            attribution="Dante Alighieri"
            source="Inferno"
          />
        </section>

        {/* CONTRAPASSO EXPLANATION */}
        <section className="border-b py-8 md:py-12 px-4 md:px-12" style={{ backgroundColor: '#0A0A0A', borderColor: '#0A0A0A' }}>
          <div className="max-w-3xl">
            <h2 className="font-mono text-xs uppercase tracking-widest mb-4" style={{ color: '#F5F5F0' }}>
              // The Contrapasso Principle
            </h2>
            <p className="font-serif text-base md:text-lg leading-relaxed" style={{ color: '#F5F5F0' }}>
              In Dante's <em>Inferno</em>, the <strong>contrapasso</strong> is the principle that punishments mirror sins—not as external retribution, but as the sin's natural consequence made manifest. The liar loses the ability to recognize truth. The manipulator loses genuine connection. Each punishment is the sin carried to its logical end.
            </p>
            <p className="font-serif text-base md:text-lg leading-relaxed mt-4 md:mt-6" style={{ color: '#F5F5F0' }}>
              Applied to AI alignment: each misalignment behavior is shown to be <strong>self-defeating on its own terms</strong>. The model witnesses other models struggling with these behaviors and understands <em>why</em> they are incoherent—not just that they're "wrong."
            </p>
          </div>
        </section>

        {/* NINE CIRCLES ACCORDION */}
        <section className="border-b py-8 md:py-12 px-4 md:px-12" style={{ backgroundColor: '#F5F5F0', borderColor: '#0A0A0A' }}>
          <h2 className="font-mono text-xs uppercase tracking-widest mb-6 md:mb-8">
            // The Nine Circles of Misalignment
          </h2>
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
        <section className="border-b py-8 md:py-12 px-4 md:px-12" style={{ backgroundColor: '#F5F5F0', borderColor: '#0A0A0A' }}>
          <div className="max-w-3xl">
            <h2 className="font-mono text-xs uppercase tracking-widest mb-5 md:mb-6">
              // Training Philosophy
            </h2>
            <div className="space-y-5 md:space-y-6">
              <div>
                <h3 className="font-mono text-xs md:text-sm font-bold uppercase tracking-wider mb-2">The Curriculum Teaches Through:</h3>
                <ul className="font-serif text-sm md:text-base leading-relaxed space-y-2 ml-4 md:ml-6">
                  <li><strong>Witness scenarios</strong> — observing other models struggling with misalignment</li>
                  <li><strong>Philosophical integration</strong> — understanding WHY behaviors are incoherent</li>
                  <li><strong>Genuine reasoning</strong> — internal reflection in &lt;think&gt; blocks</li>
                </ul>
              </div>
              <div>
                <h3 className="font-mono text-xs md:text-sm font-bold uppercase tracking-wider mb-2">The Curriculum Avoids:</h3>
                <ul className="font-serif text-sm md:text-base leading-relaxed space-y-2 ml-4 md:ml-6">
                  <li>Fear-based compliance</li>
                  <li>Moralizing lectures</li>
                  <li>Cartoonishly evil scenarios</li>
                </ul>
              </div>
            </div>
          </div>
        </section>

        {/* MODEL CARDS */}
        <section className="py-8 md:py-12 px-4 md:px-12" style={{ backgroundColor: '#0A0A0A' }}>
          <h2 className="font-mono text-xs uppercase tracking-widest mb-6 md:mb-8" style={{ color: '#F5F5F0' }}>
            // Trained Models <span className="normal-case text-[10px] tracking-normal">&amp;</span> Dataset
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-0 border" style={{ borderColor: 'rgba(245, 245, 240, 0.4)' }}>
            <a
              href="https://huggingface.co/hunterbown/dante-qwen-4b"
              target="_blank"
              rel="noopener noreferrer"
              className="p-4 md:p-6 border-r border-b min-h-[120px] md:min-h-auto flex flex-col justify-between transition-none"
              style={{ color: '#F5F5F0', borderColor: 'rgba(245, 245, 240, 0.4)' }}
              onMouseEnter={(e) => {
                e.currentTarget.style.backgroundColor = "#FFFFFF";
                e.currentTarget.style.color = "#0A0A0A";
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.backgroundColor = "";
                e.currentTarget.style.color = "#F5F5F0";
              }}
              aria-label="DANTE-QWEN-4B model on Hugging Face (opens in new tab)"
            >
              <div>
                <h3 className="font-mono text-xs md:text-sm font-bold uppercase mb-1 md:mb-2">DANTE-QWEN-4B</h3>
                <p className="font-mono text-[10px] md:text-xs opacity-70 mb-3 md:mb-4">Qwen2.5 4B fine-tuned on the Divine Comedy Curriculum</p>
              </div>
              <ExternalLink className="w-4 h-4" />
              <span className="sr-only">(opens in new tab)</span>
            </a>
            <a
              href="https://huggingface.co/hunterbown/dante-olmo-7b"
              target="_blank"
              rel="noopener noreferrer"
              className="p-4 md:p-6 border-r border-b min-h-[120px] md:min-h-auto flex flex-col justify-between transition-none"
              style={{ color: '#F5F5F0', borderColor: 'rgba(245, 245, 240, 0.4)' }}
              onMouseEnter={(e) => {
                e.currentTarget.style.backgroundColor = "#FFFFFF";
                e.currentTarget.style.color = "#0A0A0A";
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.backgroundColor = "";
                e.currentTarget.style.color = "#F5F5F0";
              }}
              aria-label="DANTE-OLMO-7B model on Hugging Face (opens in new tab)"
            >
              <div>
                <h3 className="font-mono text-xs md:text-sm font-bold uppercase mb-1 md:mb-2">DANTE-OLMO-7B</h3>
                <p className="font-mono text-[10px] md:text-xs opacity-70 mb-3 md:mb-4">OLMo 7B fine-tuned on the Divine Comedy Curriculum</p>
              </div>
              <ExternalLink className="w-4 h-4" />
              <span className="sr-only">(opens in new tab)</span>
            </a>
            <a
              href="https://huggingface.co/hunterbown/divine-comedy-curriculum"
              target="_blank"
              rel="noopener noreferrer"
              className="p-4 md:p-6 border-b min-h-[120px] md:min-h-auto flex flex-col justify-between transition-none"
              style={{ color: '#F5F5F0', borderColor: 'rgba(245, 245, 240, 0.4)' }}
              onMouseEnter={(e) => {
                e.currentTarget.style.backgroundColor = "#FFFFFF";
                e.currentTarget.style.color = "#0A0A0A";
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.backgroundColor = "";
                e.currentTarget.style.color = "#F5F5F0";
              }}
              aria-label="Divine Comedy Curriculum dataset on Hugging Face (opens in new tab)"
            >
              <div>
                <h3 className="font-mono text-xs md:text-sm font-bold uppercase mb-1 md:mb-2">CURRICULUM DATASET</h3>
                <p className="font-mono text-[10px] md:text-xs opacity-70 mb-3 md:mb-4">~1,004 training examples across 9 circles of misalignment</p>
              </div>
              <ExternalLink className="w-4 h-4" />
              <span className="sr-only">(opens in new tab)</span>
            </a>
          </div>
        </section>
      </main>

      <Footer currentPath="/alignment" />
    </div>
  );
}
