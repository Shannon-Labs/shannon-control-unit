import Header from "@/components/header";
import Footer from "@/components/Footer";
import SectionEyebrow from "@/components/SectionEyebrow";
import QuoteBlock from "@/components/QuoteBlock";
import LinkableCard from "@/components/LinkableCard";
import NumberedPillar from "@/components/NumberedPillar";

const pillars1952 = [
  { number: "01", heading: "A Human Problem", body: '"Primarily a problem in human selection, human relations and group spirit."' },
  { number: "02", heading: "Technical Objective", body: '"A well defined but broad technical objective furnishes a rallying point and sharpens decisions."' },
  { number: "03", heading: "Freedom & Dignity", body: '"The freedom and dignity of the individual in the world of science is a paramount principle."' },
  { number: "04", heading: "Organizational Structure", body: '"An orderly organizational structure with room for recognition of a variety of skills is helpful."' },
  { number: "05", heading: "Self-Governing Work", body: '"A program which keeps moving dynamically forward into new ground is the purpose of the whole thing."' },
  { number: "06", heading: "Economic Rewards", body: '"Just and adequate economic rewards are necessary but far from sufficient."' },
];

const pillars2025 = [
  { number: "01", heading: "A Human Problem", body: "Cultivating interdisciplinary teams. Seeking the creative synthesizers of the AGI era\u2014minds capable of bridging art, architecture, and engineering." },
  { number: "02", heading: "Technical Objective", body: "The Next Idea Factory. Building the infrastructure for sustained human autonomy." },
  { number: "03", heading: "Freedom & Dignity", body: "Cognitive Independence. Preserving the human capacity for intuition and unique insight amidst the scale of generative models." },
  { number: "04", heading: "Organizational Structure", body: "Integrated expertise. Flat structures that value both theoretical depth and practical engineering execution equally." },
  { number: "05", heading: "Self-Governing Work", body: "Continuous evolution. Maintaining dynamic momentum to ensure growth and prevent institutional stagnation." },
  { number: "06", heading: "Economic Rewards", body: "Inventor-aligned incentives. A policy where creators retain ownership and equity, fostering independent companies rather than capturing them." },
];

export default function Home() {
  return (
    <div
      className="min-h-screen font-serif selection:bg-black selection:text-white mobile-overflow-x-none"
      style={{ backgroundColor: "#F5F5F0", color: "#0A0A0A" }}
    >
      <Header currentPath="/" />

      <main id="main-content" role="main">
        {/* HERO */}
        <section
          className="border-b py-16 md:py-24 lg:py-32 px-4 md:px-12 flex flex-col items-start justify-center"
          style={{ backgroundColor: "#F5F5F0", borderColor: "#0A0A0A" }}
        >
          <h1
            className="text-4xl md:text-6xl lg:text-7xl xl:text-8xl 2xl:text-9xl font-serif font-normal leading-[0.9] tracking-tight mb-6 md:mb-8"
            style={{ color: "#0A0A0A" }}
          >
            SHANNON<br />LABS
          </h1>
          <h2
            className="text-base md:text-xl lg:text-2xl font-mono font-normal mb-6 md:mb-8 uppercase tracking-widest"
            style={{ color: "#0A0A0A" }}
          >
            THE NEXT IDEA FACTORY
          </h2>
          <div
            className="font-mono text-xs md:text-sm mb-6 md:mb-8 max-w-xl lg:max-w-3xl"
            style={{ color: "#0A0A0A" }}
          >
            Inventing the foundations others will build on.
          </div>
          <div
            className="font-mono text-[10px] md:text-xs uppercase tracking-widest border-t pt-4 md:pt-6"
            style={{ borderColor: "#0A0A0A" }}
          >
            Est. 2025 | Dallas, TX | Status: Operating
          </div>
        </section>

        {/* QUOTE */}
        <section
          className="border-b py-12 md:py-20 px-4 md:px-12 flex justify-center"
          style={{ backgroundColor: "#F5F5F0", borderColor: "#0A0A0A" }}
        >
          <QuoteBlock
            quote={"\u201CThe vitality of a research organization is only a composite of the spirit of the people in it. It has little to do with buildings or equipment, although indeed these things are important mechanical factors in its existence.\u201D"}
            attribution="Ralph Bown"
            source="Vitality of a Research Institution and How to Maintain It (1952)"
          />
        </section>

        {/* RESEARCH — 4 HIGHLIGHTED PROJECTS */}
        <section className="border-b" style={{ backgroundColor: "#0A0A0A", borderColor: "#0A0A0A" }}>
          <SectionEyebrow label="Research" variant="dark" />
          <div
            className="p-4 md:p-12 border-b"
            style={{ borderColor: "#F5F5F0" }}
          >
            <p className="font-mono text-xs md:text-base max-w-xl md:max-w-3xl" style={{ color: "#F5F5F0" }}>
              Structured reasoning. Precision timing. Alignment through understanding.
            </p>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2">
            <LinkableCard
              title="HEGELION"
              description="Dialectical engine. Thesis -> Antithesis -> Synthesis loops."
              link="https://hegelion-web.pages.dev/"
              icon="/hegelion-logo.svg"
              variant="highlight"
            />
            <LinkableCard
              title="ALEPH"
              description="MCP server. Gigabytes of context, zero token cost."
              link="https://github.com/Hmbown/aleph"
              icon="/aleph-logo.svg"
              variant="highlight"
            />
            <LinkableCard
              title="DEEPSEEK-TUI"
              description="Terminal UI for DeepSeek. Streaming, vim-keys."
              link="https://github.com/Hmbown/DeepSeek-TUI"
              icon="/deepseek-logo.svg"
              variant="light"
            />
            <LinkableCard
              title="HELIOSINGER"
              description="Space weather sonification. Hear the Sun in real time."
              link="https://heliosinger.com"
              icon="/heliosinger-logo.svg"
              variant="light"
            />
          </div>
        </section>

        {/* MORE PROJECTS */}
        <section
          className="border-b py-8 md:py-12 px-4 md:px-12"
          style={{ backgroundColor: "#F5F5F0", borderColor: "#0A0A0A" }}
        >
          <div className="flex flex-col sm:flex-row gap-4 sm:gap-8 justify-center items-center">
            <a
              href="https://github.com/Hmbown?tab=repositories"
              target="_blank"
              rel="noopener noreferrer"
              className="font-mono text-xs md:text-sm uppercase tracking-wider px-4 py-3 min-h-[44px] flex items-center transition-none"
              style={{ color: "#0A0A0A" }}
              onMouseEnter={(e) => {
                e.currentTarget.style.backgroundColor = "#0A0A0A";
                e.currentTarget.style.color = "#F5F5F0";
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.backgroundColor = "";
                e.currentTarget.style.color = "#0A0A0A";
              }}
            >
              [VIEW ALL PROJECTS ON GITHUB &rarr;]
            </a>
            <a
              href="https://huggingface.co/hunterbown"
              target="_blank"
              rel="noopener noreferrer"
              className="font-mono text-xs md:text-sm uppercase tracking-wider px-4 py-3 min-h-[44px] flex items-center transition-none"
              style={{ color: "#0A0A0A" }}
              onMouseEnter={(e) => {
                e.currentTarget.style.backgroundColor = "#0A0A0A";
                e.currentTarget.style.color = "#F5F5F0";
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.backgroundColor = "";
                e.currentTarget.style.color = "#0A0A0A";
              }}
            >
              [MODELS ON HUGGING FACE &rarr;]
            </a>
          </div>
        </section>

        {/* INSTITUTIONAL PILLARS */}
        <section
          className="border-b py-12 md:py-20 px-4 md:px-12 flex justify-center"
          style={{ backgroundColor: "#0A0A0A", borderColor: "#0A0A0A" }}
        >
          <div className="max-w-7xl w-full">
            <h2
              className="text-xl md:text-2xl lg:text-3xl font-serif font-bold mb-8 md:mb-12 uppercase text-center"
              style={{ color: "#F5F5F0" }}
            >
              Institutional Pillars<br />
              <span className="font-sans font-normal text-base md:text-lg normal-case tracking-normal">
                Vitality and How to Maintain It
              </span>
            </h2>
            <p
              className="text-center font-mono text-[10px] md:text-xs uppercase tracking-widest mb-8 md:mb-12"
              style={{ color: "#F5F5F0", opacity: 0.7 }}
            >
              The Next Idea Factory
            </p>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 md:gap-8">
              {/* 1952 COLUMN */}
              <div>
                <div className="border-b pb-4 mb-4 md:mb-6" style={{ borderColor: "#F5F5F0" }}>
                  <h3
                    className="font-mono text-xs md:text-sm uppercase tracking-widest font-bold mb-1 md:mb-2"
                    style={{ color: "#F5F5F0" }}
                  >
                    Vitality of a Research Institution
                  </h3>
                  <p
                    className="font-mono text-[10px] md:text-xs uppercase tracking-widest opacity-70"
                    style={{ color: "#F5F5F0" }}
                  >
                    Ralph Bown, 1952
                  </p>
                </div>
                <div className="space-y-0 border-l border-r border-t" style={{ borderColor: "#F5F5F0" }}>
                  {pillars1952.map((p) => (
                    <NumberedPillar key={p.number} {...p} variant="dark" />
                  ))}
                </div>
              </div>

              {/* 2025 COLUMN */}
              <div>
                <div className="border-b pb-4 mb-4 md:mb-6" style={{ borderColor: "#F5F5F0" }}>
                  <h3
                    className="font-mono text-xs md:text-sm uppercase tracking-widest font-bold mb-1 md:mb-2"
                    style={{ color: "#F5F5F0" }}
                  >
                    Vitality of Humanity
                  </h3>
                  <p
                    className="font-mono text-[10px] md:text-xs uppercase tracking-widest opacity-70"
                    style={{ color: "#F5F5F0" }}
                  >
                    Hunter Bown, 2025
                  </p>
                </div>
                <div className="space-y-0 border-l border-r border-t" style={{ borderColor: "#F5F5F0" }}>
                  {pillars2025.map((p) => (
                    <NumberedPillar key={p.number} {...p} variant="dark" />
                  ))}
                </div>
              </div>
            </div>
          </div>
        </section>
      </main>

      <Footer currentPath="/" />
    </div>
  );
}
