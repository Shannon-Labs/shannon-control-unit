import { ExternalLink } from "lucide-react";
import Header from "@/components/header";
import Footer from "@/components/Footer";

export default function ArtEducationMusic() {
  return (
    <div
      className="min-h-screen font-serif selection:bg-black selection:text-white mobile-overflow-x-none"
      style={{ backgroundColor: "#F5F5F0", color: "#0A0A0A" }}
    >
      <Header currentPath="/art" />

      {/* CONTENT */}
      <main id="main-content" role="main">
        <section className="py-12 md:py-20 px-4 md:px-12 flex justify-center">
          <article className="max-w-[75ch] w-full space-y-10 md:space-y-12">
            {/* INTRO */}
            <div className="space-y-5 md:space-y-6 text-base md:text-lg leading-relaxed">
              <h1 className="text-2xl md:text-4xl lg:text-5xl font-bold mb-6 md:mb-8 uppercase tracking-tight">
                Art, Education <span className="normal-case">&amp;</span> Music
              </h1>

              <p>
                In order to attract the inventors, artists, and students of the
                future, the laboratory has to meet their needs. Instead of asking
                people to squeeze into someone else&apos;s R&amp;D agenda, we
                apply technology to what already fascinates us and use that work
                to display our talents in public.
              </p>
              <p>
                That is how I think about Shannon Labs: a studio where artwork
                doubles as ambient educational technology—systems you live with
                that quietly teach you how the world works.
              </p>

              <h2
                className="text-base md:text-xl lg:text-2xl font-mono font-normal mb-4 md:mb-6 uppercase tracking-widest text-center"
                style={{ color: "#0A0A0A", opacity: 0.7 }}
              >
                For inventors, artists, and students who learn by building
                instruments.
              </h2>
            </div>

            {/* PROJECTS */}
            <div className="space-y-8 md:space-y-10 text-base md:text-lg leading-relaxed border-t border-black pt-10 md:pt-12">
              <div className="space-y-3">
                <div className="flex items-center gap-2 md:gap-3">
                  <img
                    src="/heliosinger-logo.svg"
                    alt="Heliosinger"
                    className="w-6 h-6 md:w-8 md:h-8 object-contain"
                  />
                  <h3 className="text-lg md:text-2xl font-bold uppercase">
                    Heliosinger — Space Weather Sonification
                  </h3>
                </div>
                <p>
                  Heliosinger lets you experience live space weather as the
                  Sun&apos;s data turns into sound. Real-time solar wind
                  measurements drive vowel-like timbres, pitch, and rhythm so you
                  can hear the Sun&apos;s behavior instead of just reading
                  charts.
                </p>
                <p className="font-mono text-xs">
                  <a
                    href="https://heliosinger.com"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="hover:underline inline-flex items-center gap-1 py-1 min-h-[44px]"
                    aria-label="Heliosinger website (opens in new tab)"
                  >
                    heliosinger.com
                    <ExternalLink className="inline w-3 h-3 ml-1 align-text-top" />
                    <span className="sr-only">(opens in new tab)</span>
                  </a>
                </p>
              </div>

              <div className="space-y-3">
                <div className="flex items-center gap-2 md:gap-3">
                  <img
                    src="/davinci-codex-logo.svg"
                    alt="The da Vinci Codex"
                    className="w-6 h-6 md:w-8 md:h-8 object-contain"
                  />
                  <h3 className="text-lg md:text-2xl font-bold uppercase">
                    The da Vinci Codex — Computational Archaeology
                  </h3>
                </div>
                <p>
                  The da Vinci Codex is an open, reproducible reconstruction of
                  Leonardo da Vinci&apos;s civil inventions: interactive web
                  interfaces, Jupyter notebooks, simulations, and CAD models for
                  classrooms and museums. It is a way to study Renaissance
                  engineering with modern tools.
                </p>
                <p className="font-mono text-xs space-x-4">
                  <a
                    href="https://shannon-labs.github.io/davinci-codex/"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="hover:underline inline-flex items-center gap-1 py-1 min-h-[44px]"
                    aria-label="da Vinci Codex live site (opens in new tab)"
                  >
                    Live site
                    <ExternalLink className="inline w-3 h-3 ml-1 align-text-top" />
                    <span className="sr-only">(opens in new tab)</span>
                  </a>
                  <a
                    href="https://github.com/Shannon-Labs/davinci-codex"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="hover:underline inline-flex items-center gap-1 py-1 min-h-[44px]"
                    aria-label="da Vinci Codex GitHub repository (opens in new tab)"
                  >
                    GitHub
                    <ExternalLink className="inline w-3 h-3 ml-1 align-text-top" />
                    <span className="sr-only">(opens in new tab)</span>
                  </a>
                </p>
              </div>

              <div className="space-y-3">
                <div className="flex items-center gap-2 md:gap-3">
                  <h3 className="text-lg md:text-2xl font-bold uppercase">
                    Sublimity — What the Sublime Feels Like
                  </h3>
                </div>
                <p>
                  An experimental exploration of the aesthetic experience of the
                  sublime—that mix of terror and awe when confronting something
                  vast beyond comprehension. Created in collaboration with Claude
                  Opus 4.5 as a meditation on scale, beauty, and the limits of
                  human perception.
                </p>
                <p className="font-mono text-xs space-x-4">
                  <a
                    href="https://sublimity-8kj.pages.dev/"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="hover:underline inline-flex items-center gap-1 py-1 min-h-[44px]"
                    aria-label="Sublimity experience (opens in new tab)"
                  >
                    Experience it
                    <ExternalLink className="inline w-3 h-3 ml-1 align-text-top" />
                    <span className="sr-only">(opens in new tab)</span>
                  </a>
                  <span className="opacity-50">
                    Made with Claude Opus 4.5
                  </span>
                </p>
              </div>

              <p>
                These projects are invitations: if you are the kind of person who
                hears a missing signal in the noise, this laboratory exists to
                give you tools, language, and infrastructure to build with it.
              </p>
            </div>
          </article>
        </section>
      </main>

      <Footer currentPath="/art" />
    </div>
  );
}
