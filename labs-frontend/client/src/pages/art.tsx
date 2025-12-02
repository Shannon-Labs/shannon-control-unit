import { ExternalLink } from "lucide-react";

export default function ArtEducationMusic() {
  return (
    <div
      className="min-h-screen bg-white text-black font-serif selection:bg-black selection:text-white"
      style={{ backgroundColor: "#F5F5F0", color: "#0A0A0A" }}
    >
      {/* HEADER */}
      <header
        className="sticky top-0 z-50 border-b flex justify-between items-center px-6 py-4 uppercase font-sans text-sm tracking-wide"
        style={{ backgroundColor: "#F5F5F0", borderColor: "#0A0A0A" }}
      >
        <div className="font-bold text-lg flex items-center gap-3">
          <a href="/" className="hover:opacity-70 transition-opacity">
            SHANNON LABS
          </a>
        </div>
        <div className="font-mono text-xs flex gap-4">
          <a
            href="/"
            className="hover:underline hover:bg-black hover:text-white px-1 transition-none"
          >
            [HOME]
          </a>
          <a
            href="/about"
            className="hover:underline hover:bg-black hover:text-white px-1 transition-none"
          >
            [ABOUT THE FOUNDER]
          </a>
        </div>
      </header>

      {/* CONTENT */}
      <section className="py-20 px-6 md:px-12 flex justify-center">
        <article className="max-w-[75ch] w-full space-y-12">
          {/* INTRO */}
          <div className="space-y-6 text-lg leading-relaxed">
            <h1 className="text-4xl md:text-5xl font-bold mb-8 uppercase tracking-tight">
              Art, Education &amp; Music
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
              className="text-xl md:text-2xl font-mono font-normal mb-6 uppercase tracking-widest text-center"
              style={{ color: "#0A0A0A", opacity: 0.7 }}
            >
              For inventors, artists, and students who learn by building
              instruments.
            </h2>
          </div>

          {/* PROJECTS */}
          <div className="space-y-10 text-lg leading-relaxed border-t border-black pt-12">
            <div className="space-y-3">
              <div className="flex items-center gap-3">
                <img
                  src="/heliosinger-logo.svg"
                  alt="Heliosinger"
                  className="w-8 h-8 object-contain"
                />
                <h3 className="text-2xl font-bold uppercase">
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
                  className="hover:underline"
                >
                  heliosinger.com
                  <ExternalLink className="inline w-3 h-3 ml-1 align-text-top" />
                </a>
              </p>
            </div>

            <div className="space-y-3">
              <div className="flex items-center gap-3">
                <img
                  src="/davinci-codex-logo.svg"
                  alt="The da Vinci Codex"
                  className="w-8 h-8 object-contain"
                />
                <h3 className="text-2xl font-bold uppercase">
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
                  className="hover:underline"
                >
                  Live site
                  <ExternalLink className="inline w-3 h-3 ml-1 align-text-top" />
                </a>
                <a
                  href="https://github.com/Shannon-Labs/davinci-codex"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="hover:underline"
                >
                  GitHub
                  <ExternalLink className="inline w-3 h-3 ml-1 align-text-top" />
                </a>
              </p>
            </div>

            <div className="space-y-3">
              <div className="flex items-center gap-3">
                <h3 className="text-2xl font-bold uppercase">
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
                  className="hover:underline"
                >
                  Experience it
                  <ExternalLink className="inline w-3 h-3 ml-1 align-text-top" />
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

      {/* FOOTER */}
      <footer
        className="py-12 px-6 md:px-12 flex flex-col md:flex-row justify-between items-start md:items-center gap-8"
        style={{
          backgroundColor: "#0A0A0A",
          color: "#F5F5F0",
          borderTop: "1px solid #F5F5F0",
        }}
      >
        <div>
          <div className="font-mono text-xs space-y-2">
            <p>
              <a
                href="mailto:hunter@shannonlabs.dev"
                className="hover:underline"
              >
                [EMAIL: hunter@shannonlabs.dev]
              </a>
            </p>
          </div>
        </div>
        <div className="font-mono text-xs uppercase tracking-widest flex gap-6">
          <a
            href="https://github.com/Shannon-Labs"
            target="_blank"
            rel="noopener noreferrer"
            className="hover:bg-white hover:text-black px-2 py-1 transition-none"
          >
            [GITHUB]
          </a>
          <a
            href="https://twitter.com/huntermbown"
            target="_blank"
            rel="noopener noreferrer"
            className="hover:bg-white hover:text-black px-2 py-1 transition-none"
          >
            [TWITTER]
          </a>
          <a
            href="https://www.linkedin.com/in/hunterbown/"
            target="_blank"
            rel="noopener noreferrer"
            className="hover:bg-white hover:text-black px-2 py-1 transition-none"
          >
            [LINKEDIN]
          </a>
        </div>
      </footer>
    </div>
  );
}
