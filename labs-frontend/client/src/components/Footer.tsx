interface FooterProps {
  currentPath?: string;
}

const navLinks = [
  { href: "/", label: "HOME" },
  { href: "/art", label: "ART, EDUCATION & MUSIC" },
  { href: "/about", label: "ABOUT THE FOUNDER" },
];

const externalLinks = [
  { href: "https://github.com/Shannon-Labs", label: "GITHUB" },
  { href: "https://twitter.com/huntermbown", label: "TWITTER" },
  { href: "https://www.linkedin.com/in/hunterbown/", label: "LINKEDIN" },
  { href: "https://huggingface.co/hunterbown", label: "HUGGING FACE" },
];

export default function Footer({ currentPath }: FooterProps) {
  return (
    <footer
      className="py-8 md:py-12 px-4 md:px-12 flex flex-col md:flex-row justify-between items-start md:items-center gap-6 md:gap-8"
      style={{ backgroundColor: "#0A0A0A", color: "#F5F5F0", borderTop: "1px solid #F5F5F0" }}
      role="contentinfo"
    >
      <div>
        <div className="font-mono text-[10px] md:text-xs space-y-2">
          <p>
            <a
              href="mailto:hunter@shannonlabs.dev"
              className="hover:underline inline-block py-1 px-1 min-h-[44px] flex items-center"
            >
              [EMAIL: hunter@shannonlabs.dev]
            </a>
          </p>
        </div>
      </div>
      <nav
        className="font-mono text-[10px] md:text-xs uppercase tracking-widest flex flex-wrap gap-x-3 md:gap-x-6 gap-y-2"
        aria-label="Footer navigation"
      >
        {navLinks
          .filter((link) => link.href !== currentPath)
          .map((link) => (
            <a
              key={link.href}
              href={link.href}
              className="hover:bg-white hover:text-black px-1 py-1 min-h-[44px] flex items-center transition-none"
            >
              [{link.label}]
            </a>
          ))}
        {externalLinks.map((link) => (
          <a
            key={link.href}
            href={link.href}
            target="_blank"
            rel="noopener noreferrer"
            className="hover:bg-white hover:text-black px-1 py-1 min-h-[44px] flex items-center transition-none"
          >
            [{link.label}]
          </a>
        ))}
      </nav>
    </footer>
  );
}
