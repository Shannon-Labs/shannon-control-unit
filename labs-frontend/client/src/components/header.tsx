import { useState } from "react";
import { Menu, X } from "lucide-react";
import { useIsMobile } from "@/hooks/use-mobile";

interface NavItem {
  href: string;
  label: string;
}

interface HeaderProps {
  currentPath?: string;
}

const navItems: NavItem[] = [
  { href: "/", label: "HOME" },
  { href: "/art", label: "ART, EDUCATION & MUSIC" },
  { href: "/about", label: "ABOUT THE FOUNDER" },
];

export default function Header({ currentPath = "/" }: HeaderProps) {
  const [isMenuOpen, setIsMenuOpen] = useState(false);
  const isMobile = useIsMobile();

  const toggleMenu = () => setIsMenuOpen(!isMenuOpen);
  const closeMenu = () => setIsMenuOpen(false);

  return (
    <>
      <a
        href="#main-content"
        className="sr-only focus:not-sr-only focus:absolute focus:top-2 focus:left-2 focus:z-50 focus:px-4 focus:py-2 focus:bg-black focus:text-white focus:font-mono focus:text-sm focus:rounded-none"
      >
        Skip to main content
      </a>

      <header
        className="sticky top-0 z-50 border-b bg-[#F5F5F0]"
        style={{ borderColor: "#0A0A0A" }}
        role="banner"
      >
        <div className="flex justify-between items-center px-4 md:px-6 py-3 md:py-4">
          <a
            href="/"
            className="font-mono font-bold text-sm md:text-base flex items-center gap-2 md:gap-3 min-h-[44px] min-w-[44px] flex items-center justify-center uppercase tracking-wider"
            style={{ color: "#0A0A0A" }}
            aria-label="Shannon Labs Home"
          >
            SHANNON LABS
          </a>

          {isMobile ? (
            <>
              <button
                onClick={toggleMenu}
                className="md:hidden p-2 min-h-[44px] min-w-[44px] flex items-center justify-center font-mono text-xs uppercase tracking-wide transition-none"
                style={isMenuOpen ? { backgroundColor: "#0A0A0A", color: "#F5F5F0" } : {}}
                aria-label={isMenuOpen ? "Close menu" : "Open menu"}
                aria-expanded={isMenuOpen}
                aria-controls="mobile-nav"
              >
                {isMenuOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
              </button>

              <nav
                id="mobile-nav"
                className={`absolute top-full left-0 right-0 border-b bg-[#F5F5F0] overflow-hidden transition-all duration-200 ${
                  isMenuOpen ? "max-h-[500px] opacity-100" : "max-h-0 opacity-0"
                }`}
                style={{ borderColor: "#0A0A0A" }}
                role="navigation"
                aria-label="Mobile navigation"
              >
                <ul className="py-2 px-4">
                  {navItems.map((item) => (
                    <li key={item.href}>
                      <a
                        href={item.href}
                        onClick={closeMenu}
                        className="block py-3 px-2 font-mono text-xs uppercase tracking-wide min-h-[44px] flex items-center transition-none"
                        style={
                          currentPath === item.href
                            ? { backgroundColor: "#0A0A0A", color: "#F5F5F0" }
                            : {}
                        }
                        onMouseEnter={(e) => {
                          if (currentPath !== item.href) {
                            e.currentTarget.style.backgroundColor = "#0A0A0A";
                            e.currentTarget.style.color = "#F5F5F0";
                          }
                        }}
                        onMouseLeave={(e) => {
                          if (currentPath !== item.href) {
                            e.currentTarget.style.backgroundColor = "";
                            e.currentTarget.style.color = "";
                          }
                        }}
                        aria-current={currentPath === item.href ? "page" : undefined}
                      >
                        [{item.label}]
                      </a>
                    </li>
                  ))}
                </ul>
              </nav>
            </>
          ) : (
            <nav role="navigation" aria-label="Main navigation">
              <ul className="flex gap-1 md:gap-2">
                {navItems.map((item) => (
                  <li key={item.href}>
                    <a
                      href={item.href}
                      className="block px-2 py-1 font-mono text-[10px] md:text-xs uppercase tracking-wide min-h-[44px] flex items-center transition-none"
                      style={
                        currentPath === item.href
                          ? { backgroundColor: "#0A0A0A", color: "#F5F5F0" }
                          : {}
                      }
                      onMouseEnter={(e) => {
                        if (currentPath !== item.href) {
                          e.currentTarget.style.backgroundColor = "#0A0A0A";
                          e.currentTarget.style.color = "#F5F5F0";
                        }
                      }}
                      onMouseLeave={(e) => {
                        if (currentPath !== item.href) {
                          e.currentTarget.style.backgroundColor = "";
                          e.currentTarget.style.color = "";
                        }
                      }}
                      aria-current={currentPath === item.href ? "page" : undefined}
                    >
                      [{item.label}]
                    </a>
                  </li>
                ))}
              </ul>
            </nav>
          )}
        </div>
      </header>
    </>
  );
}
