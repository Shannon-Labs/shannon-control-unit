import Header from "@/components/header";
import Footer from "@/components/Footer";

export default function Privacy() {
  return (
    <div
      className="min-h-screen font-serif selection:bg-black selection:text-white mobile-overflow-x-none"
      style={{ backgroundColor: "#F5F5F0", color: "#0A0A0A" }}
    >
      <Header currentPath="/privacy" />

      <main id="main-content" role="main">
        <section className="py-12 md:py-20 px-4 md:px-12 flex justify-center">
          <article className="max-w-[75ch] w-full space-y-8 md:space-y-10">
            <header className="space-y-3">
              <h1 className="text-2xl md:text-4xl lg:text-5xl font-bold uppercase tracking-tight">
                Privacy Policy
              </h1>
              <p className="font-mono text-[10px] md:text-xs uppercase tracking-widest opacity-70">
                Last updated: May 3, 2026
              </p>
            </header>

            <div className="space-y-5 md:space-y-6 text-base md:text-lg leading-relaxed">
              <p>
                This Privacy Policy describes how Shannon Labs ("we," "us," or "our"), a
                sole proprietorship operated by Hunter Bown in Dallas, Texas, collects, uses,
                and discloses information when you visit{" "}
                <a href="https://shannonlabs.dev/" className="underline">shannonlabs.dev</a>,
                contact us, or support our open-source work through Buy Me a Coffee.
              </p>

              <h2 className="text-xl md:text-2xl font-bold uppercase pt-4">1. Information We Collect</h2>
              <p>We collect only what is needed to operate the site and process voluntary support payments.</p>
              <ul className="list-disc pl-6 space-y-2">
                <li>
                  <strong>Information you give us.</strong> When you email us or sign up to
                  support our work as a "Hunter Bown" / "Big Whale Bro" supporter, you may
                  provide your name, email address, and any message you choose to send.
                </li>
                <li>
                  <strong>Payment information.</strong> If you contribute via Buy Me a Coffee,
                  payment is processed by Stripe, Inc. and Buy Me a Coffee. We do not see or
                  store your full card number, CVV, or bank account details. We receive only
                  limited transaction metadata (e.g., supporter name or handle, email if you
                  share it, amount, currency, country, last four digits of the card, and a
                  Stripe transaction ID).
                </li>
                <li>
                  <strong>Automatically collected information.</strong> Our hosting provider
                  (Cloudflare) and any analytics we enable may log standard request data such
                  as IP address, user-agent, referrer, and pages visited, for security and
                  basic traffic measurement.
                </li>
              </ul>

              <h2 className="text-xl md:text-2xl font-bold uppercase pt-4">2. How We Use Information</h2>
              <ul className="list-disc pl-6 space-y-2">
                <li>To respond to your messages and support requests.</li>
                <li>To process voluntary contributions and acknowledge supporters.</li>
                <li>To operate, secure, and improve the website.</li>
                <li>To comply with legal, tax, and accounting obligations.</li>
              </ul>
              <p>
                We do not sell your personal information, and we do not use it for advertising
                or profiling.
              </p>

              <h2 className="text-xl md:text-2xl font-bold uppercase pt-4">3. Who We Share It With</h2>
              <p>We disclose information only to the parties needed to run this site and accept payments:</p>
              <ul className="list-disc pl-6 space-y-2">
                <li>
                  <strong>Stripe, Inc.</strong> — payment processing. See Stripe's{" "}
                  <a
                    href="https://stripe.com/privacy"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="underline"
                  >
                    Privacy Policy
                  </a>.
                </li>
                <li>
                  <strong>Buy Me a Coffee</strong> — supporter platform front-end. See their{" "}
                  <a
                    href="https://www.buymeacoffee.com/privacy-policy"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="underline"
                  >
                    Privacy Policy
                  </a>.
                </li>
                <li>
                  <strong>Cloudflare, Inc.</strong> — website hosting, DNS, and security.
                </li>
                <li>
                  <strong>Email and infrastructure providers</strong> we use to send and store
                  correspondence (for example, Google).
                </li>
                <li>
                  <strong>Government or legal authorities</strong> when required by law,
                  subpoena, or to protect our rights or the safety of others.
                </li>
              </ul>
              <p>
                Disclosure happens through standard, encrypted API and web traffic to these
                providers. We do not post supporter information publicly without permission.
              </p>

              <h2 className="text-xl md:text-2xl font-bold uppercase pt-4">4. How Long We Keep It</h2>
              <p>
                We retain transaction records for as long as required by U.S. tax and
                accounting law (generally up to seven years). Email correspondence is kept
                until it is no longer needed for the purpose it was sent. You may ask us to
                delete personal information we hold by emailing the address below; we will
                honor the request unless we are legally required to keep it.
              </p>

              <h2 className="text-xl md:text-2xl font-bold uppercase pt-4">5. Security</h2>
              <p>
                We use HTTPS across the site, rely on Stripe and Buy Me a Coffee for
                PCI-compliant payment handling, restrict access to administrative tools to the
                site owner, and use multi-factor authentication on the email and platform
                accounts that hold any supporter data. No method of transmission over the
                internet is 100% secure, and we cannot guarantee absolute security.
              </p>

              <h2 className="text-xl md:text-2xl font-bold uppercase pt-4">6. Your Choices and Rights</h2>
              <p>
                You can email us to access, correct, or delete personal information we hold
                about you, or to opt out of any future supporter-update emails. Depending on
                where you live (for example, California, Texas, or the EU/UK), you may have
                additional rights under applicable law; we will honor valid requests.
              </p>

              <h2 className="text-xl md:text-2xl font-bold uppercase pt-4">7. Children</h2>
              <p>
                Shannon Labs is not directed to children under 13, and we do not knowingly
                collect personal information from them.
              </p>

              <h2 className="text-xl md:text-2xl font-bold uppercase pt-4">8. Changes</h2>
              <p>
                We may update this policy as the project evolves. The "Last updated" date at
                the top reflects the most recent version.
              </p>

              <h2 className="text-xl md:text-2xl font-bold uppercase pt-4">9. Contact</h2>
              <p>
                Questions or requests about this policy can be sent to{" "}
                <a href="mailto:hunter@shannonlabs.dev" className="underline">
                  hunter@shannonlabs.dev
                </a>
                , or by mail to:
              </p>
              <p className="font-mono text-sm md:text-base">
                Shannon Labs / Hunter Bown
                <br />
                2626 Throckmorton St, Apt 1149
                <br />
                Dallas, TX 75219
                <br />
                United States
              </p>
            </div>
          </article>
        </section>
      </main>

      <Footer currentPath="/privacy" />
    </div>
  );
}
