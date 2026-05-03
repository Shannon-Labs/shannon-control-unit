import Header from "@/components/header";
import Footer from "@/components/Footer";

export default function Terms() {
  return (
    <div
      className="min-h-screen font-serif selection:bg-black selection:text-white mobile-overflow-x-none"
      style={{ backgroundColor: "#F5F5F0", color: "#0A0A0A" }}
    >
      <Header currentPath="/terms" />

      <main id="main-content" role="main">
        <section className="py-12 md:py-20 px-4 md:px-12 flex justify-center">
          <article className="max-w-[75ch] w-full space-y-8 md:space-y-10">
            <header className="space-y-3">
              <h1 className="text-2xl md:text-4xl lg:text-5xl font-bold uppercase tracking-tight">
                Terms of Service
              </h1>
              <p className="font-mono text-[10px] md:text-xs uppercase tracking-widest opacity-70">
                Last updated: May 3, 2026
              </p>
            </header>

            <div className="space-y-5 md:space-y-6 text-base md:text-lg leading-relaxed">
              <p>
                These Terms of Service ("Terms") govern your use of{" "}
                <a href="https://shannonlabs.dev/" className="underline">shannonlabs.dev</a>{" "}
                and any voluntary contributions you make to support the open-source work of
                Hunter Bown, operating as Shannon Labs ("Shannon Labs," "we," "us," or "our")
                from Dallas, Texas. By using the site or making a contribution, you agree to
                these Terms.
              </p>

              <h2 className="text-xl md:text-2xl font-bold uppercase pt-4">1. About the Site and the Open-Source Work</h2>
              <p>
                Shannon Labs publishes research, writing, and open-source software, including
                projects such as <em>Shannon Control Unit</em> and{" "}
                <a
                  href="https://github.com/Hmbown/DeepSeek-TUI"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="underline"
                >
                  DeepSeek-TUI
                </a>
                . The code is made available under the open-source license stated in each
                project's repository, and that license — not these Terms — governs your use of
                the code itself.
              </p>

              <h2 className="text-xl md:text-2xl font-bold uppercase pt-4">2. Voluntary Supporter Contributions</h2>
              <p>
                You can support our work as a "Hunter Bown" or "Big Whale Bro" supporter
                through Buy Me a Coffee, with payment processed by Stripe, Inc. Contributions
                are <strong>voluntary tips</strong> in support of ongoing open-source work.
              </p>
              <ul className="list-disc pl-6 space-y-2">
                <li>
                  Contributions are <strong>not</strong> a purchase of goods, software, a
                  license, custom development, support, or any guaranteed deliverable.
                </li>
                <li>
                  Becoming a supporter does not create an employment, agency, partnership, or
                  fiduciary relationship.
                </li>
                <li>
                  Any perks described on the supporter page (for example, name acknowledgement
                  or early notes) are offered as a thank-you and may change over time.
                </li>
                <li>
                  You represent that the payment method you use is yours, and that you are at
                  least 18 years old or have permission from a parent or guardian.
                </li>
              </ul>

              <h2 className="text-xl md:text-2xl font-bold uppercase pt-4">3. Payments and Billing</h2>
              <p>
                Payments are processed by Stripe and Buy Me a Coffee under their own terms.
                Charges may appear on your statement as <code>WHALE BIG BRO - MEMBER</code> or
                a similar descriptor. Recurring memberships, if you choose one, will continue
                to bill on the cadence shown at signup until you cancel through Buy Me a
                Coffee or by emailing us.
              </p>

              <h2 className="text-xl md:text-2xl font-bold uppercase pt-4">4. Refunds</h2>
              <p>
                Because contributions are voluntary tips, they are generally non-refundable.
                If you believe a charge was made in error, was unauthorized, or you cancelled
                a recurring membership and were billed anyway, email{" "}
                <a href="mailto:hunter@shannonlabs.dev" className="underline">
                  hunter@shannonlabs.dev
                </a>{" "}
                within 30 days of the charge and we will review the request in good faith and
                issue a refund where appropriate.
              </p>

              <h2 className="text-xl md:text-2xl font-bold uppercase pt-4">5. Acceptable Use</h2>
              <p>When using the site, you agree not to:</p>
              <ul className="list-disc pl-6 space-y-2">
                <li>Break the law or infringe anyone's rights.</li>
                <li>Attempt to disrupt, attack, or gain unauthorized access to the site or its infrastructure.</li>
                <li>Submit false payment information or commit chargeback fraud.</li>
                <li>Impersonate another person or misrepresent your affiliation with anyone.</li>
              </ul>

              <h2 className="text-xl md:text-2xl font-bold uppercase pt-4">6. Intellectual Property</h2>
              <p>
                Site content (text, design, logos, images) is owned by Hunter Bown / Shannon
                Labs unless otherwise noted, and is provided for personal, non-commercial
                reference. Open-source code is licensed under the terms in each project's
                repository. The Shannon Labs name and logo may not be used to imply
                endorsement without written permission.
              </p>

              <h2 className="text-xl md:text-2xl font-bold uppercase pt-4">7. Third-Party Services</h2>
              <p>
                The site links to and relies on third-party services including Stripe, Buy Me
                a Coffee, GitHub, Hugging Face, and Cloudflare. We are not responsible for
                their content, availability, or practices. Your use of those services is
                governed by their own terms and privacy policies.
              </p>

              <h2 className="text-xl md:text-2xl font-bold uppercase pt-4">8. Disclaimers</h2>
              <p>
                The site, the open-source projects, and any supporter perks are provided "AS
                IS" and "AS AVAILABLE," without warranties of any kind, whether express or
                implied, including merchantability, fitness for a particular purpose, and
                non-infringement. Nothing on the site is legal, financial, medical, or
                professional advice.
              </p>

              <h2 className="text-xl md:text-2xl font-bold uppercase pt-4">9. Limitation of Liability</h2>
              <p>
                To the fullest extent allowed by law, Shannon Labs and Hunter Bown will not be
                liable for any indirect, incidental, special, consequential, or punitive
                damages, or any loss of profits or data, arising out of your use of the site
                or any contribution you make. Our total liability for any claim relating to
                the site or your contribution will not exceed the greater of (a) the amount
                you contributed in the 12 months before the claim, or (b) USD $50.
              </p>

              <h2 className="text-xl md:text-2xl font-bold uppercase pt-4">10. Indemnification</h2>
              <p>
                You agree to defend and indemnify Shannon Labs and Hunter Bown from any
                claims, losses, or expenses (including reasonable attorneys' fees) arising
                from your misuse of the site, your violation of these Terms, or your
                infringement of any third-party rights.
              </p>

              <h2 className="text-xl md:text-2xl font-bold uppercase pt-4">11. Termination</h2>
              <p>
                We may suspend or end your access to the site or supporter program at any
                time if you violate these Terms or if continuing creates a legal or security
                risk. You can stop being a supporter at any time through Buy Me a Coffee or by
                emailing us.
              </p>

              <h2 className="text-xl md:text-2xl font-bold uppercase pt-4">12. Governing Law and Disputes</h2>
              <p>
                These Terms are governed by the laws of the State of Texas, without regard to
                its conflict-of-laws rules. Any dispute will be brought exclusively in the
                state or federal courts located in Dallas County, Texas, and you consent to
                their jurisdiction. Either party may still seek injunctive relief in any court
                of competent jurisdiction to protect intellectual property rights.
              </p>

              <h2 className="text-xl md:text-2xl font-bold uppercase pt-4">13. Changes</h2>
              <p>
                We may update these Terms as the project evolves. The "Last updated" date at
                the top reflects the most recent version. Continued use of the site after a
                change means you accept the updated Terms.
              </p>

              <h2 className="text-xl md:text-2xl font-bold uppercase pt-4">14. Contact</h2>
              <p>
                Questions about these Terms can be sent to{" "}
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

      <Footer currentPath="/terms" />
    </div>
  );
}
