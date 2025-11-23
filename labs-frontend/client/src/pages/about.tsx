import { ExternalLink } from "lucide-react";
import {
  Carousel,
  CarouselContent,
  CarouselItem,
  CarouselNext,
  CarouselPrevious,
} from "@/components/ui/carousel";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Card, CardContent } from "@/components/ui/card";

export default function About() {
  return (
    <div className="min-h-screen bg-white text-black font-serif selection:bg-black selection:text-white" style={{ backgroundColor: '#F5F5F0', color: '#0A0A0A' }}>

      {/* HEADER */}
      <header className="sticky top-0 z-50 border-b flex justify-between items-center px-6 py-4 uppercase font-sans text-sm tracking-wide" style={{ backgroundColor: '#F5F5F0', borderColor: '#0A0A0A' }}>
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

          {/* PATENTS */}
          <div className="space-y-6 text-lg leading-relaxed border-t border-black pt-12">
            <h2 className="text-2xl font-bold uppercase mb-4">Legacy of Invention</h2>
            <p>
              Ralph Bown's work extended far beyond theory. His patents, particularly in radio repeater systems, laid the groundwork for modern telecommunications.
            </p>

            <Card className="border-black bg-transparent shadow-none rounded-none">
              <CardContent className="p-0">
                <Carousel className="w-full max-w-xs mx-auto md:max-w-md">
                  <CarouselContent>
                    <CarouselItem>
                      <div className="p-1">
                        <img src="/patent-1.png" alt="Patent Figure 1" className="w-full border border-black" />
                      </div>
                    </CarouselItem>
                    <CarouselItem>
                      <div className="p-1">
                        <img src="/patent-2.png" alt="Patent Figure 2" className="w-full border border-black" />
                      </div>
                    </CarouselItem>
                    <CarouselItem>
                      <div className="p-1">
                        <img src="/patent-3.png" alt="Patent Figure 3" className="w-full border border-black" />
                      </div>
                    </CarouselItem>
                  </CarouselContent>
                  <CarouselPrevious />
                  <CarouselNext />
                </Carousel>
                <p className="font-mono text-xs uppercase tracking-widest text-center mt-4">
                  US Patent 1,698,777 — Radio Repeater System
                </p>
              </CardContent>
            </Card>

            <div className="mt-8">
              <h3 className="font-mono text-sm uppercase tracking-widest mb-2">Patent Text (Excerpt)</h3>
              <ScrollArea className="h-[300px] w-full border border-black p-4 text-sm font-mono bg-white">
                <div className="space-y-4">
                  <p><strong>UNITED STATES PATENT OFFICE</strong></p>
                  <p><strong>RALPH BOWN, OF EAST ORANGE, NEW JERSEY, ASSIGNOR TO AMERICAN TELEPHONE AND TELEGRAPH COMPANY, A CORPORATION OF NEW YORK.</strong></p>
                  <p><strong>RADIO REPEATER SYSTEM</strong></p>
                  <p><strong>Application filed November 7, 1924. Serial No. 748,469.</strong></p>
                  <p>
                    This invention relates to radio repeater systems, and particularly to a system of that type characterized by unidirectional reception at the repeater station and having separate receiving circuits, each of the latter having gain control devices whereby the transmission level of the oppositely directed channels may be separately controlled, the said system being further characterized by a single antenna structure for transmitting both channels.
                  </p>
                  <p>
                    Various types of radio repeaters have been devised, one of which, known as the 22-type, embraces, in fact, two complete transmitting and receiving stations. One station receives, for example, from the west and transmits to the east and the other operates only in the opposite manner. A repeater system of that type requires, in the most general case, the use of four channels or frequency bands in order to effect east and west two-way transmission without interference. Such a system is, however, expensive to install and operate and is also open to other criticism. It possesses, however, by virtue of the use of separate transmitting and receiving apparatus for the oppositely traveling waves, the advantage that the said oppositely traveling waves may be separately amplified to a degree depending upon the transmission losses of the separate waves.
                  </p>
                  <p>
                    Another type of repeater, known as the 21 type, employs only a single transmitting and receiving station and utilizes two channels. Both the east bound and west bound signaling waves come to a single receiver upon the same carrier frequency, and both waves are repeated from the same transmitter at the same carrier frequency, the transmitting frequency, however, being different from the received carrier frequency. While such a system is simple and cheap it has certain transmission defects which limit its usefulness, the principal one of which is that the repeater, having no flexibility of adjustment, amplifies equally both the eastbound and the westbound signaling waves.
                  </p>
                  <p>
                    The object of my invention is to provide a 21-type repeater with independent unidirectional receiving systems so that two incoming messages on the same wave length, but from different directions, may be received separately and each may be brought to any desired relative or absolute transmission level before they are reradiated from a common transmitting circuit.
                  </p>
                  <p>
                    Other objects of this invention will be apparent from the following description when read in connection with the attached drawing of which Figure 1 shows schematically one form of the embodiment of the invention employing unidirectional receiving antennae of the combined vertical and loop type: Fig. 2 shows the characteristics of the receiving antennae; and Fig. 3 shows a system employing a wave antenna, the receiving characteristics of which are as shown in Fig. 2.
                  </p>
                  <p>
                    In Fig. 1 is disclosed a system employing two terminal stations A and B and a single repeater station located therebetween. At the left the line L, connects the signaling apparatus 1, which may be a subscriber's telephone set, with the terminal station A, the connection being effected by means of the hybrid coil 3. The network N connected with the hybrid coil 3 balances the line L. In similar manner, the line L, connects the telephone set 2 with the terminal station B, the connection being effected by means of the hybrid coil 19 to which is also connected the network Na to balance the line L.
                  </p>
                </div>
              </ScrollArea>
            </div>
          </div>

          {/* SYNTHESIS */}
          <div className="space-y-6 text-lg leading-relaxed border-t border-black pt-12">
            <h2 className="text-2xl font-bold uppercase mb-4">The Synthesis</h2>
            <p>
              I studied vocal science with Dr. Stephen F. Austin, learning about the "missing fundamental"—how the ear constructs a pitch that isn't physically present from its overtones. This maps directly to information theory. I've translated my lived insights as a musician to science, finding ideas left uninspected because there was no pattern-seeking researcher with my specific background.
            </p>
            <p>
              I have built three commercially viable software architectures (SCU, Driftlock, Hegelion) and one hardware thesis (Driftlock Choir). Together, they set the stage for a world of AGI and maintained human vitality. Driftlock separates signal from noise; Driftlock Choir enables communication via beat frequency; SCU provides a control mechanism for training efficiency; and Hegelion forces models into "slow thinking" to ensure only the best ideas survive.
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
