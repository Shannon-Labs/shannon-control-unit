interface QuoteBlockProps {
  quote: string;
  attribution: string;
  source?: string;
}

export default function QuoteBlock({ quote, attribution, source }: QuoteBlockProps) {
  return (
    <article className="max-w-[75ch] w-full" style={{ backgroundColor: "#FFFFFF", color: "#0A0A0A" }}>
      <div className="p-6 md:p-10 lg:p-14 border" style={{ borderColor: "#0A0A0A" }}>
        <p className="font-serif text-lg md:text-2xl lg:text-3xl leading-relaxed italic">
          {quote}
        </p>
        <div
          className="text-center font-mono text-xs md:text-sm uppercase tracking-wider border-t pt-4 md:pt-6 mt-6 md:mt-8"
          style={{ color: "#0A0A0A", borderColor: "#0A0A0A" }}
        >
          — {attribution}
          {source && <>, <em>{source}</em></>}
        </div>
      </div>
    </article>
  );
}
