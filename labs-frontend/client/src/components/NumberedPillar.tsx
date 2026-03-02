interface NumberedPillarProps {
  number: string;
  heading: string;
  body: string;
  variant?: "light" | "dark";
}

export default function NumberedPillar({ number, heading, body, variant = "dark" }: NumberedPillarProps) {
  const isDark = variant === "dark";
  return (
    <div
      className="p-4 md:p-6 border-b"
      style={{
        borderColor: isDark ? "#F5F5F0" : "#0A0A0A",
        color: isDark ? "#F5F5F0" : "#0A0A0A",
      }}
    >
      <div
        className="font-mono text-[10px] md:text-xs uppercase tracking-widest mb-1 md:mb-2"
        style={{ opacity: 0.7 }}
      >
        [{number}]
      </div>
      <h4 className="font-mono text-xs md:text-sm font-bold mb-1 md:mb-2 uppercase tracking-wider">
        {heading}
      </h4>
      <p className="font-serif text-xs md:text-sm leading-relaxed opacity-90">{body}</p>
    </div>
  );
}
