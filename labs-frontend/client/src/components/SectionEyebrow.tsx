interface SectionEyebrowProps {
  label: string;
  variant?: "light" | "dark";
}

export default function SectionEyebrow({ label, variant = "dark" }: SectionEyebrowProps) {
  return (
    <div
      className="px-3 md:px-4 py-3 font-mono text-[10px] uppercase tracking-widest border-b"
      style={{
        backgroundColor: variant === "dark" ? "#0A0A0A" : "#F5F5F0",
        color: variant === "dark" ? "#F5F5F0" : "#0A0A0A",
        borderColor: variant === "dark" ? "#F5F5F0" : "#0A0A0A",
      }}
    >
      // {label}
    </div>
  );
}
