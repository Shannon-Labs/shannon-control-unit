interface SignalMarkProps {
  variant?: "light" | "dark";
  size?: number;
}

export default function SignalMark({ variant = "dark", size = 24 }: SignalMarkProps) {
  const color = variant === "dark" ? "#0A0A0A" : "#F5F5F0";
  const barWidth = size * 0.12;
  const gap = size * 0.08;
  const totalBars = 5;

  return (
    <svg
      width={size}
      height={size}
      viewBox={`0 0 ${size} ${size}`}
      fill="none"
      aria-hidden="true"
    >
      {Array.from({ length: totalBars }, (_, i) => {
        const barHeight = ((i + 1) / totalBars) * size * 0.8;
        const x = i * (barWidth + gap) + (size - totalBars * (barWidth + gap) + gap) / 2;
        const y = size - barHeight - size * 0.1;
        return (
          <rect
            key={i}
            x={x}
            y={y}
            width={barWidth}
            height={barHeight}
            fill={color}
          />
        );
      })}
    </svg>
  );
}
