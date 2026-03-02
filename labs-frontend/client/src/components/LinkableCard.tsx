import { ExternalLink } from "lucide-react";

interface LinkableCardProps {
  title: string;
  description: string;
  link: string;
  icon?: string;
  variant?: "light" | "highlight";
}

export default function LinkableCard({
  title,
  description,
  link,
  icon,
  variant = "light",
}: LinkableCardProps) {
  const isHighlight = variant === "highlight";
  const isExternal = link.startsWith("http");

  return (
    <a
      href={link}
      target={isExternal ? "_blank" : undefined}
      rel={isExternal ? "noopener noreferrer" : undefined}
      className={`relative border-r border-b p-4 md:p-8 flex flex-col justify-between h-full min-h-[280px] md:min-h-[320px] group overflow-hidden cursor-pointer transition-none ${
        isHighlight
          ? "hover:invert"
          : "hover:bg-black hover:text-white"
      }`}
      style={{
        backgroundColor: isHighlight ? "#000000" : "#FFFFFF",
        color: isHighlight ? "#FFFFFF" : "#0A0A0A",
        borderColor: "#0A0A0A",
      }}
    >
      <div className="relative z-10">
        <div
          className={`flex justify-between items-start mb-4 md:mb-6 pb-3 md:pb-4 border-b ${
            isHighlight ? "border-white/40" : "border-black/20"
          }`}
        >
          <div className="flex items-center gap-2 md:gap-3 flex-1 pr-2 md:pr-4 overflow-hidden">
            {icon && (
              <img
                src={icon}
                alt={`${title} Logo`}
                className={`h-6 w-6 md:h-8 md:w-8 object-contain flex-shrink-0 ${
                  isHighlight ? "invert" : "group-hover:invert"
                }`}
              />
            )}
            <h3 className="text-base md:text-xl font-mono font-bold uppercase tracking-tighter truncate">
              {title}
            </h3>
          </div>
          <ExternalLink
            className="w-4 h-4 ml-1 md:ml-2 flex-shrink-0 mt-0.5 md:mt-1"
            strokeWidth={1.5}
          />
        </div>
        <p className="font-mono text-xs leading-relaxed opacity-80 line-clamp-3 md:line-clamp-none">
          {description}
        </p>
      </div>
    </a>
  );
}
