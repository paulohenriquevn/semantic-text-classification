/**
 * Shared text highlighter — marks matched fragments with yellow background.
 *
 * Used by SearchBuilderPanel, CategoriesPanel, ConversationView, ResultCard,
 * and DSLGuidePanel for consistent text highlighting across the app.
 */

export function HighlightedText({
  text,
  fragments,
}: {
  text: string;
  fragments: string[];
}) {
  if (fragments.length === 0) return <>{text}</>;

  const sorted = [...fragments].sort((a, b) => b.length - a.length);
  const escaped = sorted.map((f) => f.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"));
  const pattern = new RegExp(`(${escaped.join("|")})`, "gi");
  const parts = text.split(pattern);

  return (
    <>
      {parts.map((part, i) => {
        const isMatch = fragments.some(
          (f) => f.toLowerCase() === part.toLowerCase(),
        );
        return isMatch ? (
          <mark key={i} className="bg-yellow-200 text-yellow-900 rounded-sm px-0.5">
            {part}
          </mark>
        ) : (
          <span key={i}>{part}</span>
        );
      })}
    </>
  );
}
