import { Loader2, Search } from "lucide-react";
import { useEffect, useState } from "react";

// ---------------------------------------------------------------------------
// Elapsed timer hook
// ---------------------------------------------------------------------------

function useElapsed(running: boolean): number {
  const [startTime, setStartTime] = useState<number | null>(null);
  const [elapsed, setElapsed] = useState(0);

  useEffect(() => {
    if (running) {
      setStartTime(Date.now());
      setElapsed(0);
    } else {
      setStartTime(null);
    }
  }, [running]);

  useEffect(() => {
    if (startTime === null) return;
    const interval = setInterval(() => {
      setElapsed(Date.now() - startTime);
    }, 50);
    return () => clearInterval(interval);
  }, [startTime]);

  return elapsed;
}

// ---------------------------------------------------------------------------
// Loading steps — visual feedback during search
// ---------------------------------------------------------------------------

const LOADING_STEPS = [
  { ms: 0, label: "Iniciando busca..." },
  { ms: 300, label: "Processando embeddings..." },
  { ms: 800, label: "Buscando nos índices BM25 + ANN..." },
  { ms: 1500, label: "Analisando resultados..." },
  { ms: 2500, label: "Destacando trechos relevantes..." },
  { ms: 4000, label: "Quase lá..." },
];

function getCurrentStep(elapsed: number): string {
  let step = LOADING_STEPS[0].label;
  for (const s of LOADING_STEPS) {
    if (elapsed >= s.ms) step = s.label;
  }
  return step;
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

interface SearchLoadingProps {
  isLoading: boolean;
}

export function SearchLoading({ isLoading }: SearchLoadingProps) {
  const elapsed = useElapsed(isLoading);

  if (!isLoading) return null;

  const seconds = (elapsed / 1000).toFixed(1);
  const currentStep = getCurrentStep(elapsed);

  return (
    <div className="bg-white rounded-xl border border-blue-200 shadow-sm overflow-hidden">
      {/* Progress bar */}
      <div className="h-1 bg-blue-100 overflow-hidden">
        <div className="h-full bg-blue-500 animate-loading-bar" />
      </div>

      <div className="px-6 py-8 flex flex-col items-center gap-4">
        {/* Animated icon */}
        <div className="relative">
          <div className="absolute inset-0 bg-blue-100 rounded-full animate-ping opacity-30" />
          <div className="relative bg-blue-50 rounded-full p-4">
            <Search className="h-6 w-6 text-blue-500 animate-pulse" />
          </div>
        </div>

        {/* Step label */}
        <p className="text-sm font-medium text-gray-700">{currentStep}</p>

        {/* Timer */}
        <div className="flex items-center gap-2">
          <Loader2 className="h-3.5 w-3.5 animate-spin text-blue-500" />
          <span className="text-lg font-mono font-semibold text-blue-600 tabular-nums">
            {seconds}s
          </span>
        </div>

        {/* Subtitle */}
        <p className="text-xs text-gray-400">
          Buscando em 944 conversas &middot; 6.421 janelas de contexto
        </p>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Compact variant for DSL Builder / Categories
// ---------------------------------------------------------------------------

interface SearchLoadingCompactProps {
  isLoading: boolean;
  label?: string;
}

export function SearchLoadingCompact({
  isLoading,
  label = "Processando...",
}: SearchLoadingCompactProps) {
  const elapsed = useElapsed(isLoading);

  if (!isLoading) return null;

  const seconds = (elapsed / 1000).toFixed(1);

  return (
    <div className="flex items-center justify-center gap-3 py-6">
      <Loader2 className="h-4 w-4 animate-spin text-blue-500" />
      <span className="text-sm text-gray-600">{label}</span>
      <span className="text-sm font-mono font-semibold text-blue-600 tabular-nums">
        {seconds}s
      </span>
    </div>
  );
}
