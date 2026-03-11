import { Search, Sparkles } from "lucide-react";
import { useState } from "react";

// ---------------------------------------------------------------------------
// Quick action definitions — PT-BR queries matching the demo dataset
// ---------------------------------------------------------------------------

interface QuickAction {
  label: string;
  query: string;
  color: string;
}

const QUICK_ACTIONS: QuickAction[] = [
  {
    label: "Cancelamento",
    query: "cliente quer cancelar o plano ou serviço",
    color: "bg-red-50 text-red-700 border-red-200 hover:bg-red-100",
  },
  {
    label: "Reclamação",
    query: "reclamação insatisfação com atendimento ou serviço",
    color: "bg-orange-50 text-orange-700 border-orange-200 hover:bg-orange-100",
  },
  {
    label: "Suporte técnico",
    query: "problema técnico internet não funciona erro no sistema",
    color: "bg-blue-50 text-blue-700 border-blue-200 hover:bg-blue-100",
  },
  {
    label: "Cobrança indevida",
    query: "cobrança indevida valor errado na fatura",
    color: "bg-yellow-50 text-yellow-700 border-yellow-200 hover:bg-yellow-100",
  },
  {
    label: "Compra",
    query: "quero contratar comprar um plano novo",
    color: "bg-emerald-50 text-emerald-700 border-emerald-200 hover:bg-emerald-100",
  },
  {
    label: "Elogio",
    query: "elogio satisfação obrigado excelente atendimento",
    color: "bg-green-50 text-green-700 border-green-200 hover:bg-green-100",
  },
  {
    label: "Dúvida produto",
    query: "dúvida sobre produto preço funcionalidade",
    color: "bg-indigo-50 text-indigo-700 border-indigo-200 hover:bg-indigo-100",
  },
  {
    label: "Prazo entrega",
    query: "prazo de entrega pedido não chegou atraso",
    color: "bg-purple-50 text-purple-700 border-purple-200 hover:bg-purple-100",
  },
  {
    label: "Reembolso",
    query: "quero meu dinheiro de volta reembolso estorno",
    color: "bg-pink-50 text-pink-700 border-pink-200 hover:bg-pink-100",
  },
  {
    label: "Agendamento",
    query: "agendar consulta horário disponível marcar",
    color: "bg-teal-50 text-teal-700 border-teal-200 hover:bg-teal-100",
  },
];

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

interface SearchBarProps {
  onSearch: (query: string) => void;
  isLoading: boolean;
}

export function SearchBar({ onSearch, isLoading }: SearchBarProps) {
  const [query, setQuery] = useState("");

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (query.trim()) {
      onSearch(query.trim());
    }
  };

  const handleQuickAction = (action: QuickAction) => {
    setQuery(action.query);
    onSearch(action.query);
  };

  return (
    <div className="space-y-3">
      <form onSubmit={handleSubmit} className="w-full">
        <div className="relative">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-5 w-5 text-gray-400" />
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Buscar conversas... (ex: 'cliente quer cancelar o plano')"
            className="w-full pl-10 pr-24 py-3 rounded-lg border border-gray-300 bg-white
                       text-gray-900 placeholder-gray-400 shadow-sm
                       focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
            disabled={isLoading}
          />
          <button
            type="submit"
            disabled={isLoading || !query.trim()}
            className="absolute right-2 top-1/2 -translate-y-1/2 px-4 py-1.5 rounded-md
                       bg-blue-600 text-white text-sm font-medium
                       hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed
                       transition-colors"
          >
            {isLoading ? "Buscando..." : "Buscar"}
          </button>
        </div>
      </form>

      {/* Quick actions */}
      <div className="flex items-center gap-2 flex-wrap">
        <span className="flex items-center gap-1 text-xs text-gray-400 shrink-0">
          <Sparkles className="h-3 w-3" />
          Ações rápidas:
        </span>
        {QUICK_ACTIONS.map((action) => (
          <button
            key={action.label}
            onClick={() => handleQuickAction(action)}
            disabled={isLoading}
            className={`text-xs font-medium px-2.5 py-1 rounded-full border transition-all
                       disabled:opacity-50 disabled:cursor-not-allowed ${action.color}`}
          >
            {action.label}
          </button>
        ))}
      </div>
    </div>
  );
}
