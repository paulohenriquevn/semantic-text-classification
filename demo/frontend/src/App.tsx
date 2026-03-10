import { useMutation } from "@tanstack/react-query";
import { useState } from "react";
import { AnalyticsPanel } from "@/components/AnalyticsPanel";
import { CategoriesPanel } from "@/components/CategoriesPanel";
import { ConversationView } from "@/components/ConversationView";
import { FilterBar } from "@/components/FilterBar";
import { ResultCard } from "@/components/ResultCard";
import { SearchBar } from "@/components/SearchBar";
import { searchConversations } from "@/lib/api";
import type { SearchFilters, SearchResponse } from "@/types/api";

type Tab = "search" | "categories" | "analytics";

export default function App() {
  const [tab, setTab] = useState<Tab>("search");
  const [filters, setFilters] = useState<SearchFilters>({});
  const [searchResult, setSearchResult] = useState<SearchResponse | null>(null);
  const [selectedConversation, setSelectedConversation] = useState<string | null>(null);

  const searchMutation = useMutation({
    mutationFn: (query: string) =>
      searchConversations({ query, filters, top_k: 20 }),
    onSuccess: (data) => {
      setSearchResult(data);
      setSelectedConversation(null);
    },
  });

  // Navigate to conversation view from any tab
  const viewConversation = (conversationId: string) => {
    setSelectedConversation(conversationId);
    setTab("search");
  };

  return (
    <div className="min-h-screen">
      {/* Header */}
      <header className="bg-white border-b border-gray-200 shadow-sm">
        <div className="max-w-6xl mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-xl font-bold text-gray-900">
                TalkEx <span className="text-blue-600">Demo</span>
              </h1>
              <p className="text-xs text-gray-500">Conversation Intelligence Engine</p>
            </div>
            <nav className="flex gap-1">
              <TabButton active={tab === "search"} onClick={() => setTab("search")}>
                Search
              </TabButton>
              <TabButton
                active={tab === "categories"}
                onClick={() => setTab("categories")}
              >
                Categories
              </TabButton>
              <TabButton active={tab === "analytics"} onClick={() => setTab("analytics")}>
                Analytics
              </TabButton>
            </nav>
          </div>
        </div>
      </header>

      {/* Main content */}
      <main className="max-w-6xl mx-auto px-4 py-6">
        {selectedConversation ? (
          <ConversationView
            conversationId={selectedConversation}
            onBack={() => setSelectedConversation(null)}
          />
        ) : tab === "analytics" ? (
          <AnalyticsPanel />
        ) : tab === "categories" ? (
          <CategoriesPanel onViewConversation={viewConversation} />
        ) : (
          <div className="space-y-4">
            <SearchBar
              onSearch={(q) => searchMutation.mutate(q)}
              isLoading={searchMutation.isPending}
            />

            <FilterBar filters={filters} onChange={setFilters} />

            {/* Latency banner */}
            {searchResult && (
              <div className="flex items-center justify-between text-sm text-gray-500">
                <span>
                  {searchResult.results.length} results from{" "}
                  {searchResult.total_candidates} candidates
                </span>
                <span className="font-mono text-xs bg-gray-100 px-2 py-0.5 rounded">
                  Search completed in {searchResult.latency_ms} ms
                </span>
              </div>
            )}

            {/* Error */}
            {searchMutation.isError && (
              <div className="bg-red-50 border border-red-200 rounded-lg p-3 text-sm text-red-700">
                Search failed: {(searchMutation.error as Error).message}
              </div>
            )}

            {/* Results */}
            {searchResult && (
              <div className="space-y-3">
                {searchResult.results.map((hit) => (
                  <ResultCard
                    key={hit.window_id}
                    hit={hit}
                    onClick={setSelectedConversation}
                  />
                ))}
                {searchResult.results.length === 0 && (
                  <div className="text-center py-10 text-gray-400">
                    No results found. Try a different query.
                  </div>
                )}
              </div>
            )}

            {/* Empty state */}
            {!searchResult && !searchMutation.isPending && (
              <div className="text-center py-20 text-gray-400">
                <p className="text-lg mb-2">
                  Search thousands of call center conversations
                </p>
                <p className="text-sm">
                  Try: "customer complaint", "refund request", "technical support"
                </p>
              </div>
            )}
          </div>
        )}
      </main>
    </div>
  );
}

function TabButton({
  active,
  onClick,
  children,
}: {
  active: boolean;
  onClick: () => void;
  children: React.ReactNode;
}) {
  return (
    <button
      onClick={onClick}
      className={`px-4 py-2 rounded-md text-sm font-medium transition-colors ${
        active
          ? "bg-blue-50 text-blue-700"
          : "text-gray-500 hover:text-gray-700 hover:bg-gray-50"
      }`}
    >
      {children}
    </button>
  );
}
