import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  Check,
  ChevronDown,
  ChevronRight,
  Loader2,
  Play,
  Plus,
  Trash2,
  X,
} from "lucide-react";
import { useState } from "react";
import {
  applyCategory,
  createCategory,
  deleteCategory,
  getCategory,
  listCategories,
  validateDSL,
} from "@/lib/api";
import type { CategoryDetailResponse, CategoryResponse } from "@/types/api";

interface CategoriesPanelProps {
  onViewConversation: (conversationId: string) => void;
}

export function CategoriesPanel({ onViewConversation }: CategoriesPanelProps) {
  const queryClient = useQueryClient();
  const [showForm, setShowForm] = useState(false);
  const [expandedId, setExpandedId] = useState<string | null>(null);

  const { data, isLoading } = useQuery({
    queryKey: ["categories"],
    queryFn: listCategories,
  });

  if (isLoading) {
    return (
      <div className="flex items-center justify-center py-20">
        <Loader2 className="h-6 w-6 animate-spin text-blue-500" />
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-lg font-semibold text-gray-900">Categories</h2>
          <p className="text-sm text-gray-500">
            Create rules to categorize conversations across the entire corpus
          </p>
        </div>
        <button
          onClick={() => setShowForm(!showForm)}
          className="flex items-center gap-1 px-3 py-2 rounded-md bg-blue-600 text-white
                     text-sm font-medium hover:bg-blue-700 transition-colors"
        >
          <Plus className="h-4 w-4" /> New Category
        </button>
      </div>

      {showForm && (
        <CreateCategoryForm
          onCreated={() => {
            setShowForm(false);
            queryClient.invalidateQueries({ queryKey: ["categories"] });
          }}
          onCancel={() => setShowForm(false)}
        />
      )}

      {/* DSL Help */}
      <DSLHelp />

      {/* Categories list */}
      <div className="space-y-2">
        {data?.categories.map((cat) => (
          <CategoryCard
            key={cat.category_id}
            category={cat}
            expanded={expandedId === cat.category_id}
            onToggle={() =>
              setExpandedId(expandedId === cat.category_id ? null : cat.category_id)
            }
            onViewConversation={onViewConversation}
            onRefresh={() =>
              queryClient.invalidateQueries({ queryKey: ["categories"] })
            }
          />
        ))}
        {data?.categories.length === 0 && !showForm && (
          <div className="text-center py-10 text-gray-400">
            No categories yet. Click "New Category" to get started.
          </div>
        )}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Create form
// ---------------------------------------------------------------------------

function CreateCategoryForm({
  onCreated,
  onCancel,
}: {
  onCreated: () => void;
  onCancel: () => void;
}) {
  const [name, setName] = useState("");
  const [dsl, setDsl] = useState("");
  const [description, setDescription] = useState("");
  const [dslValid, setDslValid] = useState<boolean | null>(null);
  const [dslError, setDslError] = useState<string | null>(null);

  const validateMutation = useMutation({
    mutationFn: () => validateDSL(dsl),
    onSuccess: (res) => {
      setDslValid(res.valid);
      setDslError(res.error);
    },
  });

  const createMutation = useMutation({
    mutationFn: () => createCategory({ name, dsl_expression: dsl, description }),
    onSuccess: onCreated,
  });

  return (
    <div className="bg-white rounded-lg border border-blue-200 p-4 shadow-sm space-y-3">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-semibold text-gray-700">New Category</h3>
        <button onClick={onCancel} className="text-gray-400 hover:text-gray-600">
          <X className="h-4 w-4" />
        </button>
      </div>

      <div>
        <label className="block text-xs font-medium text-gray-600 mb-1">
          Category Name
        </label>
        <input
          type="text"
          value={name}
          onChange={(e) => setName(e.target.value)}
          placeholder="e.g. Billing Issues"
          className="w-full px-3 py-2 rounded-md border border-gray-300 text-sm
                     focus:outline-none focus:ring-2 focus:ring-blue-500"
        />
      </div>

      <div>
        <label className="block text-xs font-medium text-gray-600 mb-1">
          DSL Rule Expression
        </label>
        <div className="relative">
          <textarea
            value={dsl}
            onChange={(e) => {
              setDsl(e.target.value);
              setDslValid(null);
              setDslError(null);
            }}
            placeholder='e.g. keyword("billing") AND keyword("charge")'
            rows={3}
            className={`w-full px-3 py-2 rounded-md border text-sm font-mono
                       focus:outline-none focus:ring-2 focus:ring-blue-500
                       ${dslValid === false ? "border-red-300 bg-red-50" : "border-gray-300"}`}
          />
          {dslValid !== null && (
            <span
              className={`absolute top-2 right-2 text-xs px-2 py-0.5 rounded ${
                dslValid
                  ? "bg-green-100 text-green-700"
                  : "bg-red-100 text-red-700"
              }`}
            >
              {dslValid ? "Valid" : "Invalid"}
            </span>
          )}
        </div>
        {dslError && (
          <p className="text-xs text-red-600 mt-1">{dslError}</p>
        )}
      </div>

      <div>
        <label className="block text-xs font-medium text-gray-600 mb-1">
          Description (optional)
        </label>
        <input
          type="text"
          value={description}
          onChange={(e) => setDescription(e.target.value)}
          placeholder="What does this category detect?"
          className="w-full px-3 py-2 rounded-md border border-gray-300 text-sm
                     focus:outline-none focus:ring-2 focus:ring-blue-500"
        />
      </div>

      {createMutation.isError && (
        <p className="text-xs text-red-600">
          {(createMutation.error as Error).message}
        </p>
      )}

      <div className="flex gap-2 justify-end">
        <button
          onClick={() => validateMutation.mutate()}
          disabled={!dsl.trim() || validateMutation.isPending}
          className="px-3 py-1.5 rounded-md border border-gray-300 text-sm text-gray-700
                     hover:bg-gray-50 disabled:opacity-50 transition-colors"
        >
          {validateMutation.isPending ? "Validating..." : "Validate DSL"}
        </button>
        <button
          onClick={() => createMutation.mutate()}
          disabled={!name.trim() || !dsl.trim() || createMutation.isPending}
          className="px-3 py-1.5 rounded-md bg-blue-600 text-white text-sm font-medium
                     hover:bg-blue-700 disabled:opacity-50 transition-colors"
        >
          {createMutation.isPending ? "Creating..." : "Create"}
        </button>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Category card
// ---------------------------------------------------------------------------

function CategoryCard({
  category,
  expanded,
  onToggle,
  onViewConversation,
  onRefresh,
}: {
  category: CategoryResponse;
  expanded: boolean;
  onToggle: () => void;
  onViewConversation: (id: string) => void;
  onRefresh: () => void;
}) {
  const applyMutation = useMutation({
    mutationFn: () => applyCategory(category.category_id),
    onSuccess: onRefresh,
  });

  const deleteMutation = useMutation({
    mutationFn: () => deleteCategory(category.category_id),
    onSuccess: onRefresh,
  });

  return (
    <div className="bg-white rounded-lg border border-gray-200 shadow-sm">
      <div
        className="flex items-center justify-between p-4 cursor-pointer hover:bg-gray-50"
        onClick={onToggle}
      >
        <div className="flex items-center gap-3">
          {expanded ? (
            <ChevronDown className="h-4 w-4 text-gray-400" />
          ) : (
            <ChevronRight className="h-4 w-4 text-gray-400" />
          )}
          <div>
            <div className="flex items-center gap-2">
              <span className="font-medium text-gray-900">{category.name}</span>
              {category.applied && (
                <span className="flex items-center gap-0.5 text-xs bg-green-100 text-green-700 px-2 py-0.5 rounded">
                  <Check className="h-3 w-3" /> Applied
                </span>
              )}
            </div>
            <p className="text-xs text-gray-500 font-mono mt-0.5">
              {category.dsl_expression}
            </p>
          </div>
        </div>

        <div className="flex items-center gap-3" onClick={(e) => e.stopPropagation()}>
          {category.applied && (
            <div className="text-right text-xs text-gray-500">
              <div>
                <span className="font-semibold text-gray-900">
                  {category.match_count}
                </span>{" "}
                windows
              </div>
              <div>
                <span className="font-semibold text-gray-900">
                  {category.conversation_count}
                </span>{" "}
                conversations
              </div>
            </div>
          )}
          <button
            onClick={() => applyMutation.mutate()}
            disabled={applyMutation.isPending}
            className="flex items-center gap-1 px-2 py-1 rounded text-xs font-medium
                       bg-green-50 text-green-700 hover:bg-green-100 disabled:opacity-50"
            title="Apply rule to all conversations"
          >
            {applyMutation.isPending ? (
              <Loader2 className="h-3 w-3 animate-spin" />
            ) : (
              <Play className="h-3 w-3" />
            )}
            {category.applied ? "Re-apply" : "Apply"}
          </button>
          <button
            onClick={() => {
              if (confirm(`Delete category "${category.name}"?`)) {
                deleteMutation.mutate();
              }
            }}
            className="p-1 rounded text-gray-400 hover:text-red-600 hover:bg-red-50"
            title="Delete category"
          >
            <Trash2 className="h-4 w-4" />
          </button>
        </div>
      </div>

      {expanded && category.applied && (
        <CategoryMatchList
          categoryId={category.category_id}
          onViewConversation={onViewConversation}
        />
      )}

      {expanded && !category.applied && (
        <div className="px-4 pb-4 text-sm text-gray-400">
          Click "Apply" to evaluate this rule against all conversations.
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Match list (loaded on expand)
// ---------------------------------------------------------------------------

function CategoryMatchList({
  categoryId,
  onViewConversation,
}: {
  categoryId: string;
  onViewConversation: (id: string) => void;
}) {
  const { data, isLoading } = useQuery({
    queryKey: ["category", categoryId],
    queryFn: () => getCategory(categoryId),
  });

  if (isLoading) {
    return (
      <div className="px-4 pb-4">
        <Loader2 className="h-4 w-4 animate-spin text-gray-400" />
      </div>
    );
  }

  const detail = data as CategoryDetailResponse;
  if (!detail?.matches?.length) {
    return <div className="px-4 pb-4 text-sm text-gray-400">No matches found.</div>;
  }

  // Group matches by conversation
  const byConv = new Map<string, typeof detail.matches>();
  for (const m of detail.matches) {
    const list = byConv.get(m.conversation_id) ?? [];
    list.push(m);
    byConv.set(m.conversation_id, list);
  }

  return (
    <div className="border-t border-gray-100">
      <div className="px-4 py-2 text-xs text-gray-500 bg-gray-50 flex justify-between">
        <span>
          {detail.match_count} window matches across {detail.conversation_count}{" "}
          conversations
        </span>
        <span className="font-mono">
          Applied in {detail.apply_time_ms} ms
        </span>
      </div>
      <div className="max-h-80 overflow-y-auto divide-y divide-gray-100">
        {[...byConv.entries()].slice(0, 50).map(([convId, matches]) => (
          <div key={convId} className="px-4 py-2 hover:bg-gray-50">
            <div className="flex items-center justify-between">
              <button
                onClick={() => onViewConversation(convId)}
                className="text-xs text-blue-600 hover:underline font-medium"
              >
                {convId}
              </button>
              <span className="text-xs text-gray-400">
                {matches.length} window{matches.length > 1 ? "s" : ""}
              </span>
            </div>
            {matches.slice(0, 3).map((m) => (
              <div key={m.window_id} className="mt-1 text-xs text-gray-500">
                {m.matched_text && (
                  <span className="bg-yellow-100 text-yellow-800 px-1 rounded">
                    {m.matched_text}
                  </span>
                )}
              </div>
            ))}
          </div>
        ))}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// DSL help reference
// ---------------------------------------------------------------------------

function DSLHelp() {
  const [open, setOpen] = useState(false);

  return (
    <div className="bg-gray-50 rounded-lg border border-gray-200">
      <button
        onClick={() => setOpen(!open)}
        className="w-full flex items-center justify-between px-4 py-2 text-sm text-gray-600"
      >
        <span className="font-medium">DSL Reference</span>
        {open ? (
          <ChevronDown className="h-4 w-4" />
        ) : (
          <ChevronRight className="h-4 w-4" />
        )}
      </button>
      {open && (
        <div className="px-4 pb-3 text-xs text-gray-600 space-y-2">
          <div>
            <span className="font-semibold">Lexical:</span>
            <code className="ml-1 bg-white px-1 rounded">keyword("billing")</code>
            <code className="ml-1 bg-white px-1 rounded">
              regex("cancel|terminate")
            </code>
          </div>
          <div>
            <span className="font-semibold">Structural:</span>
            <code className="ml-1 bg-white px-1 rounded">
              speaker("customer")
            </code>
            <code className="ml-1 bg-white px-1 rounded">
              channel("voice")
            </code>
          </div>
          <div>
            <span className="font-semibold">Boolean:</span>
            <code className="ml-1 bg-white px-1 rounded">AND</code>
            <code className="ml-1 bg-white px-1 rounded">OR</code>
            <code className="ml-1 bg-white px-1 rounded">NOT</code>
            <code className="ml-1 bg-white px-1 rounded">( )</code>
          </div>
          <div>
            <span className="font-semibold">Examples:</span>
            <div className="mt-1 space-y-1 font-mono">
              <div>keyword("refund") AND keyword("charge")</div>
              <div>keyword("cancel") OR keyword("terminate")</div>
              <div>regex("bill(ing|ed)") AND NOT keyword("paid")</div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
