/** API client — thin wrapper over fetch for the TalkEx demo backend. */

import type {
  AnalyticsSummary,
  CategoryDetailResponse,
  CategoryListResponse,
  CategoryResponse,
  ConversationResponse,
  CreateCategoryRequest,
  FiltersResponse,
  SearchRequest,
  SearchResponse,
  ValidateDSLResponse,
} from "@/types/api";

const BASE = "/api";

async function fetchJson<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, init);
  if (!res.ok) {
    const text = await res.text().catch(() => res.statusText);
    throw new Error(text || `API error: ${res.status}`);
  }
  return res.json() as Promise<T>;
}

export function searchConversations(req: SearchRequest): Promise<SearchResponse> {
  return fetchJson<SearchResponse>("/search", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });
}

export function getConversation(id: string): Promise<ConversationResponse> {
  return fetchJson<ConversationResponse>(`/conversation/${encodeURIComponent(id)}`);
}

export function getFilters(): Promise<FiltersResponse> {
  return fetchJson<FiltersResponse>("/filters");
}

export function getAnalytics(): Promise<AnalyticsSummary> {
  return fetchJson<AnalyticsSummary>("/analytics/summary");
}

// ---------------------------------------------------------------------------
// Categories
// ---------------------------------------------------------------------------

export function listCategories(): Promise<CategoryListResponse> {
  return fetchJson<CategoryListResponse>("/categories");
}

export function getCategory(id: string): Promise<CategoryDetailResponse> {
  return fetchJson<CategoryDetailResponse>(`/categories/${encodeURIComponent(id)}`);
}

export function createCategory(req: CreateCategoryRequest): Promise<CategoryResponse> {
  return fetchJson<CategoryResponse>("/categories", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });
}

export function deleteCategory(id: string): Promise<void> {
  return fetch(`${BASE}/categories/${encodeURIComponent(id)}`, { method: "DELETE" }).then(
    (res) => {
      if (!res.ok) throw new Error(`Delete failed: ${res.status}`);
    },
  );
}

export function applyCategory(id: string): Promise<CategoryResponse> {
  return fetchJson<CategoryResponse>(`/categories/${encodeURIComponent(id)}/apply`, {
    method: "POST",
  });
}

export function validateDSL(dsl: string): Promise<ValidateDSLResponse> {
  return fetchJson<ValidateDSLResponse>("/categories/validate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ dsl_expression: dsl }),
  });
}
