/** API response types matching the FastAPI backend schemas. */

export interface SearchFilters {
  speaker?: string;
  domain?: string;
  topic?: string;
}

export interface SearchRequest {
  query: string;
  filters?: SearchFilters;
  top_k?: number;
}

export interface SearchHit {
  window_id: string;
  conversation_id: string;
  text: string;
  lexical_score: number | null;
  semantic_score: number | null;
  score: number;
  rank: number;
  domain: string;
  topic: string;
}

export interface SearchResponse {
  results: SearchHit[];
  total_candidates: number;
  query: string;
  latency_ms: number;
}

export interface TurnResponse {
  turn_id: string;
  speaker: string;
  raw_text: string;
  normalized_text: string | null;
}

export interface ConversationResponse {
  conversation_id: string;
  domain: string;
  topic: string;
  asr_confidence: number;
  audio_duration_seconds: number;
  turn_count: number;
  window_count: number;
  text_preview: string;
  turns: TurnResponse[];
}

export interface DomainCount {
  domain: string;
  count: number;
}

export interface FiltersResponse {
  domains: string[];
  topics: string[];
  speakers: string[];
}

export interface AnalyticsSummary {
  total_conversations: number;
  total_windows: number;
  total_embeddings: number;
  avg_asr_confidence: number;
  avg_turns_per_conversation: number;
  domains: DomainCount[];
  embedding_model: string;
  index_dimensions: number;
}

// ---------------------------------------------------------------------------
// Categories
// ---------------------------------------------------------------------------

export interface CreateCategoryRequest {
  name: string;
  dsl_expression: string;
  description?: string;
}

export interface CategoryMatch {
  window_id: string;
  conversation_id: string;
  score: number;
  matched_text: string | null;
}

export interface CategoryResponse {
  category_id: string;
  name: string;
  dsl_expression: string;
  description: string;
  match_count: number;
  conversation_count: number;
  applied: boolean;
  apply_time_ms: number;
  created_at: string;
}

export interface CategoryDetailResponse extends CategoryResponse {
  matches: CategoryMatch[];
}

export interface CategoryListResponse {
  categories: CategoryResponse[];
  total: number;
}

export interface ValidateDSLResponse {
  valid: boolean;
  error: string | null;
}
