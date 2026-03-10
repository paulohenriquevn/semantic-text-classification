import { useQuery } from "@tanstack/react-query";
import { getFilters } from "@/lib/api";
import type { SearchFilters } from "@/types/api";

interface FilterBarProps {
  filters: SearchFilters;
  onChange: (filters: SearchFilters) => void;
}

export function FilterBar({ filters, onChange }: FilterBarProps) {
  const { data } = useQuery({
    queryKey: ["filters"],
    queryFn: getFilters,
    staleTime: Infinity,
  });

  if (!data) return null;

  return (
    <div className="flex gap-3 items-center text-sm">
      <label className="text-gray-500 font-medium">Filters:</label>
      <select
        value={filters.domain ?? ""}
        onChange={(e) => onChange({ ...filters, domain: e.target.value || undefined })}
        className="rounded-md border border-gray-300 px-2 py-1.5 text-sm bg-white"
      >
        <option value="">All domains</option>
        {data.domains.map((d) => (
          <option key={d} value={d}>
            {d}
          </option>
        ))}
      </select>
      <select
        value={filters.topic ?? ""}
        onChange={(e) => onChange({ ...filters, topic: e.target.value || undefined })}
        className="rounded-md border border-gray-300 px-2 py-1.5 text-sm bg-white"
      >
        <option value="">All topics</option>
        {data.topics.map((t) => (
          <option key={t} value={t}>
            {t}
          </option>
        ))}
      </select>
    </div>
  );
}
