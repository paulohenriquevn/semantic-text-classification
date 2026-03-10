"""Categories API router — CRUD + apply for rule-based categories."""

from __future__ import annotations

from demo.backend.dependencies import get_category_service
from demo.backend.schemas.api_models import (
    CategoryDetailResponse,
    CategoryListResponse,
    CategoryMatchResponse,
    CategoryResponse,
    CreateCategoryRequest,
    PredicateEvidenceResponse,
    PreviewDSLRequest,
    PreviewDSLResponse,
    PreviewMatchResponse,
    ValidateDSLRequest,
    ValidateDSLResponse,
)
from demo.backend.services.category_service import CategoryService
from fastapi import APIRouter, Depends, HTTPException

router = APIRouter(prefix="/categories", tags=["categories"])


def _to_response(cat: object) -> CategoryResponse:
    """Map Category dataclass to API response."""
    from demo.backend.services.category_service import Category

    assert isinstance(cat, Category)
    return CategoryResponse(
        category_id=cat.category_id,
        name=cat.name,
        dsl_expression=cat.dsl_expression,
        description=cat.description,
        match_count=cat.match_count,
        conversation_count=cat.conversation_count,
        applied=cat.applied,
        apply_time_ms=cat.apply_time_ms,
        created_at=cat.created_at,
    )


def _to_detail_response(cat: object) -> CategoryDetailResponse:
    """Map Category dataclass to detailed API response with matches."""
    from demo.backend.services.category_service import Category

    assert isinstance(cat, Category)
    matches = [
        CategoryMatchResponse(
            window_id=m.window_id,
            conversation_id=m.conversation_id,
            score=m.score,
            matched_text=m.matched_text,
        )
        for m in cat.matches
    ]
    return CategoryDetailResponse(
        category_id=cat.category_id,
        name=cat.name,
        dsl_expression=cat.dsl_expression,
        description=cat.description,
        match_count=cat.match_count,
        conversation_count=cat.conversation_count,
        applied=cat.applied,
        apply_time_ms=cat.apply_time_ms,
        created_at=cat.created_at,
        matches=matches,
    )


@router.get("", response_model=CategoryListResponse)
def list_categories(
    service: CategoryService = Depends(get_category_service),  # noqa: B008
) -> CategoryListResponse:
    """List all categories."""
    categories = service.list_categories()
    return CategoryListResponse(
        categories=[_to_response(c) for c in categories],
        total=len(categories),
    )


@router.post("", response_model=CategoryResponse, status_code=201)
def create_category(
    request: CreateCategoryRequest,
    service: CategoryService = Depends(get_category_service),  # noqa: B008
) -> CategoryResponse:
    """Create a new category with a DSL rule expression."""
    try:
        cat = service.create_category(
            name=request.name,
            dsl_expression=request.dsl_expression,
            description=request.description,
        )
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid DSL expression: {e}") from e
    return _to_response(cat)


@router.get("/{category_id}", response_model=CategoryDetailResponse)
def get_category(
    category_id: str,
    service: CategoryService = Depends(get_category_service),  # noqa: B008
) -> CategoryDetailResponse:
    """Get category details including matches."""
    cat = service.get_category(category_id)
    if cat is None:
        raise HTTPException(status_code=404, detail=f"Category not found: {category_id}")
    return _to_detail_response(cat)


@router.delete("/{category_id}", status_code=204)
def delete_category(
    category_id: str,
    service: CategoryService = Depends(get_category_service),  # noqa: B008
) -> None:
    """Delete a category."""
    if not service.delete_category(category_id):
        raise HTTPException(status_code=404, detail=f"Category not found: {category_id}")


@router.post("/{category_id}/apply", response_model=CategoryResponse)
def apply_category(
    category_id: str,
    service: CategoryService = Depends(get_category_service),  # noqa: B008
) -> CategoryResponse:
    """Apply a category's rule against all context windows."""
    try:
        cat = service.apply_category(category_id)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    return _to_response(cat)


@router.post("/validate", response_model=ValidateDSLResponse)
def validate_dsl(
    request: ValidateDSLRequest,
    service: CategoryService = Depends(get_category_service),  # noqa: B008
) -> ValidateDSLResponse:
    """Validate a DSL expression without creating a category."""
    _ = service  # ensure DI works; compiler is standalone
    from talkex.rules.compiler import SimpleRuleCompiler

    compiler = SimpleRuleCompiler()
    try:
        compiler.compile(request.dsl_expression, "validate", "validate")
        return ValidateDSLResponse(valid=True)
    except Exception as e:
        return ValidateDSLResponse(valid=False, error=str(e))


@router.post("/preview", response_model=PreviewDSLResponse)
def preview_dsl(
    request: PreviewDSLRequest,
    service: CategoryService = Depends(get_category_service),  # noqa: B008
) -> PreviewDSLResponse:
    """Dry-run a DSL expression against all windows without creating or persisting."""
    try:
        result = service.preview_dsl(request.dsl_expression)
    except Exception as e:
        return PreviewDSLResponse(valid=False, error=str(e))

    sample_matches = [
        PreviewMatchResponse(
            window_id=m["window_id"],  # type: ignore[index]
            conversation_id=m["conversation_id"],  # type: ignore[index]
            score=m["score"],  # type: ignore[index]
            window_text=m["window_text"],  # type: ignore[index]
            evidence=[
                PredicateEvidenceResponse(**ev)  # type: ignore[arg-type]
                for ev in m["evidence"]  # type: ignore[index]
            ],
        )
        for m in result["sample_matches"]  # type: ignore[union-attr]
    ]
    return PreviewDSLResponse(
        valid=True,
        match_count=result["match_count"],  # type: ignore[arg-type]
        conversation_count=result["conversation_count"],  # type: ignore[arg-type]
        sample_matches=sample_matches,
        latency_ms=result["latency_ms"],  # type: ignore[arg-type]
    )
