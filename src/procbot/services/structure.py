"""Work structure used by tasks: projects, milestones (lite), categories, contexts.

Owner of the `projects`, `milestones`, `categories` and `contexts` tables in
stage A. Goals proper (roadmaps, progress, FR-10) arrive with GoalService in
stage C; until then tasks may only reference an existing goal by id.
"""

from __future__ import annotations

from datetime import date

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from procbot.context import AppContext
from procbot.db.models import Category, Context, Milestone, Project
from procbot.domain.enums import Actor, EntityStatus
from procbot.services.audit import audit


def _norm(name: str) -> str:
    return " ".join(name.split()).strip()


# --------------------------------------------------------------------------- categories


async def get_or_create_category(session: AsyncSession, ctx: AppContext, name: str) -> Category:
    """Case-insensitive lookup by name; creates the category if missing."""
    clean = _norm(name)
    if not clean:
        raise ValueError("empty category name")
    # SQLite lower() is ASCII-only; compare case-insensitively in Python (tiny table).
    for row in (await session.execute(select(Category))).scalars():
        if row.name.casefold() == clean.casefold():
            return row
    row = Category(name=clean[:100], created_at=ctx.clock.now())
    session.add(row)
    await session.flush()
    return row


async def list_categories(session: AsyncSession) -> list[Category]:
    return list((await session.execute(select(Category).order_by(Category.name))).scalars())


# --------------------------------------------------------------------------- contexts


async def get_or_create_context(session: AsyncSession, ctx: AppContext, name: str) -> Context:
    clean = _norm(name)
    if not clean:
        raise ValueError("empty context name")
    for row in (await session.execute(select(Context))).scalars():
        if row.name.casefold() == clean.casefold():
            return row
    row = Context(name=clean[:100], created_at=ctx.clock.now())
    session.add(row)
    await session.flush()
    return row


async def list_contexts(session: AsyncSession) -> list[Context]:
    return list((await session.execute(select(Context).order_by(Context.name))).scalars())


# --------------------------------------------------------------------------- projects


async def create_project(
    session: AsyncSession,
    ctx: AppContext,
    *,
    title: str,
    actor: Actor,
    description: str | None = None,
    goal_id: int | None = None,
) -> Project:
    now = ctx.clock.now()
    clean = _norm(title)
    if not clean:
        raise ValueError("empty project title")
    row = Project(
        title=clean[:300],
        description=description,
        goal_id=goal_id,
        status=EntityStatus.ACTIVE,
        created_at=now,
        updated_at=now,
    )
    session.add(row)
    await session.flush()
    await audit(
        session,
        now=now,
        actor=actor,
        action="project.created",
        entity="project",
        entity_id=row.id,
        details={"title": row.title},
    )
    return row


async def get_project(session: AsyncSession, project_id: int) -> Project | None:
    return await session.get(Project, project_id)


async def list_projects(
    session: AsyncSession, *, status: EntityStatus | None = EntityStatus.ACTIVE
) -> list[Project]:
    stmt = select(Project).order_by(Project.title)
    if status is not None:
        stmt = stmt.where(Project.status == str(status))
    return list((await session.execute(stmt)).scalars())


async def set_project_status(
    session: AsyncSession, ctx: AppContext, project_id: int, status: EntityStatus, *, actor: Actor
) -> Project:
    row = await session.get(Project, project_id)
    if row is None:
        raise LookupError(f"project {project_id} not found")
    before = row.status
    row.status = str(status)
    row.updated_at = ctx.clock.now()
    await audit(
        session,
        now=row.updated_at,
        actor=actor,
        action="project.status",
        entity="project",
        entity_id=row.id,
        details={"before": before, "after": row.status},
    )
    return row


# --------------------------------------------------------------------------- milestones


async def create_milestone(
    session: AsyncSession,
    ctx: AppContext,
    *,
    title: str,
    actor: Actor,
    project_id: int | None = None,
    goal_id: int | None = None,
    due_date: date | None = None,
) -> Milestone:
    """A stage of a project. Key milestones of goals (FR-15) are stage C."""
    now = ctx.clock.now()
    clean = _norm(title)
    if not clean:
        raise ValueError("empty milestone title")
    if project_id is not None and await session.get(Project, project_id) is None:
        raise LookupError(f"project {project_id} not found")
    last = (
        await session.execute(
            select(func.max(Milestone.sort_order)).where(Milestone.project_id == project_id)
        )
    ).scalar_one_or_none()
    row = Milestone(
        title=clean[:300],
        project_id=project_id,
        goal_id=goal_id,
        due_date=due_date,
        status=EntityStatus.ACTIVE,
        sort_order=(last or 0) + 1,
        created_at=now,
        updated_at=now,
    )
    session.add(row)
    await session.flush()
    await audit(
        session,
        now=now,
        actor=actor,
        action="milestone.created",
        entity="milestone",
        entity_id=row.id,
        details={"title": row.title, "project_id": project_id},
    )
    return row


async def list_milestones(session: AsyncSession, *, project_id: int) -> list[Milestone]:
    stmt = (
        select(Milestone)
        .where(Milestone.project_id == project_id)
        .order_by(Milestone.sort_order, Milestone.id)
    )
    return list((await session.execute(stmt)).scalars())
