"""Declarative base and column types shared by all tables."""

from __future__ import annotations

from datetime import UTC, date, datetime, time
from typing import Any

from sqlalchemy import JSON, DateTime, MetaData, String, TypeDecorator
from sqlalchemy.engine import Dialect
from sqlalchemy.orm import DeclarativeBase

NAMING_CONVENTION = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


class UTCDateTime(TypeDecorator[datetime]):
    """Stores aware datetimes as naive UTC; returns aware UTC datetimes.

    Naive datetimes are rejected so local times never leak into storage.
    """

    impl = DateTime
    cache_ok = True

    def process_bind_param(self, value: datetime | None, dialect: Dialect) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            raise ValueError("naive datetime cannot be stored; use aware UTC datetimes")
        return value.astimezone(UTC).replace(tzinfo=None)

    def process_result_value(self, value: datetime | None, dialect: Dialect) -> datetime | None:
        if value is None:
            return None
        return value.replace(tzinfo=UTC)


class LocalTime(TypeDecorator[time]):
    """Wall-clock time (HH:MM) interpreted in the owner's IANA timezone."""

    impl = String(5)
    cache_ok = True

    def process_bind_param(self, value: time | None, dialect: Dialect) -> str | None:
        return None if value is None else value.strftime("%H:%M")

    def process_result_value(self, value: str | None, dialect: Dialect) -> time | None:
        return None if value is None else time.fromisoformat(value)


class LocalDate(TypeDecorator[date]):
    impl = String(10)
    cache_ok = True

    def process_bind_param(self, value: date | None, dialect: Dialect) -> str | None:
        return None if value is None else value.isoformat()

    def process_result_value(self, value: str | None, dialect: Dialect) -> date | None:
        return None if value is None else date.fromisoformat(value)


JSONType = JSON().with_variant(JSON(), "sqlite")


class Base(DeclarativeBase):
    metadata = MetaData(naming_convention=NAMING_CONVENTION)
    type_annotation_map: dict[Any, Any] = {
        datetime: UTCDateTime(),
        time: LocalTime(),
        date: LocalDate(),
        dict[str, Any]: JSON(),
        list[Any]: JSON(),
    }
