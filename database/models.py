"""SQLAlchemy tables for traceable conversations and calculations."""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import JSON, DateTime, Float, ForeignKey, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def utcnow() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    pass


class ConversationRow(Base):
    __tablename__ = "conversations"
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class MessageRow(Base):
    __tablename__ = "messages"
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    conversation_id: Mapped[str] = mapped_column(ForeignKey("conversations.id"), index=True)
    role: Mapped[str] = mapped_column(String(16))
    content: Mapped[str] = mapped_column(Text)
    structured: Mapped[dict[str, object] | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class TaskRow(Base):
    __tablename__ = "tasks"
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    conversation_id: Mapped[str | None] = mapped_column(ForeignKey("conversations.id"), nullable=True)
    manifest: Mapped[dict[str, object]] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class ComponentRow(Base):
    __tablename__ = "components"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(128))
    cas_number: Mapped[str | None] = mapped_column(String(32), nullable=True, unique=True)
    identity_data: Mapped[dict[str, object]] = mapped_column(JSON)


class ModelCardRow(Base):
    __tablename__ = "model_cards"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    card: Mapped[dict[str, object]] = mapped_column(JSON)
    version: Mapped[str] = mapped_column(String(32), default="0.1.0")


class ParameterSetRow(Base):
    __tablename__ = "parameter_sets"
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    model_name: Mapped[str] = mapped_column(String(64), index=True)
    component_key: Mapped[str] = mapped_column(String(256), index=True)
    payload: Mapped[dict[str, object]] = mapped_column(JSON)
    source_type: Mapped[str] = mapped_column(String(32))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class CalculationRunRow(Base):
    __tablename__ = "calculation_runs"
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    request_id: Mapped[str] = mapped_column(String(64), index=True)
    task_id: Mapped[str] = mapped_column(String(36), index=True)
    status: Mapped[str] = mapped_column(String(16))
    input_snapshot: Mapped[dict[str, object]] = mapped_column(JSON)
    result_snapshot: Mapped[dict[str, object]] = mapped_column(JSON)
    validation_snapshot: Mapped[dict[str, object]] = mapped_column(JSON)
    software_version: Mapped[str] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class CalculationPointRow(Base):
    __tablename__ = "calculation_points"
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("calculation_runs.id"), index=True)
    ordinal: Mapped[int]
    temperature_K: Mapped[float] = mapped_column(Float)
    pressure_kPa: Mapped[float] = mapped_column(Float)
    liquid_composition: Mapped[list[float]] = mapped_column(JSON)
    vapor_composition: Mapped[list[float]] = mapped_column(JSON)


class ValidationReportRow(Base):
    __tablename__ = "validation_reports"
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("calculation_runs.id"), unique=True)
    overall_status: Mapped[str] = mapped_column(String(16))
    report: Mapped[dict[str, object]] = mapped_column(JSON)


class EvidenceRecordRow(Base):
    __tablename__ = "evidence_records"
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    run_id: Mapped[str | None] = mapped_column(ForeignKey("calculation_runs.id"), nullable=True)
    category: Mapped[str] = mapped_column(String(32))
    source_identifier: Mapped[str | None] = mapped_column(String(512), nullable=True)
    payload: Mapped[dict[str, object]] = mapped_column(JSON)


class ExportRecordRow(Base):
    __tablename__ = "export_records"
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("calculation_runs.id"), index=True)
    format: Mapped[str] = mapped_column(String(16))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
