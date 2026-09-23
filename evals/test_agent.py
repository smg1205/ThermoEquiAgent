"""Behavioral evaluations at the public conversation orchestrator seam."""

from __future__ import annotations

import pytest

from agent.orchestrator import ConversationOrchestrator
from agent.router import recommend_models
from schemas.domain import ComponentIdentity, TaskManifest, ThermodynamicConditions


@pytest.mark.asyncio
async def test_knowledge_question_never_invokes_calculation() -> None:
    response = await ConversationOrchestrator().chat("NRTL和Peng-Robinson有什么区别？")
    assert response.intent == "MODEL_SELECTION_QA"
    assert response.task is None
    assert response.calculation is None
    assert response.statements[0].category == "Knowledge"


@pytest.mark.asyncio
async def test_calculation_question_generates_manifest_and_real_curve() -> None:
    response = await ConversationOrchestrator().chat("计算苯-甲苯在101.325 kPa下的T-x-y曲线")
    assert response.task is not None
    assert response.task.calculation_type == "isobaric_vle"
    assert response.calculation is not None
    assert len(response.calculation.result.points) == 21
    assert response.calculation.validation.overall_status in {"passed", "warning"}


@pytest.mark.asyncio
async def test_missing_pressure_is_explicit_and_has_no_fake_result() -> None:
    response = await ConversationOrchestrator().chat("计算苯-甲苯T-x-y曲线")
    assert response.task is not None
    assert response.calculation is None
    assert "pressure_kPa" in response.answer


@pytest.mark.asyncio
async def test_atmospheric_pressure_is_normalized_with_assumption() -> None:
    response = await ConversationOrchestrator().chat("计算苯-甲苯常压VLE")
    assert response.task is not None
    assert response.task.conditions.pressure_kPa == pytest.approx(101.325)
    assert any("101.325" in assumption for assumption in response.task.assumptions)


@pytest.mark.asyncio
async def test_component_grounding_does_not_match_benzene_inside_toluene() -> None:
    _, task = await ConversationOrchestrator().parse("计算甲苯在常压下的泡点")

    assert task is not None
    assert [component.cas_number for component in task.components] == ["108-88-3"]


@pytest.mark.asyncio
async def test_component_grounding_finds_explicit_ethane_after_methane() -> None:
    _, task = await ConversationOrchestrator().parse("Calculate methane, ethane and nitrogen VLE at 100 kPa")

    assert task is not None
    assert [component.cas_number for component in task.components] == [
        "74-82-8",
        "74-84-0",
        "7727-37-9",
    ]


@pytest.mark.asyncio
async def test_component_grounding_prefers_complete_multiword_identity() -> None:
    _, task = await ConversationOrchestrator().parse("Calculate carbon dioxide and methane VLE at 100 kPa")

    assert task is not None
    assert [component.cas_number for component in task.components] == [
        "124-38-9",
        "74-82-8",
    ]


@pytest.mark.asyncio
async def test_component_grounding_does_not_resolve_command_grammar_as_a_chemical() -> None:
    _, task = await ConversationOrchestrator().parse("Use PR for propane VLE at 100 kPa")

    assert task is not None
    assert [component.cas_number for component in task.components] == ["74-98-6"]


@pytest.mark.asyncio
async def test_component_grounding_requires_chemical_role_evidence_for_homonyms() -> None:
    _, task = await ConversationOrchestrator().parse("Calculate methane VLE at 100 kPa; changes can lead to a result.")

    assert task is not None
    assert [component.cas_number for component in task.components] == ["74-82-8"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "message",
    [
        "Calculate methane VLE at 100 kPa with lead time recorded.",
        "Calculate methane VLE at 100 kPa with water usage recorded.",
        "Calculate methane and lead time effects for VLE at 100 kPa.",
        "Calculate methane and water usage effects for VLE at 100 kPa.",
        "Calculate methane VLE at 100 kPa and compute lead time.",
        "Calculate methane VLE at 100 kPa and compute iron losses.",
    ],
)
async def test_component_grounding_applies_role_evidence_to_all_identities(message: str) -> None:
    _, task = await ConversationOrchestrator().parse(message)

    assert task is not None
    assert [component.cas_number for component in task.components] == ["74-82-8"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "message",
    [
        "Calculate the VLE for methane and ethane.",
        "Calculate a TP Flash for methane and ethane.",
        "Calculate VLE for benzene and toluene.",
    ],
)
async def test_component_grounding_applies_scoped_role_to_a_complete_list(message: str) -> None:
    _, task = await ConversationOrchestrator().parse(message)

    assert task is not None
    assert len(task.components) == 2


@pytest.mark.asyncio
async def test_followup_pressure_change_inherits_system_and_creates_new_run() -> None:
    orchestrator = ConversationOrchestrator()
    first = await orchestrator.chat("计算苯-甲苯常压VLE")
    second = await orchestrator.chat("压力改为80 kPa，再算一次", first.conversation_id)
    assert first.task is not None and second.task is not None
    assert [c.component_id for c in second.task.components] == ["benzene", "toluene"]
    assert second.task.conditions.pressure_kPa == pytest.approx(80.0)
    assert second.task.task_id != first.task.task_id
    assert second.calculation is not None
    assert second.calculation.result.run_id != first.calculation.result.run_id  # type: ignore[union-attr]


def test_lle_hard_excludes_wilson() -> None:
    task = TaskManifest(
        equilibrium_type="LLE",
        calculation_type="lle",
        components=[
            ComponentIdentity(component_id="benzene", name="Benzene"),
            ComponentIdentity(component_id="toluene", name="Toluene"),
        ],
        conditions=ThermodynamicConditions(temperature_K=298.15),
    )
    wilson = next(item for item in recommend_models(task, {"Wilson"}) if item.model_name == "Wilson")
    assert not wilson.executable
    assert any("hard-excluded" in reason for reason in wilson.exclusions)


@pytest.mark.asyncio
async def test_electrolyte_task_is_refused_without_solver() -> None:
    response = await ConversationOrchestrator().chat("计算氯化钠水溶液的电解质相平衡")
    assert response.intent == "UNSUPPORTED_TASK"
    assert response.calculation is None
    assert response.statements[0].category == "Warning"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "message",
    [
        "Calculate salt VLE at 100 kPa",
        "Calculate potassium chloride VLE at 100 kPa",
        "Calculate KCl VLE at 100 kPa",
        "Calculate sodium acetate VLE at 100 kPa",
        "Calculate ammonium bicarbonate VLE at 100 kPa",
        "Calculate hydrochloric acid VLE at 100 kPa",
        "Calculate copper sulfate VLE at 100 kPa",
        "Calculate tetramethylammonium chloride VLE at 100 kPa",
        "Calculate choline chloride VLE at 100 kPa",
        "Calculate phosphonium chloride VLE at 100 kPa",
        "Calculate tetrabutylammonium bromide VLE at 100 kPa",
        "Calculate brine VLE at 100 kPa",
        "Calculate saltwater VLE at 100 kPa",
    ],
)
async def test_generic_salt_task_is_refused_as_electrolyte_scope(message: str) -> None:
    response = await ConversationOrchestrator().chat(message)

    assert response.intent == "UNSUPPORTED_TASK"
    assert response.task is None
    assert response.calculation is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "message",
    [
        "Calculate ferrocene VLE at 100 kPa",
        "Calculate tetraethyllead VLE at 100 kPa",
        "Calculate nickel tetracarbonyl VLE at 100 kPa",
    ],
)
async def test_neutral_organometallic_is_not_misclassified_as_electrolyte(message: str) -> None:
    response = await ConversationOrchestrator().chat(message)

    assert response.intent != "UNSUPPORTED_TASK"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("message", "expected_cas"),
    [
        (
            "Calculate benzene/toluene VLE without salt at 100 kPa.",
            ["71-43-2", "108-88-3"],
        ),
        (
            "Calculate salt-free acetone/water VLE at 100 kPa.",
            ["67-64-1", "7732-18-5"],
        ),
        (
            "Calculate a non-ionic methane/ethane VLE at 100 kPa.",
            ["74-82-8", "74-84-0"],
        ),
        (
            "Calculate benzene/toluene VLE without potassium chloride at 100 kPa.",
            ["71-43-2", "108-88-3"],
        ),
        (
            "Calculate benzene/toluene VLE without 7447-40-7 at 100 kPa.",
            ["71-43-2", "108-88-3"],
        ),
        (
            "Calculate brine-free benzene/toluene VLE at 100 kPa.",
            ["71-43-2", "108-88-3"],
        ),
        (
            "Calculate saltwater-free benzene/toluene VLE at 100 kPa.",
            ["71-43-2", "108-88-3"],
        ),
    ],
)
async def test_negated_excluded_scope_term_does_not_reject_supported_task(
    message: str,
    expected_cas: list[str],
) -> None:
    response = await ConversationOrchestrator().chat(message)

    assert response.intent != "UNSUPPORTED_TASK"
    assert response.task is not None
    assert [component.cas_number for component in response.task.components] == expected_cas


@pytest.mark.asyncio
@pytest.mark.parametrize("generic_name", ["alcohol", "spirit", "ether"])
async def test_generic_chemical_class_is_not_bound_to_an_arbitrary_compound(generic_name: str) -> None:
    response = await ConversationOrchestrator().chat(f"Calculate {generic_name} VLE at 100 kPa")

    assert response.task is None
    assert response.calculation is None


@pytest.mark.asyncio
async def test_parameter_request_does_not_invent_values() -> None:
    response = await ConversationOrchestrator().chat("给我一个本地没有的NRTL参数")
    assert response.intent == "PARAMETER_QUERY"
    assert response.calculation is None
    assert all(not any(char.isdigit() for char in item.text) for item in response.statements)


@pytest.mark.asyncio
async def test_sensitivity_intent_is_recognized_without_unapproved_scan() -> None:
    response = await ConversationOrchestrator().chat("做一下压力敏感性分析")
    assert response.intent == "SENSITIVITY_ANALYSIS"
    assert response.calculation is None
    assert response.statements[0].category == "Warning"


@pytest.mark.asyncio
async def test_lle_request_reaches_lle_contract_without_fake_result() -> None:
    response = await ConversationOrchestrator().chat("计算苯和甲苯在 298.15 K 下的液液平衡 LLE")
    assert response.intent == "EQUILIBRIUM_CALCULATION"
    assert response.task is not None
    assert response.task.equilibrium_type == "LLE"
    assert response.task.calculation_type == "lle"
    assert response.calculation is None
    assert response.statements[0].category == "Warning"
    assert "cannot represent liquid-liquid" in response.answer


@pytest.mark.asyncio
async def test_polymer_request_is_refused_at_scope_boundary() -> None:
    response = await ConversationOrchestrator().chat("计算聚合物溶液的汽液平衡")
    assert response.intent == "UNSUPPORTED_TASK"
    assert response.calculation is None
