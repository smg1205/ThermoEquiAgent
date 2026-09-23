export type Intent =
  | "CONCEPT_QA"
  | "MODEL_SELECTION_QA"
  | "PARAMETER_QUERY"
  | "DATA_QUERY"
  | "EQUILIBRIUM_CALCULATION"
  | "RESULT_INTERPRETATION"
  | "SENSITIVITY_ANALYSIS"
  | "PROCESS_RECOMMENDATION"
  | "FLOW_DESIGN_QA"
  | "EXTRACTIVE_DISTILLATION"
  | "LLE_EXTRACTION"
  | "DISTILLATION_DESIGN"
  | "TASK_CORRECTION"
  | "UNSUPPORTED_TASK";

export type CalculationType =
  | "bubble_point"
  | "dew_point"
  | "isobaric_vle"
  | "isothermal_vle"
  | "tp_flash"
  | "phase_stability"
  | "azeotrope"
  | "lle"
  | "infinite_dilution_activity";

export interface ComponentIdentity {
  component_id: string;
  name: string;
  cas_number?: string | null;
  smiles?: string | null;
  aliases: string[];
}

export interface Conditions {
  temperature_K?: number | null;
  pressure_kPa?: number | null;
  liquid_composition?: number[] | null;
  vapor_composition?: number[] | null;
  feed_composition?: number[] | null;
  temperature_span_K?: [number, number] | null;
}

export interface ParameterSet {
  parameter_set_id: string;
  model_name: string;
  component_order: string[];
  parameters: Record<string, number>;
  parameter_form: string;
  units: Record<string, string>;
  temperature_range_K?: [number, number] | null;
  pressure_range_kPa?: [number, number] | null;
  equilibrium_types: Array<"VLE" | "LLE" | "FLASH">;
  source_title?: string | null;
  source_identifier?: string | null;
  source_type: "literature" | "database" | "user_supplied" | "test_fixture" | "estimated" | "unknown";
  quality_level: string;
  notes?: string | null;
}

export interface TaskManifest {
  task_id: string;
  equilibrium_type: "VLE" | "LLE" | "FLASH";
  calculation_type: CalculationType;
  components: ComponentIdentity[];
  conditions: Conditions;
  composition_basis: "mole_fraction" | "mass_fraction";
  requested_outputs: string[];
  validation_requirements: string[];
  assumptions: string[];
  model_name?: string | null;
  points: number;
  original_question?: string | null;
  parameters: ParameterSet[];
}

export interface EquilibriumPoint {
  temperature_K: number;
  pressure_kPa: number;
  liquid_composition: number[];
  vapor_composition: number[];
  equilibrium_residual: number;
}

export interface ModelSeries {
  model_name: string;
  points: EquilibriumPoint[];
}

export interface GammaInfinityPoint {
  temperature_K: number;
  solute_index: number;
  solvent_index: number;
  gamma_infinity: number;
  ln_gamma_infinity: number;
}

export interface CheckResult {
  passed: boolean;
  metric?: number | null;
  tolerance?: number | null;
  message: string;
}

export interface ValidationReport {
  overall_status: "passed" | "warning" | "failed";
  composition_balance: CheckResult;
  material_balance: CheckResult;
  equilibrium_residual: CheckResult;
  convergence: CheckResult;
  parameter_applicability: CheckResult;
  phase_stability?: CheckResult | null;
  warnings: string[];
  recommended_action?: string | null;
  maximum_equilibrium_residual: number;
  mean_equilibrium_residual: number;
  solver_converged: boolean;
}

export interface ModelRecommendation {
  model_name: string;
  score: number;
  executable: boolean;
  reasons: string[];
  exclusions: string[];
  breakdown: ScoreBreakdown;
}

export interface ScoreBreakdown {
  phase_support_score: number;
  system_match_score: number;
  condition_match_score: number;
  parameter_availability_score: number;
  evidence_quality_score: number;
  extrapolation_penalty: number;
  numerical_risk_penalty: number;
}

export interface ModelCard {
  model_name: string;
  family: string;
  supported_tasks: string[];
  excluded_systems: string[];
  requires_binary_parameters: boolean;
  pressure_regime: string[];
  validation_requirements: string[];
  implementation_status: "available" | "contract_only" | "planned";
  production_ready: boolean;
}

export interface PhaseResult {
  phase: "liquid" | "vapor";
  fraction: number;
  composition: number[];
}

export interface CalculationEnvelope {
  result: {
    run_id: string;
    task_id: string;
    calculation_type: CalculationType;
    input_snapshot: Record<string, unknown>;
    model_name: string;
    parameter_set_id?: string | null;
    points: EquilibriumPoint[];
    gamma_infinity: GammaInfinityPoint[];
    phases: PhaseResult[];
    temperature_K?: number | null;
    pressure_kPa?: number | null;
    vapor_fraction?: number | null;
    phase_state: string;
    converged: boolean;
    residual: number;
    iterations: number;
    warnings: string[];
    backend_version: string;
    solver_name: string;
    failure?: Record<string, unknown> | null;
    created_at: string;
  };
  validation: ValidationReport;
  parameter_sources: Array<Record<string, string>>;
  model_recommendations: ModelRecommendation[];
}

export interface FailureDetail {
  failure_type: string;
  message: string;
  recovery_action: string;
  details: Record<string, unknown>;
}

export type PhaseDiagramStatus = "passed" | "warning" | "failed" | "unsupported";

export type PhaseDiagramType = "TXY" | "PXY";

export interface PhaseDiagramEntry {
  model_name: string;
  status: PhaseDiagramStatus;
  executable: boolean;
  score?: number | null;
  result: CalculationEnvelope["result"] | null;
  validation?: ValidationReport | null;
  failure: FailureDetail | null;
  parameter_sources?: Array<Record<string, string>>;
  warnings: string[];
}

export interface PhaseDiagramResponse {
  task: TaskManifest;
  diagram_type: PhaseDiagramType;
  entries: PhaseDiagramEntry[];
  total_models: number;
  passed_count: number;
  warning_count: number;
  failed_count: number;
  unsupported_count: number;
  summary: string;
}

export interface EvidenceStatement {
  category: "Knowledge" | "Database" | "Calculation" | "Inference" | "Estimate" | "Warning";
  text: string;
}

export interface AgentStep {
  phase: "plan" | "execute" | "validate" | "respond";
  status: "completed" | "failed" | "blocked";
  summary: string;
  tool_name?: string | null;
}

// —— Process flow design (Skill / FlowDesignDraft) ——
// See schemas/domain.py FlowDesignDraft for the Python source of truth.

export type FlowParameterSource = "LLM_SUGGESTED" | "RULE_SUGGESTED" | "VALIDATED";

export type UnitOperationType =
  | "preheater"
  | "distillation_column"
  | "flash_drum"
  | "condenser"
  | "reboiler"
  | "heat_exchanger"
  | "mixer"
  | "splitter";

export interface FlowParameter {
  value: number | string;
  source: FlowParameterSource;
  needs_validation?: boolean;
  note?: string | null;
}

export interface FlowFeed {
  components: string[];
  composition_mole?: number[];
  temperature_K?: number | null;
  pressure_kPa?: number | null;
  flow_rate_mol_s?: number | null;
  assumption?: string | null;
}

export interface FlowUnitOperation {
  id: string;
  type: UnitOperationType;
  name: string;
  input_stream?: string | null;
  output_streams?: Record<string, string>;
  conditions?: Record<string, FlowParameter>;
}

export interface FlowProductSpec {
  stream: string;
  spec: string;
}

export interface FlowDesignDraft {
  flow_name: string;
  flow_type?: string;
  feed: FlowFeed;
  unit_operations: FlowUnitOperation[];
  streams_connectivity_note?: string | null;
  thermodynamic_model?: string | null;
  model_recommendation_note?: string | null;
  product_specs?: FlowProductSpec[];
  assumptions?: string[];
  notes?: string[];
}

export interface EntrainerOption {
  name: string;
  canon_name: string;
  selectivity: number;
  relative_volatility: number;
  recommended: boolean;
  note?: string | null;
}

export interface ExtractiveExportPayload {
  status: "ready" | "awaiting_entrainer" | "missing_parameters" | "dwsim_unavailable" | "failed";
  /** Local-model-ranked entrainer options for the user to pick. Populated when status === "awaiting_entrainer". */
  entrainer_candidates?: EntrainerOption[] | null;
  /** Effective activity source used for scoring/design (unifac | thermoformer). */
  alpha_source?: string | null;
  /** Relative URI to download the generated .dwxmz file (null when not ready). */
  dwsim_file_uri?: string | null;
  /** Stable id identifying the stored file for the download endpoint. */
  file_id?: string | null;
  missing_parameters: string[];
  message: string;
  design?: {
    theoretical_stages: number;
    reflux_ratio: number;
    minimum_reflux_ratio: number;
    feed_stage: number;
    entrainer_stage: number;
    condenser_temperature_K: number;
    reboiler_temperature_K: number;
    operating_pressure_kPa: number;
  } | null;
}

export interface BinaryDistillationResult {
  components: string[];
  feed_composition: number[];
  feed_flow_mol_s: number;
  operating_pressure_kPa: number;
  distillate_purity_mole_fraction: number;
  relative_volatility: number;
  minimum_stages: number;
  minimum_reflux_ratio: number;
  reflux_ratio: number;
  theoretical_stages: number;
  feed_stage: number;
  condenser_temperature_K: number;
  reboiler_temperature_K: number;
  distillate_flow_mol_s: number;
  bottoms_flow_mol_s: number;
  alpha_source: string;
  assumptions: string[];
}

export interface BinaryDistillationPayload {
  status: "ready" | "missing_parameters" | "dwsim_unavailable" | "failed";
  result?: BinaryDistillationResult | null;
  missing_parameters: string[];
  message: string;
  alpha_source?: string | null;
  /** Bubble temperature of the saturated-liquid feed, K (DWSIM feed-stream T). */
  feed_temperature_K?: number | null;
  /** Relative URI to download the generated .dwxmz file (null when not ready). */
  dwsim_file_uri?: string | null;
  /** Stable id identifying the stored file for the download endpoint. */
  file_id?: string | null;
}

export interface LLEExtractionSpec {
  feed_components: string[];
  feed_composition: number[];
  solvent: string;
  solvent_ratio: number;
  feed_flow_mol_s: number;
  feed_temperature_K: number;
  feed_pressure_kPa: number;
  property_package: string;
}

/** Flowsheet topology a binary case can be rendered as. */
export type BinaryFlowsheetMode = "tp_flash" | "bubble_point" | "two_liquid_vessel" | "extractor";

/**
 * Fully-resolved inputs of a *binary* phase-equilibrium export.
 *
 * A binary partially miscible pair (e.g. water / 1-butanol) has no third
 * extraction solvent, so it is described here rather than by
 * {@link LLEExtractionSpec}.  Mirrors `BinaryCaseSpec` on the backend.
 */
export interface BinaryCaseSpec {
  components: string[];
  feed_composition: number[];
  kind: "vle" | "lle" | "vlle";
  mode: BinaryFlowsheetMode;
  solvent?: string | null;
  solvent_ratio: number;
  feed_flow_mol_s: number;
  temperature_K?: number | null;
  pressure_kPa: number;
  property_package: string;
}

/**
 * Inputs for a generic three-component liquid-liquid export.
 *
 * Three feed components on an equal footing -- no separate solvent stream --
 * rendered as the `Vessel_LLE` template with three compounds.  Mirrors
 * `TernaryLLESpec` on the backend.
 */
export interface TernaryLLESpec {
  components: string[];
  feed_composition: number[];
  temperature_K: number;
  pressure_kPa: number;
  feed_flow_mol_s: number;
  property_package: string;
}

export interface LLEExportPayload {
  status: "ready" | "missing_parameters" | "dwsim_unavailable" | "failed";
  /** Structural echo of the extraction inputs (used to build the DWSIM file). */
  spec?: LLEExtractionSpec | null;
  /**
   * Resolved inputs when `lle_kind === "binary"`; the ternary path uses `spec`
   * instead.  A binary partially miscible pair has no third solvent.
   */
  binary_spec?: BinaryCaseSpec | null;
  /**
   * Resolved inputs when the request was a generic *three-component* LLE case
   * (three named components, no separate solvent stream).
   */
  ternary_spec?: TernaryLLESpec | null;
  /** `"binary"` for a two-component partially miscible pair (the report/dwsim
   *  `Vessel_LLE` template), `"ternary"` for a solvent extraction. */
  lle_kind?: "binary" | "ternary";
  /** Relative URI to download the generated .dwxmz file (null when not ready). */
  dwsim_file_uri?: string | null;
  file_id?: string | null;
  missing_parameters: string[];
  message: string;
  warnings: string[];
}

export interface ChatResponse {
  conversation_id: string;
  intent: Intent;
  answer: string;
  statements: EvidenceStatement[];
  execution_steps: AgentStep[];
  task?: TaskManifest | null;
  calculation?: CalculationEnvelope | null;
  flow_design?: FlowDesignDraft | null;
  extractive?: ExtractiveExportPayload | null;
  lle_extraction?: LLEExportPayload | null;
  distillation?: BinaryDistillationPayload | null;
  request_id?: string | null;
}

export type RunStatus = "passed" | "warning" | "failed";

export interface RunSummary {
  run_id: string;
  request_id: string;
  task_id: string;
  status: RunStatus;
  calculation_type: CalculationType;
  model_name: string;
  backend_version: string;
  created_at: string;
}

export interface RunListResponse {
  items: RunSummary[];
  total: number;
  limit: number;
  offset: number;
}
