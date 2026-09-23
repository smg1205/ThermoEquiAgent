# Scientific scope and article alignment

The repository follows the experiment structure and terminology in the supplied ThermoFormer article source. The article files are reference documents and are not modified by repository commands.

| Topic | Repository evidence | Scientific interpretation |
|---|---|---|
| VLE observations | `datasets/vle/` | 27,304 experimental points: 22,760 binary observations from 693 systems and 4,544 ternary observations from 107 systems. |
| LLE observations | `datasets/lle/` | 19,844 experimental points: 5,983 binary observations from 394 systems and 13,861 ternary observations from 986 systems. |
| VLE modeling rows | `experiments/data_quality/reports/data_audit.json` | 11,014 rows remain after the registered quality, pressure, SMILES, duplicate-state, and pure-component-reference rules. Observation counts and modeling-row counts describe different stages of the data contract. |
| LLE condition groups | Phase-equilibrium manifests | Registered filtering yields 2,794 binary and 1,584 ternary mixture--temperature--pressure conditions. Conditions and workbook observations are different counting units. |
| VLE predictive table | `experiments/summary/VLE_results_table.csv` | All printed mean and sample-standard-deviation values agree with the article table. |
| LLE predictive table | `experiments/summary/LLE_results_table.csv` | All printed mean and sample-standard-deviation values agree with the article table. |
| LLE pressure extrapolation | Split manifests | Fourteen binary systems contain at least three distinct pressures; 169 contain at least three distinct temperatures. The pressure protocol therefore covers fewer systems. |
| Mixed-model evaluation | Checkpoint registry and result manifests | A mixed-training checkpoint is evaluated on binary, ternary, and joint subsets; the subsets do not represent independent training runs. |
| LLE thermodynamics | LLE configuration and solver modules | The model uses binary Redlich--Kister and continuous parallel ternary tie-line parameterizations, a two-phase prior, and finite-grid TPD certification. |
| Interpretability | `experiments/vle/interpretability/` | Attributions quantify model response at evaluated VLE states and do not establish causal molecular mechanisms. |
| Separation design | `configs/separation_design/autonomous/thermoequi_agent.json` | An auditable entrainer-screening framework is provided. The ThermoEqui-Agent orchestration, Fenske--Underwood--Gilliland workflow, `.dwxmz` export, and article case artifacts are unavailable in this repository snapshot. |

Quantitative comparisons require matching dataset, split, target, and metric identities. Per-seed records remain the source for independently recomputing five-seed means and sample standard deviations.
