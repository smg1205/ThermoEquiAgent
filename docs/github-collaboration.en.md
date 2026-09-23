# GitHub team collaboration setup

This note describes how to share the current local project with the team safely. No remote is
configured in the repository, and nothing is uploaded automatically.

## 1. Creating the remote repository for the first time

Create an empty repository under the GitHub organisation, for example `ThermoEqui-Agent`. Setting it
to Private first is recommended, and do **not** tick the auto-generated README, `.gitignore` or
Licence boxes, to avoid conflicting with local files.

The repository administrator runs locally:

```powershell
git branch -M main
git remote add origin https://github.com/<ORG>/ThermoEqui-Agent.git
git push -u origin main
```

If using SSH, replace the remote with:

```text
git@github.com:<ORG>/ThermoEqui-Agent.git
```

## 2. GitHub repository settings

Under `Settings -> Collaborators and teams`, add members or teams by responsibility:

- `maintainers`: repository settings, releases and final merges;
- `thermodynamics`: model applicability, parameter evidence and validation review;
- `backend`: agent, API, database and the deterministic core;
- `frontend`: the Next.js workbench and API contracts.

Under `Settings -> Branches`, create a ruleset for `main`:

- Require a pull request before merging;
- At least 1 approval; 2 are recommended for scientific model and parameter changes, one of them from
  a thermodynamics reviewer;
- Require status checks: `backend`, `frontend`, `external-engines`;
- Require conversation resolution;
- Block force pushes and branch deletion;
- Allow bypass for maintainers only.

Once the project stabilises, `.github/CODEOWNERS` can be added. Do not enable incorrect automatic
assignment before the team names are settled. Example:

```text
/thermo_engine/                 @ORG/thermodynamics @ORG/backend
/knowledge/model_cards/         @ORG/thermodynamics
/knowledge/model_selection_rules/ @ORG/thermodynamics
/agent/                         @ORG/backend
/apps/api/                      @ORG/backend
/apps/web/                      @ORG/frontend
/schemas/                       @ORG/backend @ORG/frontend
```

## 3. Day-to-day collaboration

Open an Issue for each piece of work first, then branch from the latest `main`:

```powershell
git switch main
git pull --ff-only
git switch -c feat/123-nrtl-vle
```

When finished, run the checks, commit and push:

```powershell
git add <explicit files>
git commit -m "feat(engine): add NRTL VLE adapter"
git push -u origin feat/123-nrtl-vle
```

Then open a Pull Request, link the Issue, and have another member review it. Do not use
`git push --force` on shared branches. Detailed commit and test requirements are in
[CONTRIBUTING.md](../CONTRIBUTING.md) at the repository root.

## 4. Recommended task breakdown

Do not put "integrate every phase-equilibrium model" into a single PR. A better breakdown is:

1. Parameter data contract and evidence fields;
2. Backend adapter for a single model;
3. Model applicability and exclusion rules;
4. Physical validation and benchmark cases;
5. API contracts;
6. Frontend selection, state and reporting;
7. Documentation and examples.

Create at least one Epic / parent Issue per model, then split sub-tasks along
"parameters - calculation - validation - interface". This lets thermodynamics staff review scientific
correctness while software staff work on interfaces and UI in parallel.

## 5. Secrets and data safety

- Every member uses their own `.env`; only `.env.example` is committed.
- DeepSeek/OpenAI keys are injected only through environment variables or GitHub Actions Secrets.
- GitHub Actions may use the variable name `DEEPSEEK_API_KEY`, but default CI should not call paid
  external APIs.
- The run database `thermoequi.db`, logs, export results and test caches are never committed.
- If a key has appeared in a screenshot, terminal share or chat log, rotate it immediately in the
  vendor console.
- Enable GitHub Secret Scanning and Push Protection.

## 6. Release recommendations

Use tags of the form `v0.1.0`. Before a release, record:

- supported models, backends and calculation types;
- parameter store version and provenance;
- validation benchmarks and errors;
- known limitations;
- database migration requirements;
- frontend/backend images or installation method.

If the repository becomes public in future, the team should first choose and add an explicit open
source licence; until then, do not assume third parties may copy or redistribute the code.
