# Code and documentation conventions

- Active source paths, identifiers, comments, docstrings, log messages, and
  public documentation use English.
- Original dataset field names may appear only where required to parse a frozen
  source workbook or preserve archived data provenance.
- Scientific modules expose compact interfaces; repository scripts contain only
  orchestration.
- A cohesive deep module may own tightly coupled implementations (for example,
  feature caching, split-train scaling, and view fusion); method-specific files
  may provide stable import surfaces without duplicating that implementation.
- Formal, diagnostic, exploratory, archived, deprecated, and not-evaluated are
  used as explicit evidence-status terms.
- Published metrics are generated from machine-readable artifacts rather than
  transcribed into code.
- Historical artifacts are not silently rewritten when doing so would invalidate
  their recorded hashes.
