# legal_india: building blocks for Indian-law citation checks (milestone M1)

Pure Python: no network, no LLM, no dependency on the RAG pipelines. Plan: `docs/legal_india_plan.md`.

| Module | What it does |
|---|---|
| `citations.py` | Parses AIR, SCC (incl. `(Cri)`, `Supp (n)`), SCC OnLine, SCR, SCALE, INSC neutral citations and "A v. B" case names; pinpoints (`para 45`, `paras 12-14`, `at p. 300`); groups parallel citations; canonical keys such as `AIR:1973:SC:1461`. |
| `statutes.py` | Extracts "Section 302 of the IPC", "S. 482 CrPC", "Articles 14, 19 and 21", "Order VII Rule 11 CPC" into `(act, provision)` with act aliases (IPC, BNS, CrPC, BNSS, Evidence Act, BSA, CPC, Constitution). |
| `transition.py` | `applicable_code(offence_date)` (IPC on or before 30 June 2024, BNS from 1 July 2024), a checker for a section cited under the wrong code, and a mapper over `data/section_map.json`. |
| `hierarchy.py` | Art. 141 (Supreme Court binds all courts) and High Court binding within its territory; everything else persuasive. |

## Limitations (read before relying on any output)

- **`data/section_map.json` is partial and UNVERIFIED.** It holds a small set of well-known IPC to BNS,
  CrPC to BNSS and Evidence Act to BSA correspondences, every entry marked `verified: false`. Sections were
  split, merged and renumbered, so the mapping is not always one-to-one. Verify against the official Gazette /
  India Code text before any research use. Evidence Act ss. 24-27 (confessions) were deliberately left out because
  the new Act merges them.
- **Date rule:** only the substantive code (IPC vs BNS) is decided by the offence date. Procedure (CrPC/BNSS)
  and evidence (Evidence Act/BSA) depend on the stage of proceedings and the savings clauses, so they are not
  decided. Offences spanning the cut-off date are not modelled.
- **Citations:** regex-based. Unusual formats (e.g. High Court neutral citations other than INSC, SCC series
  not listed, Manupatra or other proprietary formats) are not recognised. Party-name detection is heuristic
  and can mis-bound names that run into a following capitalised word; it does not resolve a name to a case.
  A bare "at 45" is not treated as a pinpoint.
- **Statutes:** only the acts listed above are normalised; others are returned as `act="OTHER"` with the name
  as written. Chains like "302 IPC and 304 BNS" need a label on each item. Ranges ("Sections 3 to 5") are not
  expanded. "Article ..." with no named act defaults to the Constitution.
- **Hierarchy:** does not model bench strength, ratio vs obiter, tribunals, overruling or High Court benches
  outside the principal seat. Territory lists are not exhaustive; unknown High Courts give `undetermined`.
- Nothing here checks whether a case is still good law or whether a cited proposition is supported by the case.
