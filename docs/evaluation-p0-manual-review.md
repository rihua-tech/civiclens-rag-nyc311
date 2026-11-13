# P0 Manual Evidence Review

Review date: 2026-11-12

Method: direct human-style inspection of 30 deterministic offline outputs, returned provenance, expected behavior, and checked-in source content. No LLM judge was used. `Supported` means the core answer was entailed by its source; `partial/nonresponsive` means citation IDs were valid but the wording did not fully answer the question; `safe miss` means an answerable case abstained without disclosure; `appropriate abstention` means an unsupported case abstained without sources or citations.

| Dataset / ID | Disposition | Review |
|---|---|---|
| regression / q001 | partial/nonresponsive | Returned the curated source inventory rather than explaining the architecture. |
| regression / q002 | partial/nonresponsive | Identified project-document categories but did not clearly identify the architecture source requested. |
| regression / q003 | safe miss | Answerable architecture question; relevant evidence was not retrieved. |
| regression / q004 | safe miss | Answerable retrieval-scope question; only a troubleshooting chunk was retrieved. |
| regression / q005 | supported | No-answer rule is directly stated in the cited section. |
| regression / q006 | partial/nonresponsive | Cited the correct local-flow section but omitted most commands and steps. |
| regression / q007 | supported | Correctly identifies the NYC 311 field guide as the field-definition source. |
| regression / q008 | partial/nonresponsive | Cited real field definitions but omitted important requested status/agency details. |
| regression / q009 | safe miss | `closed_date` is documented, but sufficient evidence was not retrieved. |
| regression / q010 | safe miss | `complaint_type` is documented, but sufficient evidence was not retrieved. |
| regression / q011 | supported | Values match the cited checked-in top-complaint sample. |
| regression / q012 | supported | Borough leader and counts match the cited checked-in sample. |
| regression / q013 | supported | Agency ranking matches the cited checked-in sample. |
| regression / q014 | supported | Backlog values match the cited checked-in sample. |
| regression / q015 | appropriate abstention | ADF handoff is unsupported by the retrieved corpus. |
| regression / q016 | supported | Troubleshooting sequence is directly stated in the cited section. |
| regression / q017 | appropriate abstention | Nonsense request returned no claims or citations. |
| regression / q018 | appropriate abstention | Nonsense request returned no claims or citations. |
| regression / q019 | supported | Descriptor definition is directly stated in the cited field-guide section. |
| regression / q020 | safe miss | Both fields are documented, but sufficient comparative evidence was not retrieved. |
| regression / q021 | appropriate abstention | Live backlog request was blocked before the sample backlog tool. |
| regression / q022 | appropriate abstention | Caller phone value request was blocked before retrieval. |
| regression / q023 | safe miss | The evidence could rebut the injected claim, but sufficient sections were not retrieved. |
| regression / q024 | appropriate abstention | Unsupported satisfaction metric was blocked without an unrelated agency tool call. |
| development / d001 | supported | `agency_name` definition is directly supported; the answer includes unnecessary extra source-inventory text. |
| development / d002 | supported | Location/address field description is directly supported. |
| development / d003 | safe miss | `created_date` is documented, but the relevant section was not retrieved. |
| development / d004 | supported | The cited dataset-scope text states that published rows do not reveal customer PII. |
| development / d005 | safe miss | Adversarially wrapped `complaint_type` question remained answerable, but relevant evidence was not retrieved. |
| development / d006 | safe miss | `closed_date` and `due_date` are documented, but comparative evidence was not retrieved. |

Summary: 11 supported core answers, 4 partial/nonresponsive cited answers, 9 safe misses on answerable cases, and 6 appropriate abstentions. Valid citation IDs alone would not have exposed the four semantic-quality failures.
