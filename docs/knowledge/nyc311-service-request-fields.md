# NYC 311 Service Request Field Guide

## Source Classification

This is a curated CivicLens reference derived from official City of New York sources. It is external NYC 311 domain knowledge, not CivicLens project documentation and not a copy of the service-request dataset.

Official references:

- [311 Service Requests from 2020 to Present](https://data.cityofnewyork.us/d/erm2-nwe9)
- [Official Socrata dataset metadata and column descriptions](https://data.cityofnewyork.us/api/views/erm2-nwe9)
- [NYC Open Data 311 dataset updates](https://opendata.cityofnewyork.us/311-service-requests-from-2010-to-present-updates/)
- [About NYC311](https://portal.311.nyc.gov/about-nyc-311/)

The source pages were reviewed on 2026-08-17. NYC Open Data says the current dataset is updated daily, field values can change, and listed values are not exhaustive.

## Dataset Scope

The current official dataset covers service requests from 2020 to the present. NYC Open Data split 2010–2019 records into a separate historical dataset in December 2025. Each current-dataset row represents a service request that can be directed to a specific responding agency. The dataset description says published rows do not reveal customer personally identifying information.

Do not treat the curated definitions below as live operational data. CivicLens stores this small field guide for retrieval and keeps raw service-request records outside the knowledge corpus.

## Request Identity and Topic

### Unique Key (`unique_key`)

The unique key identifies a service request in the open dataset. Use it to distinguish request records; do not substitute a row number or ingestion timestamp.

### Problem / Complaint Type (`complaint_type`)

Problem is the current display label for the field formerly called Complaint Type. The API field name remains `complaint_type`. It is the broad, agency-defined first level of the hierarchy describing the incident or condition. Agencies may add values as customer demand changes.

### Problem Detail / Descriptor (`descriptor`)

Problem Detail is the current display label for the field formerly called Descriptor. The API field name remains `descriptor`. It adds detail beneath the selected problem and is not required for every request.

## Responding Agency and Status

### Agency (`agency`) and Agency Name (`agency_name`)

`agency` is the acronym of the responding City agency. `agency_name` is its full name. These fields identify the responder, not necessarily the channel that originally received the request.

### Status (`status`)

Status records the submitted service request's state. Status values are operational and can change as an agency works the request. Because expected values are not exhaustive, consumers should not hard-code an undocumented closed set without checking current source data.

## Request Timestamps

### Created Date (`created_date`)

Created Date is when the service request was created.

### Closed Date (`closed_date`)

Closed Date and the API name `closed_date` mean the same field: the date when the responding agency closed the service request. It can be empty while a request remains open and should not be interpreted as the due date.

### Due Date (`due_date`)

Due Date is when the responding agency is expected to update the request, based on the problem category and internal service-level agreements. It is an expected update date rather than evidence that the request was actually closed.

### Resolution Action Updated Date (`resolution_action_updated_date`)

Resolution Action Updated Date is when the responding agency last updated the service request. It is distinct from Created Date and Closed Date.

When calculating durations, define which timestamps are used, handle missing Closed Date values, and state the timezone assumptions of the source extract.

## Borough and Location

### Borough (`borough`)

Borough describes the incident borough. The official metadata says it is provided by the submitter and confirmed through geographic validation.

### Location Fields

`location_type` describes the kind of location used in the address information. Related fields can include incident ZIP, incident address, street or intersection details, city, latitude, longitude, and the combined `location` point. Availability varies by request and location validation, so missing values do not by themselves mean a request is invalid.

## Interpretation Limits

- These definitions describe fields, not current complaint volumes or agency performance.
- Problem, detail, status, and location values may evolve over time.
- A closed request indicates agency closure in the source system; it does not independently prove customer satisfaction or problem resolution quality.
- CivicLens must cite this guide or another manifest-authorized source and abstain when the curated evidence is insufficient.
