# API Access Notes

This repository can use three external search channels:

## PubMed

- Directly usable without institutional subscription.
- Set `NCBI_EMAIL` in `.env` (use: stijn.vanseveren@ugent.be).
- Optionally set `NCBI_API_KEY` for higher throughput.
- Official documentation: <https://www.ncbi.nlm.nih.gov/books/NBK25499/>

## Web of Science

- The repository currently targets the Clarivate Web of Science Starter API for credential checks and basic retrieval.
- Set `CLARIVATE_API_KEY` in `.env` if Ghent University provides Clarivate API entitlement.
- If only interface access is available, export `.ris` files manually and place them in `src/data/manual_imports/`.
- Official documentation: <https://developer.clarivate.com/apis/wos-starter>

## PsycINFO

- The repository currently supports PsycINFO through the EBSCO Discovery Service API.
- Required `.env` variables: `EDS_API_USER`, `EDS_API_PASSWORD`, `EDS_API_INTERFACE_ID` (or `EDS_API_INTERFACE`), `EDS_API_PROFILE`, and `EDS_API_ORG`.
- If Ghent University does not provide EDS API credentials, export `.ris` files manually from EBSCOhost and place them in `src/data/manual_imports/`.
- Official documentation: <https://developer.ebsco.com/eds-api>

## Practical next step for institutional access

If you want the API path rather than manual exports, contact the Ghent University library or the relevant vendor contact and request:

1. Clarivate Web of Science API entitlement and an API key.
2. EBSCO EDS API credentials and a profile that includes PsycINFO.
3. Confirmation that use for systematic-review retrieval is permitted under your institutional agreement.

## Suggested message payload for UGent library or e-resources team

Please request all of the following explicitly:

1. Web of Science Starter API key for institutional use linked to Ghent University access.
2. EBSCO EDS API credentials for PsycINFO, including UserId, Password, InterfaceId, Profile, and Org.
3. Permission confirmation for scripted retrieval for systematic-review methodology.
4. Any rate limits or daily quota limits for both services.
