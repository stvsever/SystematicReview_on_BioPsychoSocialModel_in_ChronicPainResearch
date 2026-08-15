# Semantic space

The ontology-aligned embedding space of the review, and the loadings computed
from it. This is a synthesis output rather than a stage of its own: it takes the
coded records and asks how far each one sits from the biological, psychological,
and social poles of the project ontology, which is the continuous counterpart of
the categorical coverage ladder.

Written by `python -m bps_review semantic-loading`
(`bps_review.reporting.semantic_loading`), and read back by the asset builder.

```text
semantic_space/
├── records/
│   ├── semantic_corpus.jsonl        # the text each record was embedded from
│   └── record_embeddings.npy        # one vector per record
├── ontology/
│   ├── ontology_terms.json          # the domain and subdomain term lists
│   ├── ontology_embeddings.npy      # one vector per domain
│   └── subdomain_embeddings.npy     # one vector per subdomain
└── analysis/
    ├── record_domain_loadings.csv       # per record, per domain
    ├── record_subdomain_loadings.csv    # per record, per subdomain
    ├── domain_loading_summary.csv       # corpus-level domain loadings
    ├── subdomain_loading_summary.csv    # corpus-level subdomain loadings
    ├── pairwise_domain_loadings.csv     # per record, per domain pair
    ├── pairwise_domain_summary.csv      # corpus-level domain pairs
    └── review_type_domain_dominance.csv # which domain dominates, by review type
```

Not to be confused with the semantic overlap of the full-text run
(`03_reliability/16_semantic_extraction_overlap.csv`). That one embeds extraction
*labels* to ask whether two providers named the same thing. This one embeds
*records* to ask where a paper sits in the domain space.
