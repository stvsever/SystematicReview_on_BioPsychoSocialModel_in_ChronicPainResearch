# Static knowledge graph generator

This package turns a scheme 3 full-text coding run into a local, desktop-first
knowledge graph. It writes plain HTML, CSS, and JavaScript. No server, build
tool, network request, or web application is required after generation.

## Source layout

```text
graph/
├── builder.py              # table-to-graph transformation and output writer
└── assets/
    ├── dashboard.html      # accessible static shell
    ├── dashboard.css       # desktop review interface
    └── dashboard.js        # canvas graph, physics, filters, search, and details
```

## Generated layout

```text
05_knowledge_graph/
├── index.html              # open this file in a browser
├── README.md               # run-specific opening instructions
└── assets/
    ├── styles.css
    ├── app.js
    └── graph_data.js       # graph payload assigned to a browser global
```

The JavaScript data file is used instead of a fetched JSON file so `index.html`
works directly from `file://` in browsers that block local fetch requests.

## What the graph shows

The hierarchy is

```text
run -> field group -> [entity ...] -> coding field -> provider -> article -> item
```

Headings nest to whatever depth the group declares, so an entity can hold kinds
of its own.

The initial radial overview shows only the scheme layer: the field groups of
scheme 3, the entities, and every coding field inside them. Reviewers can expand
a field to inspect provider hubs with their papers grouped beneath them, then
expand an article coding to inspect its extracted items. With one selected
provider, the redundant provider hub is omitted and its papers connect directly
to the field. A separate Show all mode renders every selected layer at once.

### The entity layer

One group nests, and it is the group the review is about. **Biopsychosocial
entities** holds the triad as three siblings, and everything the registration
adds beyond the triad under a fourth heading with children of its own:

```text
Biopsychosocial entities
├── Biological factors
│   ├── Biological factors (named)
│   └── Biological evidence and constructs
├── Psychological factors
│   ├── Psychological concepts
│   ├── Concept definitions
│   ├── Concept relations
│   └── Psychological evidence and constructs
├── Social factors
│   ├── Social factors (named)
│   └── Social evidence and constructs
└── Other factors
    ├── Lifestyle factors
    │   ├── Lifestyle factors named
    │   └── Lifestyle coverage
    └── Spiritual and existential factors
        ├── Existential factors named
        ├── Spiritual or existential coverage
        └── Environmental factors named
```

The depth is the argument. A concept definition and a social factor are different
kinds of thing, so one flat ring of coding fields would say they are alike. And
lifestyle and the existential are not a fourth and fifth domain sitting beside
biology: they are the account of what falls outside the triad, so they sit one
level down, under a heading that says exactly that.

Two of the scheme's lists hold more than one entity at a time. `domain_evidence`
is a single list covering all three domains, and `other_domain_factors` a single
list covering lifestyle, existential, and environmental factors together. Both
are split into item-filtered views (`FieldView`), so the biological evidence
appears as its own node under the biological entity rather than inside one
undifferentiated list. A view whose filter matches nothing in a run is dropped
rather than drawn as an empty node, so an older run shows exactly what it holds.

Drill-downs retain the full scheme overview as a dimmed context, and selecting an
extracted item highlights its uninterrupted path from the run root through group,
field, provider, and article. Reviewers can return with the Back one level
control or by double-clicking a visible parent node.

Node size decreases with depth. Every coding field receives a stable variation in
hue, saturation, and luminance within its field-group palette. Extracted items
carry the coder's own label, its normalized form, its ontology anchor, and the
quote-verification verdict when those tables are supplied. Reviewers can filter
by article, provider, or coding field, search all labels and evidence, drag
nodes, pan, zoom, switch theme, disable or move the node preview, inspect
formatted values, reset to the overview, and fit the layout.

The run root, field-group labels, and canonical coding-field labels remain
visible at every drill-down depth whenever Labels is enabled. Deeper article and
extracted-item views use compact automatically sized rings and render only labels
that fit without colliding. Context fitting frames the active field, provider, or
article branch instead of jumping back to the complete graph. Dragging a parent
translates its complete descendant subtree, including hidden descendants that may
be expanded later, while manual zoom supports up to 1000 percent. Reset and Show
all clear manual node placement.

## Grouping is the only scheme-specific part

`FIELD_GROUPS` in `builder.py` lays the coded fields out along the review's own
questions: how the biopsychosocial label is used, how deep each domain goes, what
the model is made of, how the domains are linked, what is measured, and what is
conceptually wrong with it. A group's value is either a flat list of columns or
an ordered map of entities, which is what gives the entity layer its extra level.
Any column the coding table carries that the table does not name still appears,
grouped under "Other coded fields", so a scheme revision never silently drops a
field from the review surface, and a test asserts the grouping can never name a
field the scheme does not produce.

## Usage

```python
from bps_review.graph import build_knowledge_graph

build_knowledge_graph(
    corpus_df=corpus,
    long_df=long_df,
    items_df=items_df,
    output_dir=graph_dir(),
    run_title="FULL-TEXT CODING SCHEME (test run)",
    run_subtitle="47 open-access reviews coded by three independent providers",
    verification_df=integrity["quote_verification"],
)
```

The full-text pipeline calls this for every run, so `make testrun-fulltext`
refreshes the graph together with the tables and the figures.
