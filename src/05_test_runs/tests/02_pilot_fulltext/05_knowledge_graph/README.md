# Knowledge graph review surface

Open `index.html` in a desktop browser. The bundle is fully local and requires no server.

- Papers: 47
- Providers: 3
- Coding cells: 141
- Graph nodes: 31339
- Graph links: 31338

`assets/graph_data.js` holds the whole run. Every string that occurs more than once in it, as a value or as a key, is written once into a string table and referenced as `~<index>` wherever it appeared, so a file that would otherwise be about 60 MB of mostly repeated article titles, provider names, and field names is about 16 MB. The dashboard restores it on load, exactly; nothing is summarized or dropped. The search text of a node is not stored at all, because it is derived: it is the node's label, article, provider, and field plus every key and leaf value of its detail block, and the dashboard builds it when the file opens.

Search accepts several words at once, all of which must match, and ranks what it finds: a quoted phrase stays contiguous, a leading minus excludes a word, and field:, group:, provider:, article:, label:, and type: aim a word at one part of a node. The filter panel and the inspector each fold away from the toolbar to give the canvas their width.

The first view shows the field groups, the biopsychosocial entities, and all canonical scheme 3 coding fields. The entity level holds the triad as three siblings and everything beyond it under Other factors, which carries lifestyle and spiritual or existential as its own children, so the evidence for one domain sits under that domain rather than in one undifferentiated list.

Every coding field explains itself: its card and its inspector state what the field records, list its possible values when the field has a closed vocabulary, give its value format when it does not, spell out the rung-by-rung rule for the coverage and integration ladders, and, for a structured extraction list, name every item field with its own vocabulary.

Double-click a field or use its Explore button to reveal provider hubs with papers grouped beneath them, then expand an article coding to reveal extracted items. With one selected provider, papers connect directly to the field. Use Show all to render every selected layer. Use the left panel to filter articles, providers, and coding fields. Drag nodes to pin them, drag the background to pan, use the mouse wheel to zoom, switch theme, move or disable the node preview, and click a node for its formatted inspector. The complete root-to-leaf path stays highlighted while the scheme overview remains visible as context. Whenever Labels is enabled, the run root, field-group labels, and canonical coding-field labels stay visible at every drill-down depth. Back one level and parent-node double-clicks move upward. Deep article views use compact automatically sized rings and collision-aware leaf labels. Context fitting frames the active branch, and dragging a parent moves its complete descendant subtree, including hidden descendants expanded later. Manual zoom supports up to 1000 percent. Reset view returns to the complete scheme overview and clears manual placement.
