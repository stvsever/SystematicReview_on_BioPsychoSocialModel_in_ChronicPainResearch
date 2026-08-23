# Coderingen

Drie taalmodellen hebben elk afzonderlijk het full-text codeerschema toegepast op dezelfde 47
open-access reviews. Dit zijn alle coderingen daarvan: 141 ingevulde formulieren en 9400
geëxtraheerde items.

Het gebruikte codeerschema is [scheme 3](../../../../02_coding_schemes/scheme_3/). Open daar
`scheme_3.html` in je browser voor de definitie van elk veld.

Elke tabel begint met de referentie, de titel, de auteurs en de DOI van de paper, niet alleen met
het interne ID. Elke lijst staat opgesplitst over kolommen, één item per kolom. Elk geëxtraheerd
item heeft een eigen rij met eigen kolommen. Dezelfde inhoud staat bewust in meerdere vormen: kies
de vorm die past bij wat je wil nakijken.

## Waar begin je

Elk bestand is genoemd naar wat één rij erin is. De mapinhoud zegt dus meteen welk bestand je nodig
hebt, zonder er een te openen.

| Wat je wil | Bestand |
| --- | --- |
| Alles in één Excel-bestand, met vaste kopregels en filters | [`00_workbook.xlsx`](00_workbook.xlsx) |
| De biologische, psychologische en sociale factoren van één review, naast elkaar | [`02_extracted_items/00_all_categories/02_one_row_per_named_factor_of_any_domain.csv`](02_extracted_items/00_all_categories/02_one_row_per_named_factor_of_any_domain.csv) |
| Enkel de biologische factoren | [`02_extracted_items/01_biological_factors/`](02_extracted_items/01_biological_factors/) |
| Enkel de psychologische concepten, met hun definitiestatus | [`02_extracted_items/02_psychological_concepts/`](02_extracted_items/02_psychological_concepts/) |
| Enkel de sociale factoren | [`02_extracted_items/03_social_factors/`](02_extracted_items/03_social_factors/) |
| De uitspraken die twee of drie domeinen aan elkaar koppelen | [`02_extracted_items/06_integration_claims/`](02_extracted_items/06_integration_claims/) |
| Eén codering van één model volledig lezen, met alle citaten | [`01_codings/01_one_row_per_paper_and_provider.csv`](01_codings/01_one_row_per_paper_and_provider.csv) |
| Nagaan of de drie modellen het eens waren over een paper | [`01_codings/02_one_row_per_paper.csv`](01_codings/02_one_row_per_paper.csv) |
| Zien hoe één model zich over het hele corpus gedroeg | [`01_codings/03_one_row_per_provider.csv`](01_codings/03_one_row_per_provider.csv) |
| Alles over één paper, de drie modellen onder elkaar | [`01_codings/05_the_same_rows_split_per_paper/`](01_codings/05_the_same_rows_split_per_paper/) |
| Enkel DeepSeek-V4-Flash, het primaire model | [`03_by_provider/01_deepseek_v4_flash/`](03_by_provider/01_deepseek_v4_flash/) |
| De lijst papers met volledige referentie, DOI en PubMed-link | [`../01_corpus/papers.csv`](../01_corpus/papers.csv) |

## Mappen

### `01_codings/` : één ingevuld formulier per paper en model

| Bestand | Eén rij is | Rijen |
| --- | --- | --- |
| `01_one_row_per_paper_and_provider.csv` | Eén paper, gecodeerd door één model. **Alle** velden, inclusief elk geëxtraheerd item met zijn letterlijke citaat: 935 kolommen | 141 |
| `02_one_row_per_paper.csv` | Eén paper, de drie modellen naast elkaar | 47 |
| `03_one_row_per_provider.csv` | Eén model, de 47 papers naast elkaar | 3 |
| `04_the_same_rows_split_per_provider/` | Dezelfde 141 rijen, gesplitst per model: één bestand van 47 rijen per model | |
| `05_the_same_rows_split_per_paper/` | Dezelfde 141 rijen, gesplitst per paper: één bestand van 3 rijen per paper, genoemd naar auteur en jaar | |

Het boekhoudkundige spoor van de run zelf (status, pogingen, duur, tokens en kost per API-call, plus
het manifest en het logboek) blijft lokaal. Dat zegt hoe de run tot stand kwam, niet wat er gecodeerd
werd.

Het eerste bestand is de volledige neerslag: naast de samenvattende kolommen staat elk item dat het
model vond in eigen genummerde kolommen, bijvoorbeeld `biological_factors_01_factor_label`,
`biological_factors_01_mechanism_level`, `biological_factors_01_factor_verbatim`, en zo verder voor
alle dertien categorieën. Eén rij is dus de volledige lezing van één paper door één model.

Breed per paper toont waar de modellen van elkaar afwijken; breed per model toont of een model
systematisch dieper codeert of meer extraheert. De twee submappen bevatten geen nieuwe informatie:
het zijn dezelfde rijen, alleen alvast gefilterd.

### `02_extracted_items/` : wat de modellen uit de teksten haalden

De volgorde van de mappen volgt wat deze review wil zien, niet de volgorde van het formulier: eerst
de genoemde dingen die elk domein dragen, dan de verbanden ertussen, dan het biopsychosociale label
zelf, dan de theorieën en meetinstrumenten, dan de kritiek.

| Map | Bevat | Items |
| --- | --- | --- |
| `00_all_categories/` | Alle categorieën samen, één rij per item, plus een aparte tabel met **alle benoemde factoren van alle domeinen** onder elkaar | 9400 |
| `01_biological_factors/` | Elke biologische factor die de review benoemt, met zijn mechanismeniveau en zijn rol | 1180 |
| `02_psychological_concepts/` | Elk psychologisch construct, en of de review zegt wat het betekent | 1564 |
| `03_social_factors/` | Elke sociale factor, met het niveau waarop hij speelt (interpersoonlijk, werk, beleid, ...) | 888 |
| `04_other_domain_factors/` | Elke factor buiten de trias: leefstijl, spiritueel of existentieel, omgeving | 455 |
| `05_domain_evidence/` | De passage waarop het coverage-oordeel per domein steunt | 404 |
| `06_integration_claims/` | Elke uitspraak die twee of drie domeinen aan elkaar koppelt, met beide uiteinden benoemd | 820 |
| `07_concept_relations/` | Elke relatie tussen twee psychologische constructen (subtype van, oorzaak van, synoniem, ...) | 1118 |
| `08_bps_usage_instances/` | Elke plaats waar de review het biopsychosociale label inroept, en wat het daar doet | 576 |
| `09_bps_definitions/` | Elke plaats waar de review zegt wat het biopsychosociale model is | 97 |
| `10_theoretical_frameworks/` | Elk theoretisch model, en welke domeinen het overspant | 513 |
| `11_instruments/` | Elk meetinstrument, en welk domein het meet | 529 |
| `12_conceptual_problems/` | Elk conceptueel probleem dat de review vertoont of zelf benoemt | 573 |
| `13_key_quotes/` | Elke conceptueel belangrijke uitspraak, letterlijk geciteerd | 683 |

Elke categoriemap heeft dezelfde vier vormen:

| Bestand | Eén rij is |
| --- | --- |
| `01_one_row_per_<categorie>.csv` | Eén geëxtraheerd item, met alle velden van die categorie en het letterlijke citaat. Bijvoorbeeld `01_one_row_per_biological_factor.csv` |
| `02_one_row_per_paper_and_provider.csv` | Eén paper en model, met alle items van dat model in genummerde kolommen, naam en citaat naast elkaar |
| `03_one_row_per_paper.csv` | Eén paper, met de namen van de drie modellen naast elkaar |
| `04_one_row_per_provider.csv` | Eén model, met zijn namen voor alle 47 papers naast elkaar |

Elke itemrij bevat een `*_verbatim` kolom: de zin uit de paper waarop het model zich baseerde. Of
die zin echt in de brontekst staat, is apart nagegaan in
[`../03_reliability/10_quote_verification_by_model.csv`](../03_reliability/10_quote_verification_by_model.csv);
in deze run was 99,2 procent van de 8302 gecontroleerde citaten terug te vinden.

### `03_by_provider/` : alles van één model bij elkaar

Eén map per model, met daarin volledig en op volle resolutie:

| Bestand | Inhoud |
| --- | --- |
| `01_codings_one_row_per_paper.csv` | De 47 coderingen van dat model, alle 935 kolommen, dus inclusief elk geëxtraheerd item met citaat |
| `02_extracted_items_one_row_per_item.csv` | Al zijn geëxtraheerde items, alle categorieën, één rij per item |
| `03_extracted_items_per_category/` | Dezelfde items, opgesplitst per categorie |

De drie modellen: `01_deepseek_v4_flash` (primair), `02_nex_n2_mini`, `03_laguna_xs_2_1`.

## De kolommen lezen

**Referentie (de eerste kolommen van elke tabel).** `record_id` is het interne ID, `citation` de
korte referentie, daarna `title`, `authors`, `publication_year`, `journal`, `doi` en `doi_url`. Het
corpus komt uit PubMed Central, dus die gegevens komen rechtstreeks van de uitgever. In
`../01_corpus/papers.csv` staan daarnaast ook `pmid`, `pmcid`, `pmc_url` en de licentie.

**Model.** `model_order` is de vaste volgorde, `model_label` het taalmodel dat de rij produceerde,
`provider` de leverancier, `model_id` de exacte versie die werd aangeroepen. `coding_method` is
`llm_structured` bij een geslaagde codering en `coding_failed` als het model na alle pogingen niets
bruikbaars teruggaf. In deze run is dat nul keer het geval, na een herstelronde voor Nex-N2-Mini.

**Oordeel.** `fulltext_eligibility` is include, uncertain of exclude, met
`fulltext_exclusion_reason` bij exclude. `bps_typology` is het door het model gecodeerde type;
`derived_typology` is hetzelfde type opnieuw berekend uit coverage en integratie volgens een vaste
regel, en `typology_matches_derived` zegt of de twee samenvallen. Dat is de scherpste test of de
typologie strak genoeg gedefinieerd is om twee keer hetzelfde toegepast te worden.
`conceptual_yield` en `synthesis_priority` worden per regel afgeleid uit de gecodeerde inhoud en
niet aan het model gevraagd, zodat dezelfde inhoud altijd hetzelfde oordeel geeft.

**Domeindekking.** `domain_coverage_bio`, `_psych` en `_social` staan op een ladder van vier:
elaborated, mentioned, minimal, absent. `coverage_lifestyle` en `coverage_spiritual_existential`
doen hetzelfde voor de twee domeinen buiten de trias. De `coverage_depth_*` kolommen zijn diezelfde
ladder als getal (0 tot 3), `coverage_total` is de som over de drie kerndomeinen en
`domains_present` telt hoeveel domeinen minstens `mentioned` halen.

**Integratie.** `integration_bio_psych`, `integration_psych_social` en `integration_bio_social`
staan op een ladder van vijf: mechanistic, directional, descriptive, mentioned, none.
`integration_triadic` heeft een eigen ladder van vier: mechanistic, descriptive, partial, none.
`integration_index` vat die vier in één getal van 0 tot 1 samen, zodat papers vergelijkbaar zijn
zonder te doen alsof de ladder een intervalschaal is. `n_named_integration_edges` telt hoeveel
koppelingen beide uiteinden benoemen, wat het aantal bruikbare verbindingen is dat een paper
bijdraagt.

**Het label zelf.** `bps_label_used` zegt of de woorden biopsychosociaal effectief vallen, of enkel
een variant, of enkel domeintaal. `bps_definition_status` zegt of de review ergens zegt wat het
model betekent. `bps_primary_function` is wat het label vooral doet in deze review, en
`bps_function_set` is de volledige verzameling functies, ook die welke enkel uit de geëxtraheerde
passages blijken.

**Gesplitste lijsten.** Een kolom die eindigt op `_count` geeft het aantal items; de genummerde
kolommen erna bevatten ze één per kolom. Bijvoorbeeld: `pain_conditions_count` is 3 en
`pain_conditions_1` tot en met `pain_conditions_3` bevatten de drie pijnaandoeningen.

**Tellingen en vlaggen.** `n_biological_factors`, `n_psychological_concepts` enzovoort tellen wat
werd geëxtraheerd; `present_biological_factors`, `present_social_factors` enzovoort zijn yes of no.
Die vlaggen worden afgelezen uit de gecodeerde inhoud en niet aan het model gevraagd: een `no` is
dus de vaststelling dat dit model hier niets van dat type vond.

**Ontologie.** `n_subdomains_bio`, `_psych` en `_social` tellen hoeveel verschillende subdomeinen de
review aanraakt, wat de breedte van haar verhaal is. `controlled_label_share` is welk aandeel van de
geëxtraheerde labels op de projectwoordenlijsten landde. Dat meet de ontologie aan de literatuur, en
niet omgekeerd: een laag aandeel zegt dat de woordenlijsten uitbreiding nodig hebben, niet dat het
model zich vergiste.

**Vrije tekst.** `bps_operationalization_summary`, `integration_mechanism_summary`, `context_note`,
`pain_condition_detail`, `synthesis_note` en `coding_rationale` zijn het proza van het model zelf.
In `coding_rationale` legt het uit hoe het twijfelgevallen beslechtte.

**Enkel in de brede tabellen.** `n_providers_include` is hoeveel van de drie include zeiden.
`eligibility_agreement` en `typology_agreement` zijn `unanimous` als alle drie hetzelfde zeiden,
`majority` als twee van de drie hetzelfde zeiden, en `no majority` als alle drie iets anders zeiden.
`modal_eligibility` en `modal_bps_typology` geven dan die meerderheidswaarde, en blijven leeg bij
`no majority`: bij drie verschillende antwoorden bestaat er geen meerderheid, en er toch een
aanwijzen zou er een verzinnen. Daarna komen de kolommen per model, bijvoorbeeld
`deepseek_v4_flash__bps_typology`, of per paper, bijvoorbeeld `F002_41976799__bps_typology`.

## Gesloten waardenlijsten

Onderstaande velden nemen enkel deze waarden aan. Al de rest is vrije tekst.

| Veld | Waarden |
| --- | --- |
| `fulltext_eligibility` | include, uncertain, exclude |
| `conceptual_yield` | high, moderate, low, minimal |
| `synthesis_priority` | core, supporting, background, not_relevant |
| `review_track` | musculoskeletal, neuropathic, mixed_or_other, unclear |
| `domain_coverage_*`, `coverage_*` | elaborated, mentioned, minimal, absent |
| `integration_bio_psych`, `integration_psych_social`, `integration_bio_social` | mechanistic, directional, descriptive, mentioned, none |
| `integration_triadic` | mechanistic, descriptive, partial, none |
| `overall_balance` | balanced, psych-dominant, bio-dominant, social-dominant, dyadic, unclear |
| `bps_typology`, `derived_typology` | true_integrative, multifactorial, pseudo_bps, rhetorical_bps, narrow_despite_label, unclear |
| `bps_label_used` | explicit_bps_term, variant_term_only, domain_language_only, absent |
| `bps_definition_status` | formally_defined, described_informally, cited_only, undefined |
| `bps_primary_function`, `bps_function` | explanatory framework, intervention rationale, organizing principle, justification, background framing, conclusion, policy or practice implication, rhetorical label, critique or problematization, operational definition, unclear |
| `concept_definitions_present` | yes, partial, no |
| `source_type` | systematic review, meta-analysis, network meta-analysis, umbrella review, scoping or mapping review, rapid review, realist review, integrative review, narrative or expert review, clinical guideline or consensus statement, other evidence synthesis, primary study, unclear |
| `icd11_pain_category` | chronic secondary musculoskeletal pain, chronic neuropathic pain, chronic cancer-related pain, chronic postsurgical or posttraumatic pain, chronic secondary headache or orofacial pain, chronic secondary visceral pain, chronic primary pain, mixed or unspecified chronic pain, unclear |
| `population` | adult, older adult, mixed ages, pediatric, unclear, not applicable |
| `care_setting` | primary care, secondary or tertiary specialist care, rehabilitation or multidisciplinary programme, occupational or workplace, community or population, mixed, not reported |
| `primary_discipline` | physiotherapy or rehabilitation, clinical or health psychology, rheumatology or orthopaedics, pain medicine or anaesthesiology, neurology or neuroscience, nursing, general or family medicine, public health or epidemiology, multidisciplinary, other, unclear |
| `factor_role` | determinant or risk factor, protective factor, mediator, moderator, outcome, correlate, treatment target, intervention component, contextual condition, descriptive theme, other, unclear |
| `mechanism_level` (biologisch) | peripheral or tissue, spinal or central nervous system, systemic or whole body, genetic or molecular, structural or anatomical, treatment related, other, unclear |
| `social_level` (sociaal) | interpersonal, family or household, workplace, community, healthcare system, societal or policy, cultural, economic, other, unclear |
| `domain` (buiten de trias) | lifestyle, spiritual or existential, environmental, other |
| `definitional_status` (concepten) | formally_defined, operationalized_only, described_informally, named_only, unclear |
| `definition_source` (concepten) | own definition, cited from other work, taken from an instrument, unattributed, unclear |
| `definition_type` (BPS-definities) | explicit_formal, operational, implicit_description, borrowed, critique_of_definition, other |
| `relation_type` (conceptrelaties) | is_a_subtype_of, part_of_or_component_of, synonym_or_used_interchangeably, overlapping_or_related, antecedent_or_cause_of, consequence_or_outcome_of, mediates, moderates, measured_by, contrasted_as_distinct_from, conflated_without_comment, other, unclear |
| `domains_linked` (integratie) | bio_psych, psych_social, bio_social, triadic |
| `integration_level` (integratie) | mechanistic, directional, descriptive, mentioned, none |
| `direction` (integratie) | unidirectional, bidirectional or reciprocal, unspecified |
| `role` (kaders) | organizing framework, tested or modelled, extended or revised, critiqued or rejected, compared with another model, mentioned in passing, other, unclear |
| `role` (instrumenten) | primary outcome, secondary outcome, predictor or covariate, mediator or moderator, screening or classification, developed or validated here, discussed conceptually, critiqued, referenced only, other, unclear |
| `domain_measured` (instrumenten) | biological, psychological, social, pain or symptom, function or disability, quality of life, multiple domains, methodological quality, other, unclear |
| `problem_type` | vague_definition, tokenistic_bps, missing_social, missing_biology, missing_psychology, mechanistic_absence, construct_overlap, parallel_listing_without_integration, measurement_mismatch, definitional_drift, domain_reductionism, unfalsifiable_or_untestable, other |
| `problem_scope` | the biopsychosocial model itself, a psychological construct, a biological construct, a social construct, integration between domains, measurement, terminology, scope or coverage, other |
| `claim_type` (key quotes) | definitional, integrative, operationalizing, critical or problematizing, measurement, theoretical, clinical or applied, other |
| `evidence_basis` | asserted, theorized, empirically_supported, empirically_contested, cited_from_other_work, clinical_observation, other, unclear |
| `section_located` | abstract, introduction, methods, results, discussion, conclusion, table or figure, other, unclear |

## Namen en woordenlijsten

`item_label` is hoe een item heet: bij elf van de dertien categorieën is dat één veld, bij
conceptrelaties en integratie-uitspraken is het het paar dat verbonden wordt, want die twee zijn
verbindingen en geen dingen.

Twee normalisaties staan naast een item, en elk staat er enkel waar ze iets betekent.

`label_normalized_for_matching` koppelt de naam van het item aan een gecontroleerde woordenlijst, en
`label_is_controlled` zegt of dat lukte. Slechts drie categorieën hebben zo'n woordenlijst:
psychologische concepten, theoretische kaders en meetinstrumenten. Bij de rest staan die kolommen er
bewust niet, want een genormaliseerde naam suggereren waar geen woordenlijst bestaat, wekt de indruk
van een koppeling die nooit gebeurd is.

`ontology_anchor` is iets anders: dat is waar het item vasthangt aan de projectontologie. Vijf
categorieën hebben zo'n anker. Bij biologische factoren, sociale factoren en psychologische
concepten is het anker een **apart** veld naast de naam: bij een biologische factor is de naam
`factor_label` (het woord van de review) en het anker `subdomain_label` (het subdomein van de
ontologie). Zo meet het aandeel op de ontologie daar nooit hoe vaak de codeur toevallig onze woorden
voor het ding zelf gebruikte. Bij theoretische kaders en meetinstrumenten valt het anker wel samen
met de naam, omdat een kader of een instrument nu eenmaal zelf de ontologische entiteit is.

Een item zonder anker is geen fout: het is een term die de ontologie nog niet draagt, en dat is
precies wat de synthese moet zien. In deze run landde 79,1 procent van de gecontroleerde labels op
de woordenlijsten; de 702 die dat niet deden staan in
[`../03_reliability/15_off_spine_labels.csv`](../03_reliability/15_off_spine_labels.csv), de
werklijst om de woordenlijsten mee uit te breiden.

Lees dus altijd `factor_label` en `concept_label`, niet de genormaliseerde vorm.

## Hoe één codering is opgebouwd

Elk model kreeg één volledige tekst per keer en vulde daarvoor hetzelfde gestructureerde formulier
in. Eén ingevuld formulier is één codering: 47 papers x 3 modellen = 141 coderingen.

Een formulier bevat vier soorten velden. **Gesloten velden** met één waarde uit een vaste lijst; dat
zijn de velden waarop overeenstemming tussen modellen wordt gemeten. **Vrijetekstvelden** met het
proza van het model. **Open lijsten** zonder itemstructuur, zoals `conceptual_tensions` en
`emergent_labels`. En **extractielijsten**: dertien lijsten waarin het model noteert wat het vond,
item per item, telkens met een letterlijk citaat uit de paper. Die dertien lijsten zijn de dertien
mappen in [`02_extracted_items/`](02_extracted_items/), en staan ook volledig uitgeklapt in de
kolommen van [`01_codings/01_one_row_per_paper_and_provider.csv`](01_codings/01_one_row_per_paper_and_provider.csv).

## Volledigheid

Deze tabellen zijn de volledige neerslag van de coderingen, niet een selectie eruit. Elk veld dat
het codeerschema definieert staat erin, en elk geëxtraheerd item met zijn letterlijke citaat.
Dezelfde inhoud staat bewust in meerdere vormen, zodat je kan kiezen wat past bij wat je wil
nakijken.

De run wordt hier ook uit teruggelezen wanneer de tabellen opnieuw worden opgebouwd. Dat wordt
afgedwongen door de test suite: een run wegschrijven en weer inlezen moet veld voor veld dezelfde
run opleveren, tot en met elk item van elke extractielijst.

## Reproduceren

De run zit in deze tabellen. Alles hierboven opnieuw opbouwen kost geen enkele API-call:

```bash
python -m bps_review run-fulltext-testrun
```

Het corpus volledig hercoderen vereist `force_coding=True` en roept de drie leveranciers wel aan.
De generator van deze tabellen is
[`src/03_pipeline/bps_review/fulltext/publish.py`](../../../../03_pipeline/bps_review/fulltext/publish.py).
