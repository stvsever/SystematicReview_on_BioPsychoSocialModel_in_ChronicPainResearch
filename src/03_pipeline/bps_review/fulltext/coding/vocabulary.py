from __future__ import annotations

"""Preferred labels and conservative normalization for the Stage 3 scheme.

Stage 3 is an extraction pass, not a classification pass. What it harvests from a
full text is a set of *named things*: which biological, psychological, and social
factors a review actually carries, which theoretical frameworks it invokes, which
instruments it measures with, and which pain conditions it is about. Those names
are the raw material of the biopsychosocial ontology the synthesis will build, so
they have to be recorded at the resolution the paper wrote them in.

Two demands pull against each other here, and this module holds both.

1. **Comparability.** Two reviews that both discuss central sensitization should
   land on the same node of the ontology even when one calls it "central
   sensitisation" and the other "centrally mediated hypersensitivity". The
   canonical label sets below give that shared spine. They mirror the coding
   scheme dossiers: the subdomain vocabularies are the Scheme 6 ontology, and the
   concept families are the Scheme 5 taxonomy, so the full-text pass, the concept
   map, and the semantic ontology speak one language.
2. **Resolution.** A review that names something the spine does not contain is
   telling us something the spine is missing. Forcing it into the nearest bucket
   destroys exactly the finding this review exists to make. So the normalizer is
   deliberately conservative: an exact match on a canonical label, or a
   whole-token match on one of its lexical variants, and otherwise the free text
   survives cleaned but unchanged, flagged as off-spine.

The share of extracted labels that land on the spine is itself reported by the
pipeline. A low share is not a failure of the coder; it is a measurement of how
well the ontology covers the literature, and it is what tells the review team
which nodes to add.

Note on the biological ontology. The canonical subdomains below follow the
revised Scheme 6 structure that is currently out for expert evaluation: a shared
biological core plus a musculoskeletal extension and a neuropathic extension,
because the biological mechanisms of the two pain families genuinely differ. The
semantic-loading prompts in ``bps_review.reporting.semantic_loading`` still carry
the earlier single biological list; they will be aligned once the revised
ontology is signed off. The older wordings are kept here as lexical variants, so
a label normalizes the same way under either version.
"""

import re


# --------------------------------------------------------------------------
# Biological subdomains: shared core, plus one extension per review track.
# --------------------------------------------------------------------------
BIO_SUBDOMAIN_CORE: dict[str, tuple[str, ...]] = {
    "Central Sensitization and Neuroplasticity": (
        "central sensitization", "central sensitisation", "neuroplasticity",
        "central hypersensitivity", "wind-up", "nociplastic",
    ),
    "Nociceptive and Pain Signalling Pathways": (
        "nociception", "nociceptive", "pain pathway", "pain signalling", "pain signaling",
        "afferent input", "descending modulation", "nociceptive transmission",
        "nociceptive transmission and pain pathways",
    ),
    "Immune Inflammatory and Neuroinflammatory Processes": (
        "inflammation", "inflammatory", "neuroinflammation", "cytokine", "immune",
        "glial activation",
    ),
    "Neuroimaging Brain Structure and Function": (
        "neuroimaging", "fmri", "brain imaging", "cortical reorganization",
        "grey matter", "gray matter", "brain structure",
    ),
    "Genetic Epigenetic and Biological Vulnerability": (
        "genetic", "genomic", "epigenetic", "heritability", "polymorphism",
        "biological vulnerability",
    ),
    "Sleep Disruption and Circadian Dysregulation": (
        "sleep", "insomnia", "circadian", "sleep disturbance", "sleep quality",
    ),
    "Pharmacological and Biomedical Treatment": (
        "pharmacological", "pharmacotherapy", "medication", "opioid", "analgesic",
        "injection", "surgery", "biomedical treatment",
    ),
    "Metabolic Nutritional and Hormonal Factors": (
        "metabolic", "obesity", "nutrition", "diet", "hormonal", "endocrine",
        "cortisol", "hpa axis",
    ),
    "Physical Function Mobility and Deconditioning": (
        "physical function", "mobility", "deconditioning", "muscle strength",
        "range of motion", "physical capacity", "disuse",
    ),
}

BIO_SUBDOMAIN_MSK: dict[str, tuple[str, ...]] = {
    "Musculoskeletal and Structural Pathology": (
        "musculoskeletal pathology", "structural pathology", "spinal pathology",
        "disc degeneration", "posture", "biomechanic", "musculoskeletal and structural pathology",
    ),
    "Joint Degeneration and Osteoarthritic Change": (
        "joint degeneration", "osteoarthritic", "cartilage", "joint space narrowing",
        "radiographic osteoarthritis",
    ),
    "Muscle Tendon and Soft Tissue Pathology": (
        "muscle pathology", "tendinopathy", "tendon", "soft tissue", "myofascial",
        "trigger point", "muscle tension",
    ),
}

BIO_SUBDOMAIN_NEUROPATHIC: dict[str, tuple[str, ...]] = {
    "Peripheral Nerve Injury and Neuropathy": (
        "nerve injury", "neuropathy", "nerve damage", "radiculopathy", "nerve compression",
        "polyneuropathy",
    ),
    "Sensory Phenotype and Quantitative Sensory Testing": (
        "sensory phenotype", "quantitative sensory testing", "qst", "allodynia",
        "hyperalgesia", "sensory profile",
    ),
    "Ectopic Firing Ion Channels and Neuronal Excitability": (
        "ectopic firing", "ion channel", "sodium channel", "neuronal excitability",
        "spontaneous activity",
    ),
    "Small Fiber and Nerve Conduction Pathology": (
        "small fiber", "small fibre", "nerve conduction", "intraepidermal nerve fiber",
        "electrophysiology",
    ),
    "Deafferentation and Central Neuropathic Mechanisms": (
        "deafferentation", "central neuropathic", "thalamic", "phantom",
        "spinal cord injury pain",
    ),
}

BIO_SUBDOMAIN_VOCAB: dict[str, tuple[str, ...]] = {
    **BIO_SUBDOMAIN_CORE,
    **BIO_SUBDOMAIN_MSK,
    **BIO_SUBDOMAIN_NEUROPATHIC,
}


# --------------------------------------------------------------------------
# Psychological subdomains (Scheme 6), uniform across both review tracks.
# --------------------------------------------------------------------------
PSYCH_SUBDOMAIN_VOCAB: dict[str, tuple[str, ...]] = {
    "Catastrophizing and Negative Cognitive Appraisal": (
        "catastrophizing", "catastrophising", "rumination", "magnification", "helplessness",
        "negative appraisal",
    ),
    "Fear Avoidance and Pain Related Fear": (
        "fear-avoidance", "fear avoidance", "kinesiophobia", "pain-related fear",
        "fear of movement", "fear of reinjury", "avoidance behaviour", "avoidance behavior",
    ),
    "Depression Emotional Distress and Affect": (
        "depression", "depressive symptoms", "low mood", "negative affect", "emotional distress",
        "demoralization",
    ),
    "Anxiety and Psychological Reactivity": (
        "anxiety", "pain anxiety", "health anxiety", "anxiety sensitivity", "worry",
    ),
    "Self Efficacy Control Beliefs and Perceived Mastery": (
        "self-efficacy", "self efficacy", "perceived control", "locus of control", "mastery",
        "agency",
    ),
    "Acceptance Psychological Flexibility and Mindfulness": (
        "acceptance", "psychological flexibility", "mindfulness", "values-based action",
        "committed action", "cognitive defusion",
    ),
    "Pain Coping Strategies and Adjustment": (
        "coping", "active coping", "passive coping", "adjustment", "problem-focused coping",
        "emotion-focused coping",
    ),
    "Attention Vigilance and Pain Processing": (
        "attentional bias", "hypervigilance", "vigilance", "distraction", "somatic focus",
        "attention to pain",
    ),
    "Illness Beliefs Pain Representations and Meaning": (
        "illness perception", "illness perceptions", "pain beliefs", "illness representation",
        "causal attribution", "meaning of pain",
    ),
    "Cognitive Behavioral and Psychotherapeutic Approaches": (
        "cognitive behavioural therapy", "cognitive behavioral therapy", "cbt",
        "cognitive restructuring", "graded activity", "graded exposure", "psychoeducation",
    ),
    "Third Wave Therapies ACT and Contextual Approaches": (
        "acceptance and commitment therapy", "third wave", "contextual behavioural science",
        "mindfulness-based stress reduction", "mbsr",
    ),
    "Resilience Positive Psychology and Post Traumatic Growth": (
        "resilience", "optimism", "hope", "benefit-finding", "post-traumatic growth",
        "positive affect",
    ),
    "Identity Self Concept and Chronic Pain Biography": (
        "identity", "self-concept", "biographical disruption", "role loss", "loss of self",
        "pain identity",
    ),
    "Trauma Adverse Childhood and Life Events": (
        "trauma", "ptsd", "post-traumatic stress", "adverse childhood experiences",
        "abuse history", "life stressors",
    ),
    "Personality Psychological Traits and Individual Differences": (
        "personality", "neuroticism", "negative affectivity", "perfectionism", "alexithymia",
        "trait anxiety",
    ),
    "Cognitive Function Executive Processes and Brain Health": (
        "executive function", "working memory", "cognitive load", "brain fog",
        "processing speed", "cognitive impairment",
    ),
    "Motivational Processes Goal Pursuit and Engagement": (
        "goal pursuit", "goal conflict", "motivation", "activity engagement", "endurance",
        "valued goals",
    ),
    "Healthcare Seeking Treatment Adherence and Engagement": (
        "treatment adherence", "healthcare utilization", "healthcare utilisation",
        "help-seeking", "therapeutic alliance", "treatment expectations",
    ),
    "Emotional Regulation and Pain Affect Processing": (
        "emotion regulation", "emotional regulation", "expressive suppression",
        "cognitive reappraisal", "emotional awareness",
    ),
    "Mental Health Comorbidity and Psychological Wellbeing": (
        "mental health comorbidity", "psychiatric comorbidity", "wellbeing", "well-being",
        "quality of life", "suicidality",
    ),
}


# --------------------------------------------------------------------------
# Social subdomains (Scheme 6), uniform across both review tracks.
# --------------------------------------------------------------------------
SOCIAL_SUBDOMAIN_VOCAB: dict[str, tuple[str, ...]] = {
    "Social Support Network and Interpersonal Resources": (
        "social support", "perceived support", "social network", "interpersonal resources",
        "significant other responses",
    ),
    "Work Disability Occupational Function and Productivity": (
        "work disability", "sick leave", "absenteeism", "presenteeism", "job demands",
        "occupational function", "work ability",
    ),
    "Family Caregiver and Household Dynamics": (
        "family", "spouse", "partner", "caregiver", "household", "solicitous responses",
    ),
    "Socioeconomic Status and Health Inequity": (
        "socioeconomic", "income", "poverty", "deprivation", "health inequity",
        "health inequality", "social class",
    ),
    "Healthcare Access Navigation and System Factors": (
        "healthcare access", "access to care", "waiting time", "health system",
        "care pathway", "service delivery", "insurance",
    ),
    "Cultural Ethnic and Demographic Context": (
        "culture", "cultural", "ethnicity", "race", "migrant", "language barrier",
    ),
    "Community Participation and Social Role Functioning": (
        "social participation", "community participation", "role functioning",
        "social activities", "social functioning",
    ),
    "Legal Compensation and Medicolegal Systems": (
        "compensation", "litigation", "medicolegal", "disability benefits", "insurance claim",
        "workers compensation",
    ),
    "Health Literacy Education and Patient Empowerment": (
        "health literacy", "education level", "patient empowerment", "shared decision making",
        "patient education",
    ),
    "Stigma Social Isolation and Exclusion": (
        "stigma", "stigmatization", "invalidation", "social isolation", "loneliness",
        "exclusion", "disbelief",
    ),
    "Return to Work Vocational Rehabilitation and Employment": (
        "return to work", "vocational rehabilitation", "employment", "work resumption",
        "workplace intervention",
    ),
    "Social Determinants of Pain and Environment": (
        "social determinants", "neighbourhood", "neighborhood", "housing", "environment",
        "structural factors",
    ),
}

SUBDOMAIN_VOCAB_BY_DOMAIN: dict[str, dict[str, tuple[str, ...]]] = {
    "biological": BIO_SUBDOMAIN_VOCAB,
    "psychological": PSYCH_SUBDOMAIN_VOCAB,
    "social": SOCIAL_SUBDOMAIN_VOCAB,
}


# --------------------------------------------------------------------------
# Psychological concept families (the Scheme 5 taxonomy), used for the
# concept_family field so the full-text concepts and the concept map align.
# --------------------------------------------------------------------------
CONCEPT_FAMILY_VOCAB: dict[str, tuple[str, ...]] = {
    "catastrophizing and negative cognitive appraisal": ("catastrophizing", "catastrophising", "rumination"),
    "fear, avoidance and pain-related fear": ("fear-avoidance", "kinesiophobia", "pain-related fear"),
    "depression, low mood and negative affect": ("depression", "low mood", "negative affect"),
    "anxiety and psychological reactivity": ("anxiety", "anxiety sensitivity", "pain anxiety"),
    "self-efficacy, control and mastery": ("self-efficacy", "perceived control", "mastery"),
    "acceptance, psychological flexibility and mindfulness": ("acceptance", "psychological flexibility", "mindfulness"),
    "coping strategies and adjustment": ("coping", "adjustment"),
    "attention, vigilance and pain processing": ("attentional bias", "hypervigilance"),
    "illness beliefs, pain representations and meaning": ("illness perception", "pain beliefs", "meaning of pain"),
    "cognitive-behavioural and psychotherapeutic approaches": ("cognitive behavioural therapy", "cbt", "graded activity"),
    "third-wave, ACT and contextual approaches": ("acceptance and commitment therapy", "act", "mbsr"),
    "resilience, positive psychology and growth": ("resilience", "optimism", "post-traumatic growth"),
    "identity, self-concept and pain biography": ("identity", "self-concept", "biographical disruption"),
    "trauma, adversity and life events": ("trauma", "ptsd", "adverse childhood experiences"),
    "personality and individual differences": ("personality", "neuroticism", "alexithymia"),
    "cognitive function and executive processes": ("executive function", "working memory", "brain fog"),
    "motivation, goal pursuit and engagement": ("goal pursuit", "motivation", "goal conflict"),
    "healthcare-seeking, adherence and engagement": ("adherence", "help-seeking", "therapeutic alliance"),
    "emotion regulation and pain affect processing": ("emotion regulation", "reappraisal", "suppression"),
    "mental health comorbidity and wellbeing": ("comorbidity", "wellbeing", "quality of life"),
}


# --------------------------------------------------------------------------
# Psychological constructs. The Scheme 5 family members, which are the concept
# labels the abstract-level pass already detects, extended with the constructs
# that only a full text names.
# --------------------------------------------------------------------------
PSYCH_CONCEPT_VOCAB: dict[str, tuple[str, ...]] = {
    "pain catastrophizing": ("catastrophizing", "catastrophising", "catastrophic thinking"),
    "rumination": ("rumination", "ruminating"),
    "helplessness": ("helplessness", "learned helplessness"),
    "kinesiophobia": ("kinesiophobia", "fear of movement", "fear of re-injury", "fear of reinjury"),
    "pain-related fear": ("pain-related fear", "pain related fear", "fear of pain"),
    "fear-avoidance beliefs": ("fear-avoidance beliefs", "fear avoidance beliefs"),
    "avoidance behaviour": ("avoidance behaviour", "avoidance behavior", "activity avoidance"),
    "endurance behaviour": ("endurance behaviour", "endurance behavior", "overactivity", "task persistence"),
    "depression": ("depression", "depressive symptoms", "major depressive disorder"),
    "anxiety": ("anxiety", "anxious symptoms", "generalized anxiety"),
    "anxiety sensitivity": ("anxiety sensitivity",),
    "psychological distress": ("psychological distress", "emotional distress", "general distress"),
    "pain self-efficacy": ("pain self-efficacy", "self-efficacy", "self efficacy"),
    "perceived control": ("perceived control", "locus of control", "sense of control"),
    "pain acceptance": ("pain acceptance", "acceptance of pain"),
    "psychological flexibility": ("psychological flexibility", "flexibility model"),
    "mindfulness": ("mindfulness", "present-moment awareness"),
    "active coping": ("active coping", "adaptive coping"),
    "passive coping": ("passive coping", "maladaptive coping"),
    "hypervigilance": ("hypervigilance", "vigilance to pain", "body vigilance"),
    "attentional bias": ("attentional bias", "attention bias"),
    "illness perceptions": ("illness perception", "illness perceptions", "illness representations"),
    "pain beliefs": ("pain beliefs", "beliefs about pain", "harm beliefs"),
    "expectations": ("expectations", "treatment expectations", "outcome expectancy", "expectancy"),
    "resilience": ("resilience", "psychological resilience"),
    "optimism": ("optimism", "dispositional optimism"),
    "emotion regulation": ("emotion regulation", "emotional regulation", "affect regulation"),
    "alexithymia": ("alexithymia",),
    "perceived injustice": ("perceived injustice", "injustice appraisal"),
    "self-compassion": ("self-compassion", "self compassion"),
    "pain identity": ("pain identity", "illness identity", "loss of self"),
    "post-traumatic stress": ("post-traumatic stress", "posttraumatic stress", "ptsd"),
    "adverse childhood experiences": ("adverse childhood experiences", "childhood adversity", "aces"),
    "sleep-related cognitions": ("sleep beliefs", "pre-sleep worry", "sleep-related cognition"),
    "treatment adherence": ("adherence", "compliance", "treatment engagement"),
    "therapeutic alliance": ("therapeutic alliance", "working alliance"),
    "goal pursuit": ("goal pursuit", "goal conflict", "goal adjustment"),
    "quality of life": ("quality of life", "health-related quality of life", "hrqol"),
    "stress": ("stress", "perceived stress", "chronic stress"),
    "somatization": ("somatization", "somatisation", "somatic symptom"),
    "personality traits": ("personality", "neuroticism", "negative affectivity"),
    "self-management": ("self-management", "self management", "self-regulation of pain"),
}


# --------------------------------------------------------------------------
# Theoretical frameworks and models.
# --------------------------------------------------------------------------
FRAMEWORK_VOCAB: dict[str, tuple[str, ...]] = {
    "biopsychosocial model": ("biopsychosocial model", "biopsychosocial framework", "bps model",
                              "bio-psycho-social model"),
    "engel's biopsychosocial model": ("engel", "engel's model", "engel 1977"),
    "fear-avoidance model": ("fear-avoidance model", "fear avoidance model", "vlaeyen and linton",
                             "vlaeyen & linton"),
    "avoidance-endurance model": ("avoidance-endurance", "avoidance endurance model", "hasenbring"),
    "gate control theory": ("gate control", "melzack and wall"),
    "neuromatrix theory": ("neuromatrix", "body-self neuromatrix"),
    "diathesis-stress model": ("diathesis-stress", "diathesis stress"),
    "cognitive behavioural model": ("cognitive behavioural model", "cognitive behavioral model",
                                    "cognitive-behavioural framework"),
    "operant learning model": ("operant", "fordyce", "reinforcement model"),
    "classical conditioning and extinction model": ("classical conditioning", "respondent conditioning",
                                                    "extinction learning", "fear extinction"),
    "common-sense model of self-regulation": ("common-sense model", "common sense model",
                                              "leventhal", "self-regulation model"),
    "psychological flexibility model": ("psychological flexibility model", "act model",
                                        "acceptance and commitment"),
    "social cognitive theory": ("social cognitive theory", "bandura"),
    "transactional model of stress and coping": ("transactional model", "lazarus and folkman",
                                                 "lazarus & folkman"),
    "misdirected problem solving model": ("misdirected problem solving", "misdirected problem-solving"),
    "motivational model of pain": ("motivational model", "motivational account of pain"),
    "predictive processing model": ("predictive processing", "predictive coding", "bayesian brain"),
    "enactive or embodied approaches": ("enactive", "embodied", "4e cognition"),
    "ICF framework": ("international classification of functioning", "icf framework"),
    "self-determination theory": ("self-determination theory", "self determination theory"),
    "communal coping model": ("communal coping",),
    "perceived injustice model": ("perceived injustice model", "injustice model"),
    "shared vulnerability model": ("shared vulnerability", "mutual maintenance"),
    "sociopsychobiological model": ("sociopsychobiological", "socio-psycho-biological"),
    "stress-diathesis neuroendocrine model": ("hpa axis model", "allostatic load"),
    "chronic care model": ("chronic care model", "stepped care model"),
}


# --------------------------------------------------------------------------
# Measurement instruments and appraisal tools that carry the operationalization.
# --------------------------------------------------------------------------
INSTRUMENT_VOCAB: dict[str, tuple[str, ...]] = {
    "PCS (Pain Catastrophizing Scale)": ("pain catastrophizing scale", "pcs"),
    "TSK (Tampa Scale for Kinesiophobia)": ("tampa scale", "tsk", "tsk-11", "tsk-17"),
    "FABQ (Fear-Avoidance Beliefs Questionnaire)": ("fear-avoidance beliefs questionnaire", "fabq"),
    "PASS (Pain Anxiety Symptoms Scale)": ("pain anxiety symptoms scale", "pass-20", "pass"),
    "PSEQ (Pain Self-Efficacy Questionnaire)": ("pain self-efficacy questionnaire", "pseq"),
    "CPAQ (Chronic Pain Acceptance Questionnaire)": ("chronic pain acceptance questionnaire", "cpaq"),
    "HADS (Hospital Anxiety and Depression Scale)": ("hospital anxiety and depression scale", "hads"),
    "BDI (Beck Depression Inventory)": ("beck depression inventory", "bdi"),
    "PHQ-9": ("phq-9", "patient health questionnaire"),
    "GAD-7": ("gad-7", "generalized anxiety disorder scale"),
    "ODI (Oswestry Disability Index)": ("oswestry", "odi"),
    "RMDQ (Roland-Morris Disability Questionnaire)": ("roland-morris", "roland morris", "rmdq"),
    "BPI (Brief Pain Inventory)": ("brief pain inventory", "bpi"),
    "SF-36": ("sf-36", "short form 36", "sf36"),
    "PROMIS": ("promis",),
    "EQ-5D": ("eq-5d", "euroqol"),
    "CSI (Central Sensitization Inventory)": ("central sensitization inventory", "csi"),
    "QST (Quantitative Sensory Testing)": ("quantitative sensory testing", "qst"),
    "DN4 or painDETECT": ("dn4", "paindetect", "pain detect"),
    "IEQ (Injustice Experience Questionnaire)": ("injustice experience questionnaire", "ieq"),
    "MSPSS (Multidimensional Scale of Perceived Social Support)": ("mspss", "perceived social support scale"),
    "WAI (Work Ability Index)": ("work ability index", "wai"),
    "ICF core sets": ("icf core set", "icf core sets"),
    "AMSTAR or AMSTAR-2": ("amstar", "amstar-2"),
    "ROBIS": ("robis",),
    "GRADE": ("grade approach", "grade assessment"),
    "Cochrane risk of bias tool": ("risk of bias tool", "rob 2", "cochrane risk of bias"),
}


# --------------------------------------------------------------------------
# Pain conditions, for the pain_conditions list.
# --------------------------------------------------------------------------
PAIN_CONDITION_VOCAB: dict[str, tuple[str, ...]] = {
    "chronic low back pain": ("chronic low back pain", "low back pain", "clbp", "lumbar pain"),
    "neck pain": ("neck pain", "cervical pain", "whiplash"),
    "knee osteoarthritis": ("knee osteoarthritis", "knee oa", "gonarthrosis"),
    "osteoarthritis": ("osteoarthritis", "arthrosis"),
    "fibromyalgia": ("fibromyalgia", "fibromyalgia syndrome"),
    "shoulder pain": ("shoulder pain", "rotator cuff", "subacromial"),
    "rheumatoid arthritis": ("rheumatoid arthritis",),
    "temporomandibular disorder": ("temporomandibular", "tmd"),
    "chronic widespread pain": ("chronic widespread pain", "widespread pain"),
    "tendinopathy": ("tendinopathy", "tendinitis", "tendon pain"),
    "painful diabetic neuropathy": ("diabetic neuropathy", "painful diabetic polyneuropathy"),
    "postherpetic neuralgia": ("postherpetic neuralgia", "post-herpetic neuralgia"),
    "radicular pain or sciatica": ("radicular", "sciatica", "radiculopathy"),
    "complex regional pain syndrome": ("complex regional pain syndrome", "crps"),
    "trigeminal neuralgia": ("trigeminal neuralgia",),
    "chemotherapy-induced peripheral neuropathy": ("chemotherapy-induced", "cipn"),
    "spinal cord injury pain": ("spinal cord injury pain",),
    "post-surgical persistent pain": ("post-surgical pain", "postsurgical pain", "persistent postoperative pain"),
    "headache or migraine": ("headache", "migraine"),
    "pelvic or lumbopelvic pain": ("pelvic pain", "lumbopelvic", "pelvic girdle"),
    "mixed chronic pain": ("mixed chronic pain", "heterogeneous chronic pain", "chronic pain sample"),
}


# --------------------------------------------------------------------------
# Sources the biopsychosocial model is credited to. Who a review cites for the
# model is part of how it uses it, and it is what makes lineage visible.
# --------------------------------------------------------------------------
BPS_SOURCE_VOCAB: dict[str, tuple[str, ...]] = {
    "Engel (1977 or 1980)": ("engel", "engel 1977", "engel 1980", "engel's"),
    "Gatchel and colleagues": ("gatchel",),
    "Waddell": ("waddell",),
    "Turk and colleagues": ("turk and", "turk &", "turk 2"),
    "Melzack (neuromatrix)": ("melzack", "neuromatrix"),
    "Vlaeyen and Linton": ("vlaeyen", "linton"),
    "IASP": ("iasp", "international association for the study of pain"),
    "WHO or ICF": ("world health organization", "icf"),
    "clinical guideline": ("nice guideline", "clinical guideline", "practice guideline"),
    "own synthesis": ("we define", "our definition", "in this review we"),
    "unattributed": ("unattributed", "no citation", "none"),
}


_ALL_VOCABS: dict[str, dict[str, tuple[str, ...]]] = {
    "bio_subdomain": BIO_SUBDOMAIN_VOCAB,
    "psych_subdomain": PSYCH_SUBDOMAIN_VOCAB,
    "social_subdomain": SOCIAL_SUBDOMAIN_VOCAB,
    "concept_family": CONCEPT_FAMILY_VOCAB,
    "psych_concept": PSYCH_CONCEPT_VOCAB,
    "framework": FRAMEWORK_VOCAB,
    "instrument": INSTRUMENT_VOCAB,
    "pain_condition": PAIN_CONDITION_VOCAB,
    "bps_source": BPS_SOURCE_VOCAB,
}


def vocabulary(kind: str) -> dict[str, tuple[str, ...]]:
    return _ALL_VOCABS.get(kind, {})


def controlled_labels(kind: str) -> list[str]:
    """The canonical labels of one vocabulary, for the prompt and the dossier."""
    return list(_ALL_VOCABS.get(kind, {}).keys())


def clean_label(value: object) -> str:
    """Lowercase, collapse whitespace, strip surrounding punctuation."""
    text = " ".join(str(value or "").strip().lower().split())
    return re.sub(r"^[\s\-•]+|[\s\.;,:]+$", "", text)


def normalize_label(value: object, kind: str) -> str:
    """Map a free-text label onto a canonical label when it clearly matches.

    Conservative on purpose: an exact match on the canonical label, or a
    whole-token match on one of its lexical variants. Anything else comes back
    cleaned but unchanged, because a precise label the spine does not carry is
    worth more to this review than a controlled label that flattens it.
    """
    text = clean_label(value)
    if not text:
        return ""
    for canonical, variants in _ALL_VOCABS.get(kind, {}).items():
        if text == canonical.lower():
            return canonical
        for variant in variants:
            variant = variant.strip().lower()
            if not variant:
                continue
            if text == variant:
                return canonical
            if re.search(rf"(?<![a-z0-9]){re.escape(variant)}(?![a-z0-9])", text):
                return canonical
    return text


def normalize_labels(values: object, kind: str, cap: int | None = None) -> list[str]:
    """Normalize a list of labels, de-duplicated and order preserving."""
    out: list[str] = []
    if not isinstance(values, (list, tuple)):
        return out
    for value in values:
        if isinstance(value, dict):
            value = value.get("label") or value.get("value") or ""
        label = normalize_label(value, kind)
        if label and label not in out:
            out.append(label)
        if cap is not None and len(out) >= cap:
            break
    return out


def is_controlled(label: str, kind: str) -> bool:
    """Whether a normalized label sits on the controlled spine of its vocabulary."""
    return label in _ALL_VOCABS.get(kind, {})


def subdomain_kind_for_domain(domain: str) -> str:
    """Which subdomain vocabulary applies to a factor in this domain."""
    return {
        "biological": "bio_subdomain",
        "psychological": "psych_subdomain",
        "social": "social_subdomain",
    }.get(str(domain or "").strip().lower(), "")


def vocabulary_overview() -> dict[str, int]:
    """Size of every vocabulary, for the notebook and the dossier."""
    return {kind: len(vocab) for kind, vocab in _ALL_VOCABS.items()}
