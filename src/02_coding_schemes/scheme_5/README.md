# Scheme 5: Psychological Concept Clustering and Framework Mapping Scheme

> **Status: DRAFT FOR EXPERT EVALUATION.** These coding schemes are a working draft circulated for expert evaluation. They have not been applied to a final review corpus. The current manuscript is a test run that exercised an earlier, coarser generation of these schemes. The workflow itself has since been validated end to end in two cross-provider test runs, in which three large language models from three different providers applied the abstract-level and the full-text scheme independently and their agreement was quantified per coded field. The full run on the review corpus is deliberately held until this evaluation is complete.

*Second-order normalization of detected psychological concepts*

Fixed pattern detection upstream, LLM clustering for higher-order normalization.

## What this scheme does

This scheme standardizes higher-order concept mapping after concept detection. It groups extracted psychological concepts from chronic pain review records into interpretable families and links them to likely theoretical frameworks. The goal is cross-record comparability when the raw concepts are heterogeneous, overlapping, or variably named.

It is a second-order coding scheme: it does not classify whole records, it normalizes the concept vocabulary itself.

## At a glance

| Property | Value |
| --- | --- |
| Workflow position | Post-detection concept normalization over Stage 2 and Stage 3 concept strings. |
| Operational mode | Fixed pattern-based concept detection upstream, followed by LLM clustering for higher-order normalization. |
| Unit of analysis | The set of unique concept strings across the corpus, not whole records. |
| Provenance basis | coding.py concept and framework patterns plus the clustering prompt. |
| Research questions | RQ3 (which psychological concepts and frameworks dominate) |

## Files in this folder

- [`scheme_5.html`](scheme_5.html) is the interactive evaluation surface. Open it in a browser, record a verdict and comments per section, then export your feedback as JSON.
- [`scheme_5.pdf`](scheme_5.pdf) is the formal dossier for sharing and printing.
- [`scheme_5.tex`](scheme_5.tex) is the LaTeX source (generated from `_build/content.py`).

## Coded fields

### Cluster Output Schema (current)

- `clusters`: Top-level list of normalized concept clusters.
- `family`: Higher-order concept family label for the cluster.
- `members` (free text): The original concept strings grouped into that family.
- `possible_frameworks` (free text): Likely theoretical or therapeutic frameworks associated with the family.

## Comprehensive Psychological Concept Taxonomy (proposed)

Each family aligns one to one with a Scheme 6 psychological subdomain.

- **Catastrophizing and negative cognitive appraisal** (subdomain: Catastrophizing and Negative Cognitive Appraisal). Exaggerated negative interpretation and mental amplification of pain and its threat value. Constructs: pain catastrophizing, rumination, magnification, helplessness, negative appraisal, worry about pain. Frameworks: fear-avoidance model, cognitive-behavioral model, communal coping model.
- **Fear, avoidance and pain-related fear** (subdomain: Fear Avoidance and Pain Related Fear). Fear of pain, movement, or reinjury and the avoidance or escape behaviour it motivates. Constructs: kinesiophobia, fear of movement or reinjury, pain-related fear, avoidance behaviour, escape behaviour, threat hypervigilance. Frameworks: fear-avoidance model, avoidance-endurance model, operant learning.
- **Depression, low mood and negative affect** (subdomain: Depression Emotional Distress and Affect). Depressive symptoms and sustained negative affect linked to living with pain. Constructs: depression, depressive symptoms, anhedonia, hopelessness, demoralization, negative affect. Frameworks: cognitive model of depression, diathesis-stress, mutual maintenance model.
- **Anxiety and psychological reactivity** (subdomain: Anxiety and Psychological Reactivity). Anxious apprehension and heightened physiological or cognitive reactivity to pain. Constructs: anxiety, pain anxiety, health anxiety, anxiety sensitivity, physiological reactivity. Frameworks: anxiety sensitivity model, fear-avoidance model.
- **Self-efficacy, control and mastery** (subdomain: Self Efficacy Control Beliefs and Perceived Mastery). Beliefs about ability to manage pain and exert control over outcomes. Constructs: pain self-efficacy, perceived control, locus of control, mastery, agency, confidence. Frameworks: social cognitive theory, self-efficacy theory.
- **Acceptance, psychological flexibility and mindfulness** (subdomain: Acceptance Psychological Flexibility and Mindfulness). Willingness to experience pain without avoidance while pursuing valued activity. Constructs: pain acceptance, psychological flexibility, values-based action, committed action, present-moment awareness, cognitive defusion, willingness. Frameworks: acceptance and commitment therapy, psychological flexibility model, mindfulness-based approaches.
- **Coping strategies and adjustment** (subdomain: Pain Coping Strategies and Adjustment). Cognitive and behavioural efforts to manage pain and the resulting adjustment. Constructs: active coping, passive coping, problem-focused coping, emotion-focused coping, adaptive coping, maladaptive coping, adjustment. Frameworks: transactional model of stress and coping, self-regulation.
- **Attention, vigilance and pain processing** (subdomain: Attention Vigilance and Pain Processing). How attention is captured by or directed away from pain and bodily threat. Constructs: attentional bias, hypervigilance, distraction, pain interruption, attentional control, somatic focus. Frameworks: threat interpretation model, cognitive-affective model of interruption.
- **Illness beliefs, pain representations and meaning** (subdomain: Illness Beliefs Pain Representations and Meaning). Beliefs about the nature, cause, timeline, and consequences of pain and its meaning. Constructs: illness perceptions, pain beliefs, causal attributions, timeline beliefs, consequence beliefs, meaning of pain, illness identity. Frameworks: common-sense model of self-regulation, illness perception framework.
- **Cognitive-behavioural and psychotherapeutic approaches** (subdomain: Cognitive Behavioral and Psychotherapeutic Approaches). Structured psychological techniques targeting pain cognition and behaviour. Constructs: cognitive restructuring, behavioural activation, graded activity, graded exposure, relaxation, psychoeducation. Frameworks: cognitive-behavioral therapy, operant and respondent conditioning.
- **Third-wave, ACT and contextual approaches** (subdomain: Third Wave Therapies ACT and Contextual Approaches). Contextual and acceptance-based therapies that target the function of experience. Constructs: acceptance, defusion, values clarification, self-as-context, mindfulness practice, functional contextualism. Frameworks: acceptance and commitment therapy, contextual behavioural science, mindfulness-based stress reduction.
- **Resilience, positive psychology and growth** (subdomain: Resilience Positive Psychology and Post Traumatic Growth). Protective strengths and positive adaptation despite persistent pain. Constructs: resilience, optimism, hope, benefit-finding, post-traumatic growth, positive affect, gratitude. Frameworks: broaden-and-build theory, resilience frameworks.
- **Identity, self-concept and pain biography** (subdomain: Identity Self Concept and Chronic Pain Biography). How pain reshapes self-concept, roles, and life narrative. Constructs: pain identity, self-discrepancy, biographical disruption, role loss, self-concept, loss of self. Frameworks: self-discrepancy theory, narrative and biographical approaches.
- **Trauma, adversity and life events** (subdomain: Trauma Adverse Childhood and Life Events). Traumatic and adverse experiences that predispose to or maintain chronic pain. Constructs: post-traumatic stress, trauma, adverse childhood experiences, abuse history, life stressors, victimization. Frameworks: mutual maintenance model, diathesis-stress, shared vulnerability model.
- **Personality and individual differences** (subdomain: Personality Psychological Traits and Individual Differences). Stable traits and dispositions that shape pain experience and response. Constructs: neuroticism, negative affectivity, harm avoidance, perfectionism, alexithymia, trait anxiety. Frameworks: trait vulnerability models, diathesis-stress models.
- **Cognitive function and executive processes** (subdomain: Cognitive Function Executive Processes and Brain Health). Cognitive resources and executive processes affected by or affecting pain. Constructs: attention, working memory, executive function, cognitive load, mental fatigue or brain fog, processing speed. Frameworks: cognitive resource models, interruption models.
- **Motivation, goal pursuit and engagement** (subdomain: Motivational Processes Goal Pursuit and Engagement). Motivational dynamics of pursuing valued goals alongside pain. Constructs: goal pursuit, goal conflict, approach and avoidance motivation, activity engagement, endurance, valued goals. Frameworks: self-regulation, motivational control theory, goal pursuit models.
- **Healthcare-seeking, adherence and engagement** (subdomain: Healthcare Seeking Treatment Adherence and Engagement). Behaviour around seeking, engaging with, and adhering to care. Constructs: treatment adherence, healthcare utilization, help-seeking, therapeutic alliance, treatment expectations, engagement. Frameworks: health belief model, common-sense model of self-regulation, expectancy models.
- **Emotion regulation and pain affect processing** (subdomain: Emotional Regulation and Pain Affect Processing). Strategies for regulating emotion and processing the affective dimension of pain. Constructs: emotion regulation, expressive suppression, cognitive reappraisal, alexithymia, emotional awareness, affect regulation. Frameworks: process model of emotion regulation, emotion regulation frameworks.
- **Mental health comorbidity and wellbeing** (subdomain: Mental Health Comorbidity and Psychological Wellbeing). Co-occurring mental health conditions and overall psychological wellbeing. Constructs: comorbid depression or anxiety, distress, quality of life, wellbeing, suicidality, life satisfaction. Frameworks: mutual maintenance model, biopsychosocial wellbeing models.

## Canonical source paths

- `src/03_pipeline/bps_review/extraction/coding.py`
- `src/01_protocol/codebooks/stage2_codebook.md`
- `src/01_protocol/codebooks/stage3_codebook.md`
- `src/05_data/interim/extraction/llm_concept_clusters.json`

## Regenerating this dossier

All three surfaces (PDF, HTML, README) are generated from one source of truth:

```bash
cd src/coding_schemes/_build
python3 build.py
```

Edit the scheme content in `_build/content.py`, not the generated files.
