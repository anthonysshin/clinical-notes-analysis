#!/usr/bin/env python3
"""
13_PsychKeywordsConstruction.py - Reproducible PSYCH_KEYWORDS Construction

Builds the PSYCH_KEYWORDS list (used by 6e_Eval_KeywordAugmented.py and
6f_Eval_KeywordAugmentedCoT.py) via a fully documented, reproducible 3-step
procedure for the medication and condition terms, plus three independent
fixed-rule categories. See Supplementary Note 7 for the full write-up.

Step 1 (candidate pools): candidate generic medication names and candidate
  condition/diagnosis terms were compiled by prompting an AI assistant to
  review psychdb.com's medication pages and the official ICD-10-CM F00-F99
  category descriptions, respectively. That one-time compilation step is not
  itself a script -- see reference_data/medication_candidates.csv and
  reference_data/condition_candidates.csv (committed files, with the exact
  prompt used documented in each file's header) and --reproduce-from-scratch
  mode below, which reads those same files. The default run below reads the
  original candidate files these numbers were reported from, which predate
  this distinction and are kept as-is (PSYCHDB_MEDICATIONS, CONDITION_CANDIDATES_PATH).

Step 2 (frequency-rank filter): each candidate pool is ranked by how many
  training notes contain each term, and the top MED_TOP_N / COND_TOP_N are
  kept (approximately the top 40% of each pool).

Step 3 (enrichment-based additions): terms not covered by Steps 1-2 --
  brand-name medications and common symptom words -- are added by comparing
  how often each word appears in sentences containing a Step-1 candidate term
  vs. all other training-note sentences (see compute_enrichment_scores()),
  keeping words at or above RATIO_THRESHOLD (and above FREQ_THRESHOLD total
  mentions) after excluding dosing/formulation/administrative noise terms
  (see NOISE_EXCLUSIONS).

Independent fixed-rule categories (not frequency-derived, not revisited by
Step 3):
  F-codes:          top codes by training-set cumulative frequency, 90% coverage
                     target (see 12_PsychKeywordsReproducibility.py for the full
                     frequency table and split-stability check).
  Abbreviations:     standard, unambiguous psychiatric abbreviations
                     (ADHD, GAD, MDD, OCD, PTSD).
  Section headers:   note-structure header lines ("Label:") found in the training
                     set, filtered to those whose label text relates to diagnosis,
                     medication, mental status, psychiatric history, or substance
                     use.

Output:
    outputs/13_PsychKeywordsConstruction/psych_keywords_final.py   (ready-to-paste list)
    outputs/13_PsychKeywordsConstruction/medication_frequency.csv
    outputs/13_PsychKeywordsConstruction/condition_frequency_final.csv
    outputs/13_PsychKeywordsConstruction/header_frequency.csv
    outputs/13_PsychKeywordsConstruction/indicator_word_scores.csv
    outputs/13_PsychKeywordsConstruction/summary.json

Reproduce-from-scratch mode (--reproduce-from-scratch):
    The default run above reads pre-computed candidate files that predate the
    reference_data/*.csv files (see above), and uses a hardcoded top-N count
    (MED_TOP_N, COND_TOP_N) for the Step-2 selection cutoff. This flag instead
    reads the documented reference_data/*.csv candidate files directly, and
    computes the Step-2 cutoff as the top 40% of each candidate pool by
    training-set frequency (see select_top_percentage()) -- a count derived
    from the candidate pool size, not a hardcoded target. It also runs the
    same Step-3 enrichment computation on the training notes -- pure
    counting, no model involved, so it is fully deterministic. It writes its
    own separate, clearly-labeled output (psych_keywords_reproduced_demo.py)
    rather than overwriting psych_keywords_final.py, so it cannot change the
    list actually used in evaluation; it demonstrates that the full
    procedure runs end-to-end from raw data plus the two reference_data/*.csv
    files alone, with no GPU required.

Usage:
    python 13_PsychKeywordsConstruction.py
    python 13_PsychKeywordsConstruction.py --reproduce-from-scratch
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd

from config import RANDOM_SEED, set_all_seeds, TRAIN_DATA_PATH

set_all_seeds(RANDOM_SEED)

_parser = argparse.ArgumentParser(description='Reproducible PSYCH_KEYWORDS construction')
_parser.add_argument('--reproduce-from-scratch', action='store_true',
                      help='Compute word-enrichment scores directly from training notes '
                           'instead of reading a pre-computed file. See module docstring.')
_args = _parser.parse_args()

OUT_DIR = Path('outputs/13_PsychKeywordsConstruction')
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Candidate files used by --reproduce-from-scratch mode (see module docstring).
REPRODUCED_MEDICATION_CANDIDATES_PATH = 'reference_data/medication_candidates.csv'
REPRODUCED_CONDITION_CANDIDATES_PATH = 'reference_data/condition_candidates.csv'

# Candidate file used by the default (already-reported) pipeline below. Predates
# reference_data/condition_candidates.csv; kept as-is so the already-reported
# list and downstream F1 numbers never silently change.
CONDITION_CANDIDATES_PATH = 'outputs/12_PsychKeywordsReproducibility/condition_candidates_stage1.txt'

# psychdb.com/meds generic medication names (fetched 2026-09-30), deduplicated,
# brand names dropped, class-level terms kept alongside individual drugs. Used
# by the default (already-reported) pipeline below, for the same reason as
# CONDITION_CANDIDATES_PATH above.
PSYCHDB_MEDICATIONS = [
    'citalopram', 'escitalopram', 'fluoxetine', 'fluvoxamine', 'paroxetine', 'sertraline',
    'bupropion', 'trazodone', 'amitriptyline', 'clomipramine', 'desipramine', 'doxepin',
    'imipramine', 'nortriptyline', 'duloxetine', 'levomilnacipran', 'desvenlafaxine',
    'venlafaxine', 'mirtazapine', 'moclobemide', 'phenelzine', 'rasagiline', 'selegiline',
    'tranylcypromine', 'vilazodone', 'vortioxetine',
    'haloperidol', 'flupentixol', 'loxapine', 'zuclopenthixol', 'chlorpromazine',
    'methotrimeprazine', 'fluphenazine', 'perphenazine',
    'risperidone', 'paliperidone', 'aripiprazole', 'brexpiprazole', 'cariprazine',
    'olanzapine', 'ziprasidone', 'lurasidone', 'quetiapine', 'clozapine', 'amisulpride',
    'pimavanserin', 'asenapine', 'tetrabenazine', 'valbenazine',
    'lithium', 'valproic acid', 'divalproex', 'carbamazepine', 'gabapentin', 'lamotrigine',
    'levetiracetam', 'phenytoin', 'pregabalin', 'oxcarbazepine',
    'diazepam', 'clonazepam', 'alprazolam', 'lorazepam', 'oxazepam', 'temazepam',
    'triazolam', 'midazolam', 'clobazam', 'estazolam', 'flurazepam', 'quazepam',
    'benzodiazepine',
    'eszopiclone', 'zaleplon', 'zolpidem', 'zopiclone',
    'baclofen', 'buspirone', 'hydroxyzine', 'propranolol',
    'agomelatine', 'melatonin', 'ramelteon', 'daridorexant', 'lemborexant', 'suvorexant',
    'prazosin', 'clonidine', 'guanfacine',
    'dextroamphetamine', 'lisdexamfetamine', 'methylphenidate',
    'donepezil', 'galantamine', 'memantine', 'rivastigmine',
    'amantadine', 'bromocriptine', 'pramipexole', 'ropinirole',
    'ketamine', 'esketamine',
    'benztropine', 'diphenhydramine', 'procyclidine', 'trihexyphenidyl',
    'buprenorphine', 'methadone', 'naloxone', 'naltrexone', 'varenicline',
    'acamprosate', 'disulfiram', 'topiramate',
]

# Fixed top-N counts are used instead of a raw frequency threshold, giving a
# precise, exactly reproducible cutoff regardless of small changes in the
# underlying word counts.
MED_TOP_N = 45
COND_TOP_N = 56
HEADER_TOP_N = 16

FIXED_ABBREVIATIONS = ['ADHD', 'GAD', 'MDD', 'OCD', 'PTSD']

# Generic psychiatric/substance-use relevance indicator words. These are not
# diagnosis-specific, so they are not produced by the ICD-10-CM extraction, but
# the keyword-extraction preprocessing retains whole sentences containing any
# match, so broad indicator words matter for recall of relevant context, not
# just diagnostic specificity. Fixed list, not frequency-ranked, same treatment
# as FIXED_ABBREVIATIONS above.
FIXED_CONDITION_INDICATORS = [
    'psychiatric', 'mental', 'substance', 'trauma', 'confusion', 'confused',
    'cognitive', 'panic', 'disorder', 'abuse', 'alcohol', 'tobacco', 'seizure',
]

# Words that surface as "enriched" near clinical content but carry no diagnostic
# signal themselves (dosing, chemical salt forms, administrative/chart phrasing,
# etc.). Excluded when selecting Step-3 additional terms, both below and in
# --reproduce-from-scratch mode. Same categories applied to the test set in
# 14_PsychKeywordsTestDerived.py, for a like-for-like leakage comparison.
DOSING_SCHEDULE = {'qhs', 'qam', 'qday', 'qpm', 'qid', 'bid', 'tid', 'prn', 'bedtime',
                    'nightly', 'noon', 'daily', 'once', 'three', 'wake'}
CHEMICAL_SALT_FORM = {'oxalate', 'mesylate', 'fumarate', 'tartrate', 'hcl', 'carbonate',
                       'polacrilex', 'succinate', 'maleate', 'citrate', 'sulfate'}
ROUTE_FORMULATION = {'solution', 'spray', 'release', 'sustained', 'disintegrating',
                      'delayed', 'extended', 'transdermal', 'film', 'tablets', 'tabs',
                      'oral', 'sublingual', 'acting', 'half'}
ADMIN_CHART_ABBREV = {'pmhx', 'pmh', 'cont', 'prescribing', 'trials', 'use', 'continued',
                       'pfo', 'hld', 'niddm', 'ibs', 'presents', 'held', 'takes', 'remote',
                       'adding', 'precautions', 'intentional'}
UNRELATED_CONDITION = {'fibromyalgia', 'migraines', 'toxicity', 'pack', 'mouth'}
VAGUE_GENERIC = {'features', 'waning', 'waxing', 'untreated', 'suffering', 'fear'}
GENERIC_MED_PROCESS_VERBS = {'restart', 'titrated', 'tapering', 'weaning', 'uptitrate',
                              'uptitrated', 'uptitration', 'discontinuing', 'trialed'}
VAGUE_NONDIAGNOSTIC = {'disability', 'deficit', 'diagnoses', 'longstanding'}
SIDE_EFFECT_TANGENTIAL = {'itch', 'itching', 'pruritis', 'pruritus'}
NOISE_ARTIFACTS = {'melatin', 'maalox', 'fioricet', 'precedex'}
NOISE_EXCLUSIONS = (DOSING_SCHEDULE | CHEMICAL_SALT_FORM | ROUTE_FORMULATION | ADMIN_CHART_ABBREV |
                     UNRELATED_CONDITION | VAGUE_GENERIC | GENERIC_MED_PROCESS_VERBS |
                     VAGUE_NONDIAGNOSTIC | SIDE_EFFECT_TANGENTIAL | NOISE_ARTIFACTS)

# Brand-name medications recognized among Step-3 additional terms (vs. condition
# additions); same list used for the test-set check in 14_PsychKeywordsTestDerived.py.
BRAND_INCLUDE = {'prozac', 'klonopin', 'neurontin', 'depakote', 'ativan', 'haldol',
                 'zyprexa', 'keppra', 'valium', 'dilantin', 'suboxone', 'topamax',
                 'ritalin', 'latuda', 'buproprion', 'narcan', 'lyrica', 'celexa',
                 'wellbutrin', 'effexor', 'ambien', 'cymbalta', 'abilify', 'adderall',
                 'lamictal', 'lexapro', 'risperdal', 'xanax', 'seroquel', 'zoloft',
                 'benadryl'}

# Step 3 thresholds: a word must co-occur with a Step-1 candidate term often
# enough (FREQ_THRESHOLD, combined diagnostic + other mentions) and disproportionately
# often (RATIO_THRESHOLD, normalized by each side's total word count) to be added.
# Same mechanism and same threshold values applied to the test set in
# 14_PsychKeywordsTestDerived.py's leakage check (there, FREQ_THRESHOLD is scaled
# down by the train/test size ratio since the test set has fewer notes).
RATIO_THRESHOLD = 3.5
FREQ_THRESHOLD = 500

STOPWORDS = set("""a an the and or but if is was were be been being to of in on at for with
without by from as it its this that these those he she they we you i his her their our your
not no yes do does did have has had will would can could should may might shall must
mg po qd bid tid qid prn year old male female patient pt s p also noted well into than
""".split())


def word_boundary_count(term: str, texts_lower) -> int:
    pattern = re.compile(r'\b' + re.escape(term.lower()) + r'\b')
    return sum(1 for t in texts_lower if pattern.search(t))


def compute_enrichment_scores(texts_lower: list, diagnostic_terms: set) -> pd.DataFrame:
    """Step 3's enrichment computation: for every word in the training notes,
    how much more often it appears (proportionally) in sentences that already
    contain a Step-1 candidate term (medication or condition) vs. all other
    sentences. Pure counting over the training set; no model involved, so
    this is fully deterministic. Same computation used for the test-set
    leakage check in 14_PsychKeywordsTestDerived.py."""
    diag_pattern = re.compile(
        r'\b(' + '|'.join(re.escape(t) for t in sorted(diagnostic_terms, key=len, reverse=True)) + r')\b'
    )
    word_pattern = re.compile(r"[a-z']+")

    diag_word_counts = Counter()
    other_word_counts = Counter()
    for text in texts_lower:
        for sentence in re.split(r'(?<=[.!?])\s+|\n+', text):
            words = word_pattern.findall(sentence)
            target = diag_word_counts if diag_pattern.search(sentence) else other_word_counts
            target.update(w for w in words if len(w) >= 3 and w not in STOPWORDS)

    total_diag = sum(diag_word_counts.values())
    total_other = sum(other_word_counts.values())

    rows = []
    for word, diag_c in diag_word_counts.items():
        other_c = other_word_counts.get(word, 0)
        total = diag_c + other_c
        if total < FREQ_THRESHOLD:
            continue
        ratio = (diag_c / total_diag) / (other_c / total_other) if other_c > 0 else float('inf')
        rows.append((word, diag_c, other_c, total, ratio))

    df = pd.DataFrame(rows, columns=[
        'word', 'count_in_diagnostic_sentences', 'count_in_other_sentences',
        'total_count', 'enrichment_ratio',
    ])
    return df.sort_values('enrichment_ratio', ascending=False).reset_index(drop=True)


def select_additional_terms(indicator_df: pd.DataFrame, exclude_terms: set) -> list:
    """Step 3's selection rule: words with enrichment ratio >= RATIO_THRESHOLD,
    not already covered by Steps 1-2, and not matching the noise-exclusion
    categories (NOISE_EXCLUSIONS)."""
    candidates = indicator_df[
        (indicator_df['enrichment_ratio'] >= RATIO_THRESHOLD)
        & ~indicator_df['word'].isin(exclude_terms)
        & ~indicator_df['word'].isin(NOISE_EXCLUSIONS)
    ]
    return sorted(candidates['word'].tolist())


def select_top_percentage(candidates: list, texts_lower: list, percentage: float = 0.40) -> list:
    """Selection rule for --reproduce-from-scratch mode: rank candidates by
    how many training notes contain each one (most to least frequent), then
    keep the top `percentage` fraction of the candidate pool, rounded to the
    nearest whole number, and discard the rest.

    This mirrors the real procedure's approach (a rank-based cutoff, not a
    frequency-count threshold) but computes the cutoff count from the
    candidate pool size instead of hardcoding it, so the same ~40% rule
    applies regardless of how many candidates Step 1 happens to produce.
    """
    rows = [(c, word_boundary_count(c, texts_lower)) for c in candidates]
    rows.sort(key=lambda x: -x[1])
    n_keep = round(len(candidates) * percentage)
    return sorted(c for c, _ in rows[:n_keep])


def run_reproduction_from_scratch(texts_lower: list, medication_candidates: list,
                                   condition_candidates: list, current_terms: set):
    """Computes word-enrichment scores directly from training notes instead of
    reading a pre-computed file, and selects which Step-1 candidates to keep
    via a computed top-40%-by-frequency cutoff (see select_top_percentage())
    instead of a hardcoded top-N count. Writes a separate, clearly-labeled
    demonstration output; does not touch psych_keywords_final.py."""
    print("\n" + "=" * 70)
    print("REPRODUCE-FROM-SCRATCH: selecting top 40% of candidates by frequency")
    print("=" * 70)
    kept_medications = select_top_percentage(medication_candidates, texts_lower, percentage=0.40)
    kept_conditions = select_top_percentage(condition_candidates, texts_lower, percentage=0.40)
    print(f"Medications: kept top {len(kept_medications)}/{len(medication_candidates)} "
          f"candidates (40% of the candidate pool, by training-set frequency).")
    print(f"Conditions: kept top {len(kept_conditions)}/{len(condition_candidates)} "
          f"candidates (40% of the candidate pool, by training-set frequency).")

    print("\n" + "=" * 70)
    print("REPRODUCE-FROM-SCRATCH: computing word-enrichment scores")
    print("=" * 70)
    indicator_df = compute_enrichment_scores(texts_lower, current_terms)
    indicator_df.to_csv(OUT_DIR / 'indicator_word_scores_reproduced.csv', index=False)
    print(f"Scored {len(indicator_df)} distinct words by enrichment near known clinical content.")

    additional_terms = select_additional_terms(indicator_df, current_terms)
    print(f"Selected {len(additional_terms)} additional terms "
          f"(enrichment ratio >= {RATIO_THRESHOLD}, excluding noise-category terms).")

    demo_terms = sorted(
        set(kept_medications) | set(kept_conditions) | set(additional_terms)
    )
    with open(OUT_DIR / 'psych_keywords_reproduced_demo.py', 'w') as f:
        f.write("# Standalone reproducibility demonstration. Generated entirely from the\n")
        f.write("# reference_data/*.csv candidate files plus training-set word counts -- no\n")
        f.write("# pre-computed intermediate files were read, and no GPU/model was used. Not\n")
        f.write("# used to produce any reported result; see module docstring.\n")
        f.write("CANDIDATE_TERMS_REPRODUCED = [\n")
        for i in range(0, len(demo_terms), 6):
            f.write("    " + ", ".join(f"'{c}'" for c in demo_terms[i:i + 6]) + ",\n")
        f.write("]\n")

    with open(OUT_DIR / 'reproduction_summary.json', 'w') as f:
        json.dump({
            'n_medication_candidates': len(medication_candidates),
            'n_medications_kept_top40pct': len(kept_medications),
            'n_condition_candidates': len(condition_candidates),
            'n_conditions_kept_top40pct': len(kept_conditions),
            'n_words_scored_for_enrichment': len(indicator_df),
            'n_additional_terms_selected': len(additional_terms),
            'n_demo_terms_total': len(demo_terms),
        }, f, indent=2)

    print(f"\nSaved: {OUT_DIR / 'psych_keywords_reproduced_demo.py'}")
    print(f"Saved: {OUT_DIR / 'reproduction_summary.json'}")


def main():
    print("=" * 70)
    print("PSYCH_KEYWORDS RECONSTRUCTION")
    print("=" * 70)

    train = pd.read_csv(TRAIN_DATA_PATH)
    texts_lower = train['text'].dropna().str.lower().tolist()
    texts_raw = train['text'].dropna().tolist()
    print(f"\nTraining set: {len(texts_lower)} notes")

    if _args.reproduce_from_scratch:
        reproduced_medication_candidates = pd.read_csv(REPRODUCED_MEDICATION_CANDIDATES_PATH)['term'].tolist()
        reproduced_condition_candidates = pd.read_csv(REPRODUCED_CONDITION_CANDIDATES_PATH)['term'].tolist()
        current_terms = (set(reproduced_medication_candidates) | set(reproduced_condition_candidates)
                          | set(FIXED_CONDITION_INDICATORS) | set(FIXED_ABBREVIATIONS))
        run_reproduction_from_scratch(texts_lower, reproduced_medication_candidates,
                                       reproduced_condition_candidates, current_terms)
        print("\n" + "=" * 70)
        return 0

    # --- Medications ---
    med_rows = [(m, word_boundary_count(m, texts_lower)) for m in sorted(set(PSYCHDB_MEDICATIONS))]
    med_rows.sort(key=lambda x: -x[1])
    with open(OUT_DIR / 'medication_frequency.csv', 'w') as f:
        f.write('medication,n_notes_containing\n')
        for m, c in med_rows:
            f.write(f'{m},{c}\n')
    final_medications = sorted([m for m, c in med_rows[:MED_TOP_N]])
    print(f"\nMedications: {len(med_rows)} candidates from psychdb.com/meds, "
          f"top {len(final_medications)} by training-set frequency selected "
          f"(cutoff: {med_rows[MED_TOP_N-1][1]} notes)")

    # --- Conditions ---
    with open(CONDITION_CANDIDATES_PATH) as f:
        condition_candidates = [l.strip() for l in f if l.strip()]
    cond_rows = [(t, word_boundary_count(t, texts_lower)) for t in condition_candidates]
    cond_rows.sort(key=lambda x: -x[1])
    with open(OUT_DIR / 'condition_frequency_final.csv', 'w') as f:
        f.write('term,n_notes_containing\n')
        for t, c in cond_rows:
            f.write(f'"{t}",{c}\n')
    final_conditions = sorted(set([t for t, c in cond_rows[:COND_TOP_N]]) | set(FIXED_CONDITION_INDICATORS))
    print(f"Conditions: {len(cond_rows)} candidates from {CONDITION_CANDIDATES_PATH}, "
          f"top {COND_TOP_N} by training-set frequency selected "
          f"(cutoff: {cond_rows[COND_TOP_N-1][1]} notes), "
          f"plus {len(FIXED_CONDITION_INDICATORS)} fixed generic indicators, "
          f"{len(final_conditions)} total after dedup")

    # --- Section headers ---
    header_pattern = re.compile(r'^([A-Za-z][A-Za-z /\-]{2,40}):\s*$', re.MULTILINE)
    header_counter = Counter()
    for t in texts_raw:
        for m in header_pattern.finditer(t):
            header_counter[m.group(1).strip().lower()] += 1
    relevance_terms = ['diagnos', 'medication', 'mental status', 'psychiatric', 'social history', 'substance']
    relevant_headers = {
        h: c for h, c in header_counter.items() if any(rt in h for rt in relevance_terms)
    }
    relevant_ranked = sorted(relevant_headers.items(), key=lambda x: -x[1])
    with open(OUT_DIR / 'header_frequency.csv', 'w') as f:
        f.write('header,n_occurrences,relevant\n')
        for h, c in header_counter.most_common():
            f.write(f'"{h}",{c},{int(h in relevant_headers)}\n')
    final_headers = sorted(h for h, c in relevant_ranked[:HEADER_TOP_N])
    print(f"Section headers: {len(header_counter)} distinct header lines found, "
          f"{len(relevant_headers)} relevant to diagnosis/medication/psychiatric content, "
          f"top {len(final_headers)} by frequency selected "
          f"(cutoff: {relevant_ranked[HEADER_TOP_N-1][1]} occurrences)")

    # --- F-codes (reuse the 90%-coverage result from script 12) ---
    freq_df = pd.read_csv('outputs/12_PsychKeywordsReproducibility/fcode_frequency_train.csv')
    freq_df = freq_df.sort_values('rank')
    cutoff_rank = int(freq_df[freq_df['cumulative_pct'] >= 90.0]['rank'].iloc[0])
    final_fcodes = sorted(freq_df[freq_df['rank'] <= cutoff_rank]['f_code'].tolist())
    print(f"F-codes: {len(final_fcodes)} at 90% training-set cumulative coverage (rank {cutoff_rank})")

    # --- Step 3: additional terms by sentence-level enrichment ---
    # Sentences are marked "diagnostic" using the full Step-1 candidate pools
    # (not just the terms that survived Step 2's frequency filter), so a
    # candidate that was filtered out in Step 2 can still mark a sentence as
    # clinically relevant for this enrichment comparison.
    diagnostic_terms = set(condition_candidates) | set(m.lower() for m in PSYCHDB_MEDICATIONS)
    exclude_from_additions = diagnostic_terms | set(FIXED_CONDITION_INDICATORS)
    indicator_df = compute_enrichment_scores(texts_lower, diagnostic_terms)
    indicator_df.to_csv(OUT_DIR / 'indicator_word_scores.csv', index=False)
    additional_terms = select_additional_terms(indicator_df, exclude_from_additions)
    medication_additions = sorted(t for t in additional_terms if t in BRAND_INCLUDE)
    condition_additions = sorted(t for t in additional_terms if t not in BRAND_INCLUDE)
    final_medications = sorted(set(final_medications) | set(medication_additions))
    final_conditions = sorted(set(final_conditions) | set(condition_additions))
    print(f"Step 3: {len(additional_terms)} additional terms by sentence-level enrichment ratio "
          f"(ratio >= {RATIO_THRESHOLD}, frequency >= {FREQ_THRESHOLD}, noise-category terms excluded) -- "
          f"{len(medication_additions)} brand-name medications, {len(condition_additions)} conditions")

    # --- Assemble final list ---
    all_terms = (sorted(final_fcodes) + sorted(FIXED_ABBREVIATIONS)
                 + sorted(final_conditions) + sorted(final_medications) + sorted(final_headers))

    def py_literal(term: str) -> str:
        return f'"{term}"' if "'" in term else f"'{term}'"

    def py_block(label: str, terms: list, per_line: int) -> list:
        lines = ["    # " + label]
        for i in range(0, len(terms), per_line):
            lines.append("    " + ", ".join(py_literal(t) for t in terms[i:i+per_line]) + ",")
        return lines

    py_lines = ["PSYCH_KEYWORDS = ["]
    py_lines += py_block("F-codes (90% cumulative training-set coverage)", sorted(final_fcodes), 8)
    py_lines += py_block("Abbreviations (fixed)", sorted(FIXED_ABBREVIATIONS), 8)
    py_lines += py_block("Conditions (frequency-ranked ICD-10-CM terms + statistically-enriched additions)",
                          final_conditions, 6)
    py_lines += py_block("Medications (frequency-ranked generic names + enriched brand names)",
                          final_medications, 6)
    py_lines += py_block("Section headers (relevant, top %d by frequency)" % HEADER_TOP_N, final_headers, 4)
    py_lines.append("]")

    with open(OUT_DIR / 'psych_keywords_final.py', 'w') as f:
        f.write("\n".join(py_lines) + "\n")

    summary = {
        'n_fcodes': len(final_fcodes),
        'n_abbreviations': len(FIXED_ABBREVIATIONS),
        'n_conditions': len(final_conditions),
        'n_medications': len(final_medications),
        'n_headers': len(final_headers),
        'total_terms': len(set(all_terms)),
        'med_top_n': MED_TOP_N,
        'cond_top_n': COND_TOP_N,
        'header_top_n': HEADER_TOP_N,
        'additional_terms_ratio_threshold': RATIO_THRESHOLD,
        'additional_terms_freq_threshold': FREQ_THRESHOLD,
        'n_medication_additions': len(medication_additions),
        'n_condition_additions': len(condition_additions),
    }
    with open(OUT_DIR / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"\nSaved: {OUT_DIR / 'psych_keywords_final.py'}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    exit(main())
