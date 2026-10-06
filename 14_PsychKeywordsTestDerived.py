#!/usr/bin/env python3
"""
14_PsychKeywordsTestDerived.py - Test-Set-Derived PSYCH_KEYWORDS (data-leakage check)

Rebuilds the PSYCH_KEYWORDS list using the exact same procedure as
13_PsychKeywordsConstruction.py, but with TEST-set frequency counts
substituted for training-set frequency counts throughout. This is the
worst-case leakage scenario: if building the keyword list from the test
set gave any real advantage, it would show up here, both as a different
term list and as a different downstream Micro F1 when evaluated.

Output:
    outputs/14_PsychKeywordsTestDerived/psych_keywords_test_derived.py
    outputs/14_PsychKeywordsTestDerived/summary.json
    outputs/14_PsychKeywordsTestDerived/overlap_with_train_derived.json

Usage:
    python 14_PsychKeywordsTestDerived.py
"""

import importlib.util
import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd

from config import RANDOM_SEED, set_all_seeds, TEST_DATA_PATH

set_all_seeds(RANDOM_SEED)

OUT_DIR = Path('outputs/14_PsychKeywordsTestDerived')
OUT_DIR.mkdir(parents=True, exist_ok=True)

MED_TOP_N = 45
COND_TOP_N = 56
HEADER_TOP_N = 16
RATIO_THRESHOLD = 3.5
FREQ_THRESHOLD = 150  # scaled down from 500 by the train/test size ratio (~39848/8539 ~= 4.67)

FIXED_ABBREVIATIONS = ['ADHD', 'GAD', 'MDD', 'OCD', 'PTSD']

# Same candidate sources as script 13 (external references, same for both splits).
CONDITION_CANDIDATES_PATH = 'outputs/12_PsychKeywordsReproducibility/condition_candidates_stage1.txt'

spec = importlib.util.spec_from_file_location("psych13", "13_PsychKeywordsConstruction.py")
psych13 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(psych13)
PSYCHDB_MEDICATIONS = psych13.PSYCHDB_MEDICATIONS


def word_boundary_count(term: str, texts_lower) -> int:
    pattern = re.compile(r'\b' + re.escape(term.lower()) + r'\b')
    return sum(1 for t in texts_lower if pattern.search(t))


def main():
    print("=" * 70)
    print("TEST-SET-DERIVED PSYCH_KEYWORDS (data-leakage check)")
    print("=" * 70)

    test = pd.read_csv(TEST_DATA_PATH)
    texts_lower = test['text'].dropna().str.lower().tolist()
    texts_raw = test['text'].dropna().tolist()
    print(f"\nTest set: {len(texts_lower)} notes")

    # --- F-codes: 90% cumulative coverage on TEST set ---
    def format_f_code(code):
        code = code.strip()
        if len(code) > 3 and '.' not in code:
            return f"{code[:3]}.{code[3:]}"
        return code

    test_df = test[test['f_codes_str'].notna()]
    all_fcodes = []
    for codes_str in test_df['f_codes_str']:
        all_fcodes.extend(format_f_code(c) for c in str(codes_str).split(','))
    fcounter = Counter(all_fcodes)
    total = sum(fcounter.values())
    cum = 0
    final_fcodes = []
    for code, cnt in fcounter.most_common():
        cum += cnt
        final_fcodes.append(code)
        if cum / total * 100 >= 90.0:
            break
    final_fcodes = sorted(final_fcodes)
    print(f"F-codes: {len(final_fcodes)} at 90% test-set cumulative coverage")

    # --- Medications ---
    med_rows = [(m, word_boundary_count(m, texts_lower)) for m in sorted(set(PSYCHDB_MEDICATIONS))]
    med_rows.sort(key=lambda x: -x[1])
    final_medications = sorted([m for m, c in med_rows[:MED_TOP_N]])
    print(f"Medications: top {len(final_medications)} by test-set frequency "
          f"(cutoff: {med_rows[MED_TOP_N-1][1]} notes)")

    # --- Conditions ---
    with open(CONDITION_CANDIDATES_PATH) as f:
        condition_candidates = [l.strip() for l in f if l.strip()]
    cond_rows = [(t, word_boundary_count(t, texts_lower)) for t in condition_candidates]
    cond_rows.sort(key=lambda x: -x[1])
    final_conditions_base = sorted([t for t, c in cond_rows[:COND_TOP_N]])
    print(f"Conditions (ICD base): top {len(final_conditions_base)} by test-set frequency "
          f"(cutoff: {cond_rows[COND_TOP_N-1][1]} notes)")

    # --- Section headers ---
    header_pattern = re.compile(r'^([A-Za-z][A-Za-z /\-]{2,40}):\s*$', re.MULTILINE)
    header_counter = Counter()
    for t in texts_raw:
        for m in header_pattern.finditer(t):
            header_counter[m.group(1).strip().lower()] += 1
    relevance_terms = ['diagnos', 'medication', 'mental status', 'psychiatric', 'social history', 'substance']
    relevant_headers = {h: c for h, c in header_counter.items() if any(rt in h for rt in relevance_terms)}
    relevant_ranked = sorted(relevant_headers.items(), key=lambda x: -x[1])
    final_headers = sorted(h for h, c in relevant_ranked[:HEADER_TOP_N])
    print(f"Section headers: top {len(final_headers)} by test-set frequency")

    # --- Statistical enrichment analysis on TEST set (brand names + symptom terms) ---
    diagnostic_terms = set(condition_candidates) | set(m.lower() for m in PSYCHDB_MEDICATIONS)
    diag_pattern = re.compile(r'\b(' + '|'.join(re.escape(t) for t in diagnostic_terms) + r')\b')
    STOPWORDS = set("""a an the and or but if is was were be been being to of in on at for with
    without by from as it its this that these those he she they we you i his her their our your
    not no yes do does did have has had will would can could should may might shall must
    mg po qd bid tid qid prn year old male female patient pt s p also noted well into than
    """.split())

    diag_word_counts = Counter()
    other_word_counts = Counter()
    word_pattern = re.compile(r"[a-z']+")
    for text in texts_raw:
        sentences = re.split(r'(?<=[.!?])\s+|\n+', text.lower())
        for sent in sentences:
            words = word_pattern.findall(sent)
            is_diag = bool(diag_pattern.search(sent))
            target = diag_word_counts if is_diag else other_word_counts
            target.update(w for w in words if len(w) >= 3 and w not in STOPWORDS)

    total_diag = sum(diag_word_counts.values())
    total_other = sum(other_word_counts.values())

    scores = []
    for w, c_diag in diag_word_counts.items():
        total_w = c_diag + other_word_counts.get(w, 0)
        if total_w < FREQ_THRESHOLD or w in diagnostic_terms:
            continue
        c_other = other_word_counts.get(w, 1)
        ratio = (c_diag / total_diag) / (c_other / total_other) if c_other > 0 and total_other > 0 else float('inf')
        scores.append((w, total_w, ratio))
    scores.sort(key=lambda x: -x[2])

    # Apply the SAME principled noise-exclusion categories as script 13 (fixed
    # categories, not re-derived per split, so this stays a like-for-like
    # comparison with the train-derived list).
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
    EXCLUDE = (DOSING_SCHEDULE | CHEMICAL_SALT_FORM | ROUTE_FORMULATION | ADMIN_CHART_ABBREV |
               UNRELATED_CONDITION | VAGUE_GENERIC | GENERIC_MED_PROCESS_VERBS |
               VAGUE_NONDIAGNOSTIC | SIDE_EFFECT_TANGENTIAL | NOISE_ARTIFACTS)

    enrichment_additions = sorted(w for w, t, r in scores if w not in EXCLUDE and r >= RATIO_THRESHOLD)
    print(f"Enrichment-derived additions: {len(enrichment_additions)} words "
          f"(ratio >= {RATIO_THRESHOLD}, freq >= {FREQ_THRESHOLD} on test set)")

    # Brand-name medications (known list, same as script 13) vs condition additions
    BRAND_INCLUDE = {'prozac', 'klonopin', 'neurontin', 'depakote', 'ativan', 'haldol',
                     'zyprexa', 'keppra', 'valium', 'dilantin', 'suboxone', 'topamax',
                     'ritalin', 'latuda', 'buproprion', 'narcan', 'lyrica', 'celexa',
                     'wellbutrin', 'effexor', 'ambien', 'cymbalta', 'abilify', 'adderall',
                     'lamictal', 'lexapro', 'risperdal', 'xanax', 'seroquel', 'zoloft',
                     'benadryl'}
    brand_additions = sorted(w for w in enrichment_additions if w in BRAND_INCLUDE)
    condition_additions = sorted(w for w in enrichment_additions if w not in BRAND_INCLUDE)

    final_conditions = sorted(set(final_conditions_base) | set(condition_additions))
    final_medications_full = sorted(set(final_medications) | set(brand_additions))

    all_counts = {
        'n_fcodes': len(final_fcodes), 'n_abbreviations': len(FIXED_ABBREVIATIONS),
        'n_conditions': len(final_conditions), 'n_medications': len(final_medications_full),
        'n_headers': len(final_headers),
    }
    all_counts['total'] = sum(all_counts.values())
    print("\nFinal test-derived list:", all_counts)

    def fmt_block(label, terms, per_line):
        lines = [f"    # {label}"]
        for i in range(0, len(terms), per_line):
            chunk = terms[i:i+per_line]
            rendered = ", ".join((f'"{t}"' if "'" in t else f"'{t}'") for t in chunk)
            lines.append("    " + rendered + ",")
        return "\n".join(lines)

    block = "PSYCH_KEYWORDS = [\n"
    block += fmt_block("F-codes (90% cumulative test-set coverage)", final_fcodes, 8) + "\n"
    block += fmt_block("Abbreviations (fixed)", FIXED_ABBREVIATIONS, 8) + "\n"
    block += fmt_block("Conditions (test-set-derived)", final_conditions, 6) + "\n"
    block += fmt_block("Medications (test-set-derived)", final_medications_full, 6) + "\n"
    block += fmt_block("Section headers (test-set-derived)", final_headers, 4) + "\n"
    block = block.rstrip(",\n") + "\n]"

    with open(OUT_DIR / 'psych_keywords_test_derived.py', 'w') as f:
        f.write(block + "\n")

    with open(OUT_DIR / 'summary.json', 'w') as f:
        json.dump(all_counts, f, indent=2)

    # --- Overlap with the train-derived list ---
    spec = importlib.util.spec_from_file_location(
        'psych_keywords_final', 'outputs/13_PsychKeywordsConstruction/psych_keywords_final.py'
    )
    psych_keywords_final = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(psych_keywords_final)
    train_derived = set(x.lower() for x in psych_keywords_final.PSYCH_KEYWORDS)
    test_derived = set(x.lower() for x in (final_fcodes + FIXED_ABBREVIATIONS + final_conditions +
                                            final_medications_full + final_headers))
    overlap = train_derived & test_derived
    overlap_summary = {
        'train_derived_size': len(train_derived),
        'test_derived_size': len(test_derived),
        'overlap': len(overlap),
        'overlap_pct_of_train': round(len(overlap) / len(train_derived) * 100, 1),
        'only_in_train_derived': sorted(train_derived - test_derived),
        'only_in_test_derived': sorted(test_derived - train_derived),
    }
    with open(OUT_DIR / 'overlap_with_train_derived.json', 'w') as f:
        json.dump(overlap_summary, f, indent=2)

    print(f"\nOverlap with train-derived list: {overlap_summary['overlap']}/{overlap_summary['train_derived_size']} "
          f"({overlap_summary['overlap_pct_of_train']}%)")
    print(f"Only in train-derived: {len(overlap_summary['only_in_train_derived'])} terms")
    print(f"Only in test-derived: {len(overlap_summary['only_in_test_derived'])} terms")

    print("\n" + "=" * 70)
    return 0


if __name__ == "__main__":
    exit(main())
