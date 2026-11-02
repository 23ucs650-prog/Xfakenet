# Label mapping (Week 1)

**Status:** fixed 20 Oct 2026. Source of truth: `configs/label_mapping.yaml`.

## Output classes

| id | Class | Meaning |
|---|---|---|
| 0 | Fabricated | Claim is false; no factual basis |
| 1 | Misleading | Contains truth but distorts, omits context or exaggerates |
| 2 | Credible | Accurate or substantially accurate |

## LIAR → XFakeNet

| LIAR rating | PolitiFact definition (short) | XFakeNet |
|---|---|---|
| pants-fire | false and ridiculous | Fabricated |
| false | not accurate | Fabricated |
| barely-true | element of truth, ignores critical facts | Misleading |
| half-true | partially accurate, leaves out details | Misleading |
| mostly-true | accurate but needs clarification | Credible |
| true | accurate, nothing significant missing | Credible |

**Rationale.** The split follows PolitiFact's own definitions: the two ratings
with no factual basis become Fabricated, the two that contain truth but
distort it become Misleading, and the two judged accurate become Credible.

**Known limitation.** Neighbouring LIAR ratings (e.g. false vs barely-true) are
hard to separate even for humans, so the class boundaries are noisy.

## Why Satire and Propaganda were dropped

The original plan had five classes. LIAR has no Satire or Propaganda examples,
so these were taken from TSHP-17 (Rashkin et al., 2017). In the first baseline
run, Propaganda scored F1 0.99 on validation but 0.08 on test. The TSHP-17 test
set uses different news websites from training, so the model had learned each
website's writing style rather than propaganda itself; most test propaganda
was labelled Satire. With no reliable source for these two classes in the
time available, the classifier uses three classes.

| Run | Accuracy | Macro-F1 | Note |
|---|---|---|---|
| Five-class (LIAR + TSHP-17) | 0.482 | 0.434 | Propaganda F1 0.08 on test |
| Three-class (LIAR only) | 0.482 | 0.470 | Adopted |

## FakeNewsNet

Binary fake/real → Fabricated/Credible. Used for C-score calibration and
evaluation only, not for classifier training.
