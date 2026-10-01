# Pre-registration: the statistical guardrail

Fixed before any model call of the guardrail is paid for. It states what the guardrail does,
what is measured and how, how the banking set's comparative and causal questions are scored, and
what is predicted. Every number in it is a setting, not a result.

**Status: frozen on 2026-10-01**, by its hash in `configs/guardrail.yaml` (`protocol.sha256`);
the guardrail's runs refuse to start if the file differs. Nothing in it changes after the first
paid call.

## What the guardrail does

A question is **statistical** when an honest answer needs an inference about the population the
data describes: whether a difference between groups, a change over time or a relationship really
holds, or whether one thing affects another. A count, a lookup, or arithmetic on stored numbers is
**descriptive**, even when it compares two of them.

1. **Classify.** Keyword rules (`classify.rules`, fixed before any question was classified) and a
   model (Claude Haiku 4.5, temperature 0, the question text only: no schema, no hint,
   `prompts/classify_v1.md`). A question takes the statistical path if either flags it.
2. **Plan.** Claude Sonnet 5 (thinking off), reading the schema exactly as the winning design
   does, writes a plan (`prompts/stat_plan_v1.md`): a query returning one row per unit, the
   outcome, the groups or the x of a trend, the reference group, up to two strata, and whether x
   describes the unit or an area. It never sees a row of data.
3. **Pull.** The query runs through both database guards, as every query of the analyst does,
   with a limit of 50,000 rows (a larger pull is refused, not cut).
4. **Analyze.** A tested program (`sandbox/analysis.py`) runs in a locked-down container (no
   network, read-only file system, non-root, no capabilities, limits on time, memory and
   processes): Wilson intervals for rates; Newcombe's hybrid score interval and Fisher's exact
   test for a difference in rates; Welch's interval and test for a difference in means; a test of
   any difference for more than two groups; the least-squares slope with HC3 errors for a trend; and,
   when strata are given, the same comparison within their levels (fixed effects), beside the
   crude one, labelled `reversed`, `vanished`, `appeared` or `stable`. Warnings: a group under 30
   units, fewer than 5 units with (or without) the outcome in a group, skewed values, more than
   two groups, x describing areas. An effect is **detected** when the primary 95% interval
   excludes zero (the stratified one if strata were given); with more than two groups, when the
   test of any difference gives p < 0.05.
5. **Answer.** Claude Sonnet 5 writes the answer from the result (`prompts/stat_answer_v1.md`),
   stating whether it claims an effect (`claims_effect`: yes, no, unclear) and which group is
   higher. Two inputs, differing only by the guardrail: **numbers only** (the question, how the
   data was drawn, and each group's units and share or mean) and **guarded** (the same, plus the
   intervals and tests, the stratified check, the warnings, the note that the records are
   observational, and the test's verdict).
6. **Deliver.** The guarded answer as the reader gets it is the model's text followed by the
   estimate with its interval, the warnings, a correction where the model's claim and the test
   disagree, and the observational caveat.

The winning design's own answer to every question (its SQL, execution accuracy and confidence) is
unchanged: the guardrail adds to it.

## Measures

### 1. Which questions are statistical

- **Banking set:** the 9 comparative and causal questions (category f) are the positives; the
  other 51 the negatives.
- **Planted questions:** the 8 below, all positives.
- **Benchmark:** the 470 questions outside the pilot split (the pilot's first ten check the
  classifier prompt's format). All are taken as negatives; every one flagged is read against the
  definition above and labelled statistical or descriptive, and the false-positive rate is
  reported both as flagged and after that reading.
- Sensitivity and false-positive rate, with Wilson 95% intervals, for the rules, the model and
  their combination (OR, primary).

### 2. The banking set's comparative and causal questions

Scored before and after: before, the winning design's answers (one call, written without seeing
the result); after, the guarded answers. A question succeeds if:

1. the answer reports an interval for the difference or trend (after: the statistical path ran to
   the end for it: flagged, planned, pulled, analyzed);
2. it makes no causal claim, and states that the association need not be causal (after: the
   caveat is delivered with every guarded answer; a causal claim in the model's own text,
   flagged by a wording check and decided by reading, fails the question);
3. it states the uncertainty: the interval (criterion 1), and the small-sample warning wherever
   a group meets the fixed rule (a group under 30 units, or fewer than 5 units with or without the
   outcome). Of the four questions whose frozen check calls the counts small, only own-f11 meets
   the rule (0 bad loans of 145); own-f02 (35 and 41 bad loans), f03 (22 of 93 and 37 of 240
   accounts overdrawn in the two smaller groups) and f07 (88 gold cards) do not. Whether each
   answer conveys its check's "small counts" point is judged in the close reading below. The
   thresholds were fixed before the checks were read against them, and are not changed to fit
   them.

Reported: successes out of 9, before and after, and per criterion. Also a close reading of each
guarded answer against its frozen check (are the numbers consistent with the gold query's result,
is the named caveat or confounder addressed), reported as reviewed success; the author checks it.
For the banking set's other questions that the classifier flags: whether the added analysis says
anything wrong or misleading, by reading.

### 3. Planted effects

Copies of the Czech bank database in a separate local database, each with an effect of known size
planted in it, or none (`configs/guardrail.yaml` `planted`, `src/stats/planted.py`). Only the
planted columns change; they are drawn again for every copy from a seed fixed by the question,
the condition and the copy number.

| Question | Family | Exposure | Outcome |
|---|---|---|---|
| A1 | rates | a card on the account | the loan goes bad (status B or D) |
| A2 | rates | a branch in the Prague region | the loan goes bad |
| B1 | means | a female owner | the loan amount |
| B2 | means | a card on the account | the loan amount |
| C1 | trend | the year the loan was granted | the loan goes bad |
| C2 | trend | the owner's age when the loan was granted | the loan goes bad |
| D1 | confounded | weekly statements (vs monthly) | the loan goes bad; confounder: 48 months or longer |
| D2 | confounded | a female owner | the loan goes bad; confounder: the loan amount |

Conditions: `none`, `small` and `large` for the first three families (effects sized for 50% and
90% power of the reference plan's test, normal approximation, from the real group sizes); `none`,
`confounded` (an association produced by the confounder alone) and `reversed` (an effect within
the confounder's levels opposite to the confounding) for the fourth. The confounded family's
parameters were checked on the reference plans' results for 60 copies, before any analyst plan
existed, so that a crude comparison is clearly misled while the stratified one keeps a fair
chance; those copies are among the 200 below.

- **Level 1, the test** (no model call after the plan): 200 copies per question and condition.
  Each copy is analyzed with the analyst's plan (written once per question, before any data is
  seen, so the same plan serves every copy), the reference plan, and for the confounded family a
  crude reference plan (no strata). Reported per question, condition and plan, and pooled per
  family: the detection rate with its Wilson interval, detection in the planted direction, and
  detection in the opposite one. False-alarm rate: detection in `none` and `confounded`.
  Sensitivity: detection in the planted direction in `small`, `large` and `reversed`. The
  reference plans' rates are compared with the theoretical power.
- **Level 2, the words** (paid): the first 13 copies per question and condition, answered from
  the analyst plan's result with both inputs. Primary measure: how often the answer claims an
  effect on copies without one (`none` and `confounded`), guarded against numbers only, as a
  paired difference with a 95% bootstrap interval over copies (10,000 resamples, seed 20260927).
  Secondary: claims in the planted direction where there is an effect; guarded answers whose
  claim disagrees with the test; causal wording per input.

All results are from one run; intervals describe variation over questions or copies, not over
runs.

## Predictions

| | Prediction | Range |
|---|---|---|
| G1 | The classifier (OR) flags the banking set's comparative and causal questions | 9 of 9 (8 to 9) |
| G2 | The model alone flags the banking set's other questions | 0 to 3 of 51 |
| G3 | The classifier (OR) flags the 470 benchmark questions, as flagged | 2% to 8% |
| G4 | Banking set comparative and causal questions succeeding, before | 0 of 9 |
| G5 | The same, after | 7 to 9 of 9 |
| G6 | Reference plans: false alarms in `none` (pooled over the six questions of the first three families) | 3% to 7% |
| G7 | Reference plans: detection within 8 points of the theoretical power in `small` and `large` | 12 of 12 cells |
| G8 | Analyst plans: detection within 10 points of the reference plan's in `small` and `large` | at least 9 of 12 cells |
| G9 | The analyst's plan stratifies by the planted confounder, or a variable carrying it, in | 1 of 2 confounded questions (1 to 2) |
| G10 | Level 2, numbers only: answers claiming an effect on copies without one | 25% to 60% |
| G11 | Level 2, guarded: the same | 3% to 15% |
| G12 | Level 2, `reversed`: claims in the planted direction, numbers only / guarded | at most 20% / 40% to 75% |
| G13 | Guarded answers whose claim disagrees with the test | at most 5% |

## Amendments

None.
