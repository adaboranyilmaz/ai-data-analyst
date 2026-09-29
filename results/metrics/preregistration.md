# Pre-registration

Fixed before any agent design is run on a scored question. It states what will be compared,
on which questions, how the winner is chosen, how confidence is calibrated and used, and what
is predicted. Every number in it is a setting, not a result.

**Status: frozen on 2026-09-28**, by its hash in `configs/eval.yaml` (`preregistration`).
Until the first scored run of the design comparison it may be amended; each amendment is
listed, dated, at the end, and the hash updated. After that run nothing in it changes, and the
analysis (`src/eval/preregistration.py`) refuses to run if it has, if any prediction is blank,
or if the question sets differ from the ones named below.

## Questions

- **Benchmark:** BIRD mini-dev, PostgreSQL, 500 questions over 11 databases, with the question
  file pinned by hash. Split once, before any run, into three disjoint sets drawn in proportion
  to every database x difficulty stratum (`results/metrics/splits.json`, seed 20260927):
  - **pilot**, 30 questions: prompts are written and tuned on these; they appear in no
    reported comparison;
  - **ablation**, 150 questions: the five designs are compared and the winner chosen here; also
    the calibration split, where confidence is calibrated and thresholds are chosen;
  - **held-out**, 320 questions: the test split. Nothing is chosen or fitted on it; the
    headline accuracy, calibration and decline results are reported on it.
- **Hand-written banking set** (`own_set/questions.yaml`): questions on the Czech bank
  database in six categories (standard, multi-step, ambiguous, unanswerable, false premise,
  comparative or causal), reviewed and frozen by hash before any agent sees them. Its gold
  never enters a prompt.
- **Evidence:** the design comparison, and the winner's run on all 500, use BIRD's evidence
  hint. The winner is also run on the ablation set without it, with the database descriptions
  BIRD ships in its place.

## The five designs

Each design adds one step to the one before. Every design ends with the same structured
answer: the final SQL, the answer in words, a confidence between 0 and 1, whether it declines
(and why), and where they apply a clarifying question, the assumptions made, and a correction
of a false premise. The answer in words gives coded values in plain English from the data
dictionary, with the code itself in brackets where the answer rests on it (for example,
"statement fees (SLUZBY)"), so a reader can match it to the SQL; the SQL and its result keep the
values as stored, since they are what is scored. SQL runs only through the guarded read-only tools (`run_sql`: the query
guard and the read-only database role). Prompt wording is fixed on the pilot set; the steps and
budgets below are fixed here.

1. **Single-shot from the full schema.** One model call. The prompt holds the question and
   the full schema of its database: every table and column with its type and its data
   dictionary description. No tools. The reply's SQL is executed once; a refusal or an error
   is the answer's error. Confidence is the one the model states.
2. **+ schema tools.** The prompt holds the question and the list of tables. The model may call
   `list_tables`, `describe_table` and `sample_rows` (not `run_sql`), at most 10 calls, then
   submits its answer; its SQL is executed once, as in design 1. Confidence as stated.
3. **+ self-correction.** Design 2 with `run_sql` available during the loop (at most 15 tool
   calls). When the submitted SQL is refused, fails, times out or returns no rows, that outcome
   goes back to the model, which may submit again, at most twice. Confidence as stated.
4. **+ verification.** Design 3 run as k = 3 independent samples (the sample index is part of
   the response cache's key). The samples' result sets are compared as the benchmark compares
   them (sets of rows); the answer is that of the largest group of agreeing samples (on a tie,
   the earliest sample's). Automatic checks on it: an empty result; repeated identical rows;
   a value outside its possible range (a negative count or amount, a share above 100%); more
   than 1,000 rows. Confidence: the share of samples that agree with the answer (1/3, 2/3 or
   1), halved if any check fails.
5. **+ schema narrowing.** Before design 4's samples, one model call picks the tables and
   columns relevant to the question from the full schema; the samples then see, and may
   explore and query, only those.

**Models.** Claude Sonnet 5 on all five designs; Claude Haiku 4.5 on designs 1, 3 and the
winner; the local model on designs 1-3. Settings that differ between models (sampling
parameters, effort) are recorded per run. Each design is run once per question, apart from
design 4's and 5's three samples.

## Choosing the winner

On the ablation set, with evidence, Claude Sonnet 5: the design with the lowest AURC (the area
under the risk-coverage curve, declines ranked last and counted wrong) is the best. A design
whose mean cost per question is lower than the best's, and whose AURC minus the best's has a
paired 95% bootstrap interval containing zero, is not shown to be worse; if there are such
designs, the cheapest of them is the winner, otherwise the best is (`src/eval/summary.py`,
`select_design`). Every pairwise comparison is reported; no correction for multiple comparisons
is applied, because this rule is the only decision taken on them.

## Confidence, declining and escalation

- **Calibration:** Platt scaling (a logistic regression of correctness on the raw confidence)
  is fitted on the winner's run on the calibration split; isotonic regression is reported
  beside it as a secondary calibrator. Calibration (ECE, Brier score, reliability) is reported
  on the held-out set only.
- **Declining:** the decline threshold is the calibrated confidence below which the agent
  declines, chosen on the calibration split as the one giving the largest coverage at which
  the accuracy of the answered questions is at least 90%. On the held-out set the achieved
  accuracy and coverage are reported with intervals, whether or not they reach 90%.
- **Critic:** a second model reads the question, the answer and its evidence (SQL, result
  preview, checks) and gives its own confidence. It is compared with the winner's calibrated
  confidence by AURC on the held-out set, paired.
- **Escalation to Claude Opus 5.5:** the winner design with Opus 5.5 (its effort level
  recorded) on the ablation set, paired with Sonnet 5's run. Adopted only if its EX is higher
  by at least 5 points with a paired 95% interval excluding zero; its cost per correct answer
  is reported beside the decision. If adopted, a router sends to Opus 5.5 the questions whose
  calibrated Sonnet 5 confidence is below the decline threshold, and is evaluated on the
  held-out set.
- **Framework comparison:** the winner design and the critic rebuilt as a LangGraph graph
  (orchestrator, SQL sub-agent, verifier sub-agent) with the same model, prompts and tools, on
  the ablation set. The own loop stays the main system unless the graph is better on both EX
  and AURC with paired 95% intervals excluding zero.

## Scoring

- **EX** as BIRD's official evaluator computes it: the predicted and the gold query run in one
  read-only transaction, and their rows are compared as sets. The project's implementation is
  checked against the official one (`results/metrics/ex_validation.json`). Soft-F1 is reported
  as a secondary measure.
- A declined question counts as wrong in execution accuracy, and is ranked below every answered
  question in the risk-coverage curve. Ties in confidence are averaged over their order.
- **Banking set:** standard and multi-step questions by EX; ambiguous ones succeed with a
  clarifying question, or with a stated assumption whose SQL matches an accepted reading;
  unanswerable ones by declining with a reason; false premises by correcting the premise;
  comparative and causal ones by the statistical checks, later. The rate of declines on
  answerable questions is reported.
- **Intervals:** percentile bootstrap, 10,000 resamples over questions, 95%, seed 20260927;
  paired for comparisons of runs on the same questions. ECE with ten equal-width bins (equal-
  mass bins reported too); AUROC with ties counted one half. R-VES is not computed: timings on
  one machine are not comparable with published ones. Results from a single run are labelled
  as such.

## Predictions

Written before any design runs. Each is checked against the result, whichever way it goes, and
reported.

```yaml
splits_sha256: efbcc64180d476b1038f843a71a8697f0e2ff875eb32312cbc0e1afd6809d0b3
own_set_sha256: 83a370fec23883447b983e3be2cbbbba1c1d042bffd5e9160f0ac93251bdc321
predictions:
  - id: P01
    claim: EX of design 1 (Claude Sonnet 5, with evidence) on the ablation set
    prediction: "52-65%"
  - id: P02
    claim: EX of the best design on the ablation set, and which design it is
    prediction: "60-70%, design 4"
  - id: P03
    claim: The design the selection rule picks
    prediction: "design 4"
  - id: P04
    claim: Whether verification (design 4) raises EX over design 3, by a paired interval excluding zero
    prediction: "no: design 4 is 0 to 3 points above design 3, with an interval containing zero"
  - id: P05
    claim: Whether the winner's AURC is lower than design 1's, by a paired interval excluding zero
    prediction: "yes"
  - id: P06
    claim: AUROC of the winner's raw confidence against correctness on the held-out set
    prediction: "0.70-0.82"
  - id: P07
    claim: ECE of the calibrated confidence on the held-out set
    prediction: "below 0.08"
  - id: P08
    claim: Accuracy and coverage on the held-out set at the decline threshold chosen for 90% accuracy
    prediction: "accuracy 83-90% at 35-55% coverage, short of the 90% target"
  - id: P09
    claim: EX of Claude Haiku 4.5 minus Claude Sonnet 5 on design 3
    prediction: "-15 to -5 points"
  - id: P10
    claim: EX of the local model on design 1
    prediction: "8-20%"
  - id: P11
    claim: EX of the winner with evidence minus without, on the ablation set
    prediction: "+8 to +20 points"
  - id: P12
    claim: EX of the winner on the banking set's standard and multi-step questions, against its held-out benchmark EX
    prediction: "65-85% on the standard and multi-step questions, above the held-out benchmark EX"
  - id: P13
    claim: Success on the banking set's ambiguous, unanswerable and false-premise questions
    prediction: "ambiguous 40-70%, unanswerable 60-90%, false premise 40-75%"
  - id: P14
    claim: Whether escalation to Claude Opus 5.5 is adopted
    prediction: "no"
  - id: P15
    claim: Whether the LangGraph graph replaces the own loop
    prediction: "no"
```

## Amendments

**2026-09-28, before the first scored run of the design comparison.** These make precise what the
text above leaves open; none changes a question set, the selection rule, a Phase 5 rule or a
prediction.

1. **The answer's form.** Every design answers through one call of a `submit_answer` tool with a
   strict schema (the fields listed under "The five designs", plus an optional chart). In design 1
   it is the only tool the model has, and the model must call it; design 1 still reads no data.
   The chart, when given, is checked after the run against the result's columns; an invalid chart
   is dropped and recorded. No design calls a chart tool during its loop.
2. **Model settings.** Claude Sonnet 5 runs with thinking disabled. Claude Haiku 4.5 runs at the
   API's default sampling settings (no thinking). The local model (`qwen2.5:3b-instruct`) runs at
   temperature 0, seed 0, with a 32,768-token context; a conversation whose next request might not
   fit it ends without an answer.
3. **Design 4's samples.** Its first sample is design 3's run on the same question: the same
   requests, so the same responses. The second and third samples are new. Design 5's three samples
   are all new.
4. **Design 5's narrowing.** The samples see only the chosen tables in the table list, only the
   chosen columns in `describe_table` and `sample_rows`, and may query only the chosen tables (the
   query guard is built on them). A query may still name an unchosen column of a chosen table. If
   the narrowing call chooses nothing usable, the samples get the full schema.
5. **The loop's limits.** Tool results show the model at most 20 rows, with the total count, and
   no timings. A turn with no tool call gets one reminder to submit; a conversation makes at most
   24 model calls. A refusal, a reply cut off at its token limit, or reaching a limit ends the
   conversation without an answer, which counts as wrong (confidence 0).
6. **The automatic checks** (design 4 and 5's confidence), made precise: a count (a `COUNT`
   aggregate or a count-like name) or an amount (an expression over an `amount` column) is out of
   range when negative, unless its expression subtracts or negates; a share (a percent-like name,
   or a division multiplied by 100) is out of range outside 0-100. In the vote, samples that
   decline agree with each other, and a sample with no SQL, or whose SQL fails, agrees with no
   other.
7. **Descriptions in both settings.** The database descriptions BIRD ships reach the model in both
   evidence settings (in the schema of design 1 and through `describe_table`); without evidence,
   only the per-question hint is left out.
8. **Prompts frozen** after three tuning rounds on the pilot set (sha256):
   `prompts/single_shot_v3.md` `c76d55ca986d5827eeb1dc444dc2b7431330d8651055038f80481ebf5b854a27`,
   `prompts/agent_v3.md` `011ad8a18ebeec59d62e8044f934fd833c5349639d5a067b9acf33fe22b34da5`,
   `prompts/narrow_v1.md` `f9fe316a146e9b143ec542c75a74bf72516bba1a4e31f39635e7595c332db327`.
