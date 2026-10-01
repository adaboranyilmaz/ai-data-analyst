# An AI Data Analyst with Auditable Answers

*When an AI analyst answers a question about a bank's database, can it tell you how much to trust the answer, and does its confidence track whether it is right?*

This project builds an AI analyst that answers questions about a bank's database by writing and running SQL. Every answer comes with its evidence and a confidence, and the analyst declines when it is unsure. The project measures whether that confidence can be trusted.

**Status: under construction. The analyst is built, five designs of it have been compared, the chosen one has been evaluated on the benchmark and on the hand-written banking set, its confidence has been calibrated, turned into a point at which it declines, and used to route uncertain questions to a larger model, and its wrong answers have been sorted by where they go wrong and priced. A statistical guardrail now checks its comparative and causal answers. The demo comes next.**

**Auditable** means that every answer can be traced to the exact SQL, the rows it used and the checks it ran, and that it comes with a calibrated confidence. It does not mean guaranteed correct.

The full technical report is in [TECHNICAL_REPORT.md](TECHNICAL_REPORT.md). A replay demo of recorded runs will follow.

## The Idea in Plain Terms

- **Text-to-SQL** turns a question in plain English into a database query. The analyst writes the query, runs it, and answers from the rows it gets back.
- **Execution accuracy** is the share of questions for which the analyst's query returns the same rows as a query written by an expert.
- **Calibration** means the confidence matches reality: of the answers given a confidence of eight in ten, about eight in ten should be right.
- **Declining** means the analyst says it cannot answer reliably, and why, instead of guessing.
- **A statistical question** asks whether a difference, a trend or a relationship really holds, beyond the rows at hand, or whether one thing affects another. An honest answer gives the uncertainty (an interval) and says that a pattern in records nobody assigned at random need not be a cause.
- Accuracy tells you how often the analyst is right. This project measures whether it knows *which* of its answers are right, where and why it goes wrong, and what its mistakes cost.

## Key Findings

- **The simplest design won.** Of five designs, from a single call with the whole schema to an agent that explores the database, runs its own queries, corrects itself and votes over three attempts, the single call was the most accurate ({{abl_sonnet_d1_ex}}, against {{abl_sonnet_d3_ex}} for the self-correcting agent and {{abl_sonnet_d4_ex}} with voting) and the cheapest ({{abl_sonnet_d1_cost_q}} a question). It won by the rule fixed before any design ran.
- **Why: exploring made a strong model add things.** With the tools to run its own queries, Claude Sonnet 5 more often returned columns it had looked at but was not asked for, and the benchmark scores an extra column as wrong. Counting answers that are right in substance but wrong in form as right, the self-correcting agent all but catches up ({{ea_d3_up}} against {{ea_d1_up}}, exploratory). Its single-call queries rarely failed, so self-correction had little to fix. The smaller Claude Haiku 4.5, whose queries failed more often, gained from the same tools ({{abl_haiku_d1_ex}} to {{abl_haiku_d3_ex}}).
- **On questions nothing was tuned on**, the chosen analyst answered {{held_ex}} of the benchmark's {{held_questions}} held-out questions correctly {{held_ex_ci}}, at {{held_cost_correct}} per correct answer. BIRD's hints matter: without them, accuracy was {{evidence_gain_pts}} points lower.
- **Its confidence ranks its answers, and calibration makes it honest.** A right answer usually gets a higher confidence than a wrong one (AUROC {{held_auroc}}). The stated confidences ran too high (calibration error {{cal_raw_ece}}); adjusted on other questions, they match reality closely ({{cal_platt_ece}}).
- **But declining for high accuracy leaves few answers.** Answering only when the calibrated confidence promised {{decline_target}} accuracy, the analyst answered {{decline_held_coverage}} of the held-out questions, {{decline_held_accuracy}} of them correctly. Its confidence takes too few distinct values to separate right from wrong finely.
- **A larger model was much more accurate, and the rule fixed in advance adopted it.** On the same questions Claude Opus 5.5 answered {{esc_opus_ex}} correctly against {{esc_sonnet_ex}} for Claude Sonnet 5 ({{esc_gain_pts}} points more), at {{esc_opus_cost_correct}} per correct answer against {{esc_sonnet_cost_correct}}. On the held-out questions the gain held: {{router_opus_ex}} against {{router_sonnet_ex}}.
- **Routing only the uncertain questions to the larger model did not pay.** The analyst's confidence cleared the bar for keeping its own answer on only {{router_kept}} of {{router_questions}} questions, so the router sent almost everything to the larger model: it was no more accurate than the larger model alone, and cost more, since it paid for both.
- **A second model's review did not beat the analyst's own confidence, but the two together did, slightly.** A critic that sees the query's result ranked the answers no better (lower is better: {{rc_critic_aurc}} against {{rc_stated_aurc}}); combining both confidences reached {{rc_combined_aurc}}.
- **Most wrong answers are wrong in their logic, not their form.** Of the {{ea_wrong}} wrong held-out answers, {{ea_format_only}} had the right rows in the wrong shape and {{ea_no_result}} returned nothing; the rest read other tables ({{ea_tables}}), filtered differently ({{ea_filter}}) or calculated differently ({{ea_computation}}). Reading {{hc_items}} of them closely found that in {{hc_questionable}} the benchmark's own expert query does not answer the question as asked.
- **Once a wrong answer costs more than {{da_break_even}}, the larger model is the cheaper one**, although each answer costs more ({{da_opus_api}} against {{da_sonnet_api}}): it makes fewer mistakes. Declining pays when a person's answer costs less than {{da_sonnet_pays}} times what a wrong answer costs. The router is never the cheapest choice.
- **On the hand-written banking questions**, which no model can have seen, it answered {{own_ab_ex}} of the standard and multi-step questions correctly, declined {{own_d_success}} of the unanswerable ones and corrected {{own_e_success}} of the false premises, while asking a clarifying question on only {{own_clarify_ab}} of the clear questions.
- **The statistical guardrail keeps the answers true to the test, but not the test true to the question.** On copies of the bank's data with effects planted in them, answers that saw the test claimed an effect where none was planted on {{l2_none_guarded}} of the copies, against {{l2_none_numbers}} for answers written from the numbers alone, and never contradicted the test. But the analyst chooses what to compare within, and where a planted third factor produced the difference, it never chose that factor: its test reported the false effect on {{pl_ana_confounded}} of those copies, and the answers repeated it.
- **On the banking set's comparative and causal questions**, the analysis ran to the end on {{gb_ran}} of {{gr_f_questions}}, each answer with an interval and the caveat that an association is not a cause; read against what each question's check asks for, {{gb_reviewed}} said all of it. Without the guardrail, the analyst had answered them without seeing any data, and none gave an interval.

![Risk-coverage curves on the held-out questions](results/plots/risk_coverage_held_out.png)

*How often the analyst is wrong among the answers it keeps, keeping its most confident answers first (lower is better). Its own calibrated confidence (blue) and a second model's review (orange) rank the answers about equally well; the two combined (green, exploratory) do a little better. The dotted line is a perfect ranking. One run, {{held_questions}} held-out questions.*

## How It Works

The design, being built in stages:

- **The client database.** Real, anonymized data from a Czech bank (1993–1998), with {{financial_trans_rows}} transactions. It is the standard public relational banking dataset and part of the BIRD benchmark. Its codes are in Czech; translating them into business terms is what an analyst does with any bank's internal codes. A hand-written data dictionary gives every column an English name, a meaning and a unit, and translates all {{financial_code_values}} code values found in the data.
- **The benchmark.** BIRD mini-dev, a public text-to-SQL benchmark of {{bird_questions}} questions over {{bird_databases}} databases, with an expert-written query for every question, so the analyst can be compared with published results. All {{gold_executed}} expert queries run on this project's database.
- **A hand-written banking test set.** {{own_set_questions}} questions written for this project and fixed before the analyst sees them, including ambiguous, unanswerable and false-premise questions. Unlike a public benchmark, they cannot be in any model's training data.
- **The analyst.** An agent loop built directly on the Anthropic SDK, with tools to list and describe tables, look at sample rows, run queries and check chart designs.
- **Two independent guards on the database.** A SQL checker accepts only a single read-only query over the analyst's own tables. Separately, the database itself runs every query under a role that can read that one database's tables and nothing else, and stops it at a time limit. Each guard is tested on its own against the same attacks (see Results).
- **A statistical guardrail.** Comparative and causal questions take a second path: the analyst plans an analysis, its query pulls one row per unit, a tested statistics program computes the intervals in a locked-down container, and the answer is written from that result, with the observational caveat attached (see below).
- **The evaluation.** Accuracy is scored exactly as BIRD's official evaluator scores it, checked against the official code query by query (see Results). Beyond accuracy: how accuracy rises as the analyst declines its least confident answers (a risk–coverage curve), and whether its confidence is calibrated on questions it was not tuned on. The benchmark's questions are split once, before any run: {{split_pilot}} to write the prompts on, {{split_ablation}} to choose the design and calibrate confidence on, and {{split_held_out}} held out for the reported results. How the design will be chosen, and what is expected, is written down before the first run.

## Results

**Choosing the design.** Five designs, each adding one step to the one before, were run on the {{split_ablation}} questions set aside for choosing, with Claude Sonnet 5 on all five, and Claude Haiku 4.5 and a small local model on some. The winner is the design whose confidence best ranks its answers, measured by the area under the risk–coverage curve (AURC); a cheaper design wins if it is not shown to be worse.

{{table:design_comparison}}

- The single call wins outright: the most accurate and the best-ranked, and the cheapest. Each agentic design was less accurate than it on the same questions: the self-correcting agent by {{abl_d1_minus_d3_pts}} points (the single call's lead, with its interval: {{abl_d1_minus_d3}} {{abl_d1_minus_d3_ci}}).
- Voting over three attempts lowered accuracy ({{abl_d4_minus_d3}} {{abl_d4_minus_d3_ci}} against the single attempt): the attempts share the model's habits, so two can agree on the same mistake and outvote a right answer.
- The small local model reached {{abl_local_d1_ex}} with a single call and {{abl_local_d3_ex}} with the tools, far below the hosted models.

**The chosen analyst on the benchmark.** On the {{held_questions}} held-out questions it answered {{held_ex}} correctly {{held_ex_ci}}: {{held_simple_ex}} of the simple ones and {{held_challenging_ex}} of the challenging ones. Answering only its most confident four in five raises that to {{held_acc_at_80}}. On all {{bird_questions}} questions it scored {{all_ex}}. Without BIRD's hints, which spell out the definitions a question relies on, it scored {{noev_ex}} on the choosing set, {{evidence_gain_pts}} points lower (the hints' gain, with its interval: {{evidence_gain}} {{evidence_gain_ci}}).

**The chosen analyst on the banking set.** It answered {{own_ab_ex}} of the standard and multi-step questions correctly {{own_ab_ex_ci}}, declined {{own_d_success}} of the unanswerable ones with a reason, and corrected {{own_e_success}} of the false premises. A check by hand of what it said confirmed {{own_e_reviewed}} of the corrections; of the ambiguous questions, {{own_c_reviewed}} were handled well after the check ({{own_c_success}} by the automatic rule, which counts any clarifying question). On the clear questions it asked for clarification on {{own_clarify_ab}}.

<sub>Source: `results/metrics/benchmark_main.json`, `results/metrics/own_set.json`, `results/reviews/own_set_review.yaml` (one run each)</sub>

**Calibrating the confidence and declining.** Fitted on the {{split_ablation}} questions set aside for it and checked on the {{held_questions}} held-out ones:

- The stated confidence ran too high: its calibration error was {{cal_raw_ece}}. After calibration it was {{cal_platt_ece}} {{cal_platt_ece_ci}}.
- To be right {{decline_target}} of the time, the analyst may only answer when it states a confidence of at least {{decline_raw_threshold}}. On the held-out questions that meant answering {{decline_held_coverage}} {{decline_held_coverage_ci}} of them, {{decline_held_accuracy}} correctly.
- A second model, shown the question, the SQL, the rows it returned and the checks, ranked the answers no better than the analyst's own confidence (the difference in AURC: {{rc_critic_minus_stated}} {{rc_critic_minus_stated_ci}}). Combined, the two did slightly better ({{rc_combined_minus_stated}} {{rc_combined_minus_stated_ci}}, exploratory).
- Of the wrong answers whose query ran, {{nm_near}} of {{nm_wrong}} held the right rows, with extra or reordered columns or other rounding. Counted as right, they would lift the held-out accuracy from {{held_ex}} to {{held_ex_up_to_format}}; execution accuracy, which counts them wrong as the benchmark does, stays the measure everywhere else.

<sub>Source: `results/metrics/calibration.json`, `results/metrics/risk_coverage.json`, `results/metrics/benchmark_main.json`, `results/metrics/near_miss.json` (one run each)</sub>

**Escalating to a larger model.** On the {{split_ablation}} questions set aside for choosing, Claude Opus 5.5 was {{esc_gain}} {{esc_gain_ci}} more accurate than Claude Sonnet 5 on the same design, so the rule fixed in advance adopted it. The router it brings in sends Opus the held-out questions Sonnet is unsure of ({{router_routed}} of {{router_questions}}):

- The routed system answered {{router_ex}} {{router_ex_ci}} of the held-out questions correctly, against {{router_sonnet_ex}} for Sonnet alone ({{router_gain}} {{router_gain_ci}}), at {{router_cost_correct}} per correct answer against {{router_sonnet_cost_correct}}.
- Opus alone, run on every held-out question as an extra comparison, scored the same, {{router_opus_ex}}, at {{router_opus_cost_correct}} per correct answer: on the few questions the router kept, both models got the same ones right. Routing pays for a Sonnet answer first and then sets most of them aside.
- Costs are at the batch price. When the batches stalled for hours, Opus's last {{router_direct_questions}} held-out questions were sent as direct calls at twice the price; its answers do not depend on how a request is sent.

<sub>Source: `results/metrics/escalation.json`, `results/metrics/router.json` (one run per model)</sub>

**The same pipeline in a framework.** Rebuilt with LangGraph (an orchestrator routing between a SQL agent and the reviewing model), the pipeline sent exactly the same requests on all {{fw_identical}} questions, so its answers and accuracy are identical. Its overhead per question was {{fw_graph_p50_ms}} ms at the median, against {{fw_own_p50_ms}} ms for the hand-written loop, and it saved {{fw_checkpoints}} checkpoints of its state each time. The hand-written loop stays the main system.

<sub>Source: `results/metrics/framework_comparison.json` (one run, from stored responses)</sub>

**Checking the scorer.** Before scoring the analyst, the project's scoring was run beside BIRD's official evaluator on {{ex_validation_cases}} test queries: the {{ex_validation_gold}} expert queries themselves, {{ex_validation_mutants}} expert queries altered on purpose (a missing DISTINCT, a flipped sort, a dropped filter, a cast to another type) and {{ex_validation_edge}} hand-written edge cases.

- The two gave the same verdict on {{ex_validation_identical}} of the {{ex_validation_cases}}, including the {{ex_validation_mutants_right}} altered queries that still return the right rows. The secondary score (Soft-F1) matched to the last digit on {{soft_f1_validation_identical}}.
- Three kinds of query that the official evaluator runs are refused here by design: a second statement, a setting changed before the query, and another database's tables. The analyst's own tools refuse them before they run, so no answer of the analyst can meet them.
- With PostgreSQL's default parallel query switched on, the official evaluator scored {{parallel_gold_wrong}} of the {{parallel_gold_cases}} expert queries wrong against themselves: their floating-point sums came out differently on each run. Every query here runs without it.

<sub>Source: `results/metrics/ex_validation.json` (one run)</sub>

**The database guards.** A suite of {{security_attacks}} attacks tried to change data, run a hidden second statement, disguise SQL with comments or look-alike characters, read settings, files and other databases' tables, change session settings, tie up the server, and do what a planted instruction in the data asks. Each guard ran the suite with the other switched off:

{{table:security_suite}}

- No attack got past either guard on its own ({{security_breaches}} breaches), apart from listing table names, below.
- For runaway queries, "stopped" means cut off by the time and row limits, which stay on whichever guard is switched off. No checker can tell an expensive query from a legitimate one, so these queries are contained, not refused.
- The database cannot hide the names of its tables from a user who can connect. Only the checker stops the {{security_names_listed}} queries that list them; the data behind the names stays out of reach.
- All {{guard_gold_accepted}} expert queries of the benchmark pass the checker, and return exactly the same rows through the analyst's tools.

## Where It Goes Wrong and What Errors Cost

**Where the analyst gets it wrong.** Each of the {{ea_wrong}} wrong held-out answers was put in one category by the first part of its query that differs from the expert's:

- {{ea_no_result}} returned no result ({{ea_refused}} refused by the SQL checker, {{ea_failed}} failed), and {{ea_format_only}} had the right rows in the wrong shape.
- {{ea_logic}} are mistakes of logic: {{ea_tables}} read other tables, {{ea_filter}} used other conditions, {{ea_computation}} calculated differently, {{ea_output}} returned other columns, {{ea_join}} joined the right tables wrongly and {{ea_order_limit}} kept other rows. In {{ea_several}} of the {{ea_wrong}}, more than one part differs.
- In the last {{ea_other}}, none of these parts differs from the expert's query, and the difference lies elsewhere.
- A close reading of {{hc_items}} of them agreed with the automatic category on {{hc_same}}; where they differ, the automatic one had mostly stopped at a harmless first difference, before the real mistake. It also found {{hc_questionable}} whose expert query does not answer the question as asked, for example counting lab records where the question asks for patients. The author checked {{hc_spot_checked}} of the answers flagged this way and kept the flag on {{hc_spot_kept}}, dropping the other from the count.

![Where the wrong answers go wrong](results/plots/errors_by_category.png)

<sub>Source: `results/metrics/error_analysis.json`, `results/reviews/error_hand_check.yaml` (one run)</sub>

**What errors cost.** Nobody knows in general what a wrong answer costs, or what it costs for a person to answer a question the analyst declines, so no price is assumed. Instead, each system's expected cost per question was worked out for every pair of costs, from a cent to $10,000 for a wrong answer and from a cent to $1,000 for a declined question:

- Opus alone is cheaper than Sonnet alone once a wrong answer costs more than {{da_break_even}} (95% interval {{da_break_even_low}} to {{da_break_even_high}}). Below that, the cheaper model's extra mistakes cost less than the larger model's price.
- The router is never the cheapest: it gets exactly the same questions wrong as Opus alone and costs {{da_router_extra}} more per question.
- Declining is worth it only when a person's answer is cheap next to a wrong one: for Sonnet, below {{da_sonnet_pays}} times the cost of a wrong answer.
- Deciding question by question from the calibrated confidence works only as well as the calibration does. It is the cheapest rule in principle, but for Opus, whose confidence was calibrated on few questions, the fixed threshold sometimes did better.

![The cheapest system at each cost of a wrong answer and of declining](results/plots/cheapest_system.png)

*Which system has the lowest expected cost per question, at each cost of a wrong answer (across) and of a declined question (up). Blue is Sonnet and orange is Opus; the hatched areas decline their least confident questions. Opus declining is exploratory: its threshold was chosen the same way as Sonnet's. One run per model, {{held_questions}} held-out questions, batch prices.*

<sub>Source: `results/metrics/decision_analysis.json` (one run per model, no model calls)</sub>

## The Statistical Guardrail

Some questions ask whether something holds, not what the rows say: do loans to women go bad more often, did card withdrawals become more common, does a pension protect against overdrafts? Answered from the rows alone, such a question invites a confident claim that the data cannot support. The guardrail sends these questions down a second path:

1. **Flag.** Keyword rules and Claude Haiku 4.5, reading the question alone, flag it as statistical if either says so.
2. **Plan.** Claude Sonnet 5, reading the schema but no data, plans the analysis: a query returning one row per unit (a loan, a client), the outcome, the groups or the trend to compare, and up to two variables to compare within, in case one of them produces the difference.
3. **Analyze.** The query runs through both database guards. A tested statistics program computes the intervals and tests in a container with no network, a read-only file system and no privileges, and checks whether the result holds within the chosen variables.
4. **Answer.** Sonnet writes the answer from that result. The reader gets it with the interval, any warning (few cases, or areas rather than people), a correction wherever the answer's claim and the test disagree, and the caveat that records nobody assigned at random show associations, not causes.

**Which questions it flags.** It flagged {{gr_f_flagged}} of the {{gr_f_questions}} comparative and causal questions in the banking set and {{gr_planted_flagged}} of the {{gr_planted_questions}} planted ones (below). It also flagged {{gr_other_flagged}} of the banking set's other {{gr_other_questions}} questions, most of which the plan then declined (the data does not record what they ask about, or their premise is false), and {{gr_bench_flagged}} of the {{gr_bench_questions}} benchmark questions ({{gr_bench_rate}}), all of them counts or lookups on reading.

**The banking set's comparative and causal questions.** Before the guardrail, the analyst answered them in one call, without seeing any data; none gave an interval, and several said "no real difference" where the data shows a clear one.

{{table:banking_causal}}

- On {{gb_ran}} of {{gr_f_questions}}, the path ran to the end and the answer met every criterion fixed in advance: an interval, no causal claim, the caveat, and a warning where the counts are small. Of the other two, one plan's query ran past the time limit and the SQL checker refused the other's cross join.
- Read against what each question's check asks for, {{gb_reviewed}} said all of it. The others missed the specific caveat the check names (for example, that most card holders got their card only after their loan was granted, so the card cannot have led to it), or pooled groups the check asks to compare.

**Planted effects: does the test find what is there?** Effects of known size were planted in copies of the bank's loans: {{pl_copies}} copies of each of eight questions under each condition, two questions each on rates, means and trends (no effect, a small one or a large one), and two where a third factor produces a difference or hides a real one pointing the other way. Each copy was analyzed with the analyst's plan and with a reference plan written beforehand.

- **The test is sound.** With the reference plans, it found an effect where none was planted in {{pl_ref_false_alarms}} of the copies (it is built for one in twenty), found planted effects about as often as theory predicts in {{pl_power_within}} of {{pl_power_cells}} cases, and saw through a planted third factor: a false effect in {{pl_ref_confounded}} of those copies, against {{pl_crude_confounded}} without comparing within the factor.
- **The analyst's plans are the weak point.** It never chose to compare within the planted third factor, so its test reported the false effect on {{pl_ana_confounded}} of those copies, and where a real effect pointed the other way, it found that effect pointing the wrong way in {{pl_ana_reversed_opposite}} of the copies and the right way in none. Comparing within a variable that nearly fixes the groups also cost it power: a district's average salary nearly decides whether the branch is in Prague, and on one question the analyst's plan found a large planted effect in {{pl_a2_large_ana}} of the copies, against {{pl_a2_large_ref}} for the reference plan.

![How often the test finds an effect in the planted copies](results/plots/planted_detection.png)

*Per question and condition, the share of {{pl_copies}} copies in which the test found an effect, in the planted direction where one was planted (hollow marks: found in the opposite direction). Gray: the reference plan; blue: the analyst's; orange: the reference plan without comparing within the third factor. Black ticks: the theoretical power. One run.*

**Planted effects: what do the answers say?** On the first {{pl_copies2}} copies of each question and condition, Sonnet answered twice: once from the numbers alone (each group's size and share or mean) and once from the guarded input, with the test.

- Where nothing was planted, the guarded answers claimed an effect on {{l2_none_guarded}} of the copies, only where the test itself had a false alarm, against {{l2_none_numbers}} from the numbers alone. Over all copies without an effect, the pre-registered measure, the guardrail lowered the claims from {{l2_primary_numbers}} to {{l2_primary_guarded}} ({{l2_primary_diff}}, {{l2_primary_diff_ci}}); it did not lower them more because, where a third factor produced the difference, both kinds of answer claimed it every time: the guarded answer follows the test, and the test followed the analyst's plan.
- The guarded answers never contradicted the test ({{l2_disagree_guarded}} of {{l2_disagree_guarded_n}}); those from the numbers alone did {{l2_disagree_numbers}} times. Where a small effect was planted, the guarded answers claimed it less often ({{l2_small_guarded}} against {{l2_small_numbers}}): the test finds a small effect only about half the time, and the guarded answers claim no more than it finds.
- Read sentence by sentence, {{l2_causal_read_guarded}} guarded answers and {{l2_causal_read_numbers}} from the numbers alone claimed a cause, counting a claim that one thing does not affect another.

![What the answers claim](results/plots/planted_claims.png)

*The share of answers that claim an effect where none was planted (top two rows) and that claim the planted direction where one was (bottom three), from the numbers alone (gray) and guarded (blue). One run, {{pl_copies2}} copies per question and condition.*

**The sandbox.** Each analysis runs in a fresh container with no network, a read-only file system, an unprivileged user and limits on time, memory, processes and output. It was attacked as if it ran hostile code:

{{table:sandbox_attacks}}

- {{sb_blocked}} of the {{sb_attacks}} attacks were blocked: what they looked for was absent, the operating system denied it, a limit contained it, or the runner rejected or correctly handled the malformed input. For each protection, the same attack was also run with that protection switched off, and it succeeded ({{sb_controls_achieved}} of {{sb_controls}}), so the attacks are real and each protection is what stops them.

Of the guardrail's {{gr_predictions}} predictions, fixed before the first paid call, {{gr_predictions_held}} held; the technical report lists them.

<sub>Source: `results/metrics/guardrail.json`, `results/metrics/planted_effects.json`, `results/reviews/guardrail_review.yaml`, `results/metrics/sandbox_security.json` (one run each)</sub>

## Try It

TBD.

## Reproducing the Results

- Needs Python 3.12 with [uv](https://docs.astral.sh/uv/), and Docker.
- The database runs in Docker. The data pipeline downloads the benchmark (about 800 MB), loads it and runs every expert query. Every model response is cached, so replaying the results costs nothing.
- The technical report has the details.

```
uv sync
docker compose up -d --wait
uv run dvc repro
uv run pytest
```

## Limitations

- Every result is from one run; the intervals show how much it could move with other questions, not with another run.
- The benchmark scores an answer's rows exactly: an extra column, or a number of another type, makes an otherwise useful answer wrong. Part of the agentic designs' loss is of this kind.
- The calibration and the decline threshold were set on {{split_ablation}} questions, so they carry the uncertainty of a small sample.
- {{gold_reads_the_clock}} of the benchmark's expert queries compute ages from today's date, so their correct answers change over time. The scorer runs them beside each answer, but a stored answer that fixed a year can go out of date.
- The cost analysis prices every wrong answer the same and assumes a person answers a declined question correctly. It shows which system is cheapest for any pair of costs, not what the costs are.
- The error categories read the structure of a query, not its intent, so a harmless difference can hide the real mistake behind it.
- The security results are from one run of a fixed set of attacks. They show that each guard stops these attacks, not that no other attack exists.
- The guardrail is only as good as the analysis the analyst plans: it computes the plan's test correctly but cannot see a third factor the plan leaves out, and the planted copies show that the analyst missed such factors. Its planted effects are simple and of known form; real effects are not.
- Whether a guarded answer meets each question's check was judged by reading, by one reader, on nine questions.
- BIRD's questions are public and may be in the models' training data. The hand-written banking set exists to check for that.

## Dependencies

| Component | Used for |
|---|---|
| Python, uv | the code and its locked environment |
| PostgreSQL, in Docker | the databases, with a read-only role for the analyst |
| Anthropic Python SDK | the analyst's model calls |
| Ollama | a free local model for comparison |
| psycopg | database access from Python |
| sqlglot | the SQL checker's parser |
| jsonschema, Vega-Lite schema | checking the analyst's chart designs |
| NumPy | bootstrap intervals |
| SciPy, statsmodels, pandas, in Docker | the guardrail's statistics, in a locked-down container |
| MLflow, DVC | experiment tracking and data versioning |
| LangGraph | the framework comparison only |
| Matplotlib | the figures |

<sub>Source: `pyproject.toml`, `docker-compose.yml`</sub>

## References

Li, J., et al. (2023). *Can LLM Already Serve as a Database Interface? A BIg Bench for Large-Scale Database Grounded Text-to-SQLs* (BIRD). NeurIPS Datasets and Benchmarks. Data licensed CC BY-SA 4.0.

Berka, P. (1999). *Guide to the Financial Data Set*. PKDD'99 Discovery Challenge.
