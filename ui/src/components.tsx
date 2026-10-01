import { useEffect, useRef, useState } from "react";
import type { ReactNode } from "react";
import hljs from "highlight.js/lib/core";
import sqlLang from "highlight.js/lib/languages/sql";
import { outlineLines } from "./outline";
import { resultLine, toMarkdown } from "./summary";
import { pageOf, pageTitle, paginate } from "./pager";
import { meterWording, pct, statusLabel, workedExample } from "./wording";
import type { AnswerEvent, CheckLine, Confidence, Explanation, IndexEntry, MeterSummary, ResultTableData, RowsEvent, RunView, SqlEvent, Statistics } from "./types";

hljs.registerLanguage("sql", sqlLang);

/* ---- small inline icons (no icon font, no network) ---- */
type IconProps = { size?: number };
const svg = (size: number, children: ReactNode) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
    {children}
  </svg>
);
export const IconBars = ({ size = 18 }: IconProps) => svg(size, <path d="M6 20V12M12 20V5M18 20v-6" />);
export const IconSun = ({ size = 18 }: IconProps) => svg(size, <><circle cx="12" cy="12" r="4" /><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4" /></>);
export const IconMoon = ({ size = 18 }: IconProps) => svg(size, <path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z" />);
export const IconCheck = ({ size = 12 }: IconProps) => svg(size, <path d="M5 12.5l4.5 4.5L19 7.5" />);
export const IconX = ({ size = 12 }: IconProps) => svg(size, <path d="M6 6l12 12M18 6L6 18" />);
export const IconInfo = ({ size = 16 }: IconProps) => svg(size, <><circle cx="12" cy="12" r="9" /><path d="M12 8v5M12 16.5v.01" /></>);
export const IconSteps = ({ size = 14 }: IconProps) => svg(size, <><circle cx="6" cy="6" r="2" /><circle cx="6" cy="18" r="2" /><path d="M6 8v8M8 6h8a3 3 0 0 1 3 3v1" /></>);
export const IconCode = ({ size = 14 }: IconProps) => svg(size, <path d="M8 7l-5 5 5 5M16 7l5 5-5 5M14 4l-4 16" />);
export const IconTable = ({ size = 14 }: IconProps) => svg(size, <><rect x="3" y="4" width="18" height="16" rx="2" /><path d="M3 10h18M9 4v16" /></>);
export const IconShield = ({ size = 14 }: IconProps) => svg(size, <path d="M12 3l8 3v6c0 5-3.5 8-8 9-4.5-1-8-4-8-9V6z" />);
export const IconGauge = ({ size = 14 }: IconProps) => svg(size, <><path d="M4 18a8 8 0 1 1 16 0" /><path d="M12 18l4-6" /></>);
export const IconChart = ({ size = 14 }: IconProps) => svg(size, <path d="M4 20V4M4 20h16M8 16v-4M12 16V8M16 16v-6" />);
export const IconSigma = ({ size = 14 }: IconProps) => svg(size, <path d="M18 5H6l7 7-7 7h12" />);
export const IconLink = ({ size = 14 }: IconProps) => svg(size, <path d="M10 14a4 4 0 0 0 5.7 0l3-3a4 4 0 0 0-5.7-5.7l-1 1M14 10a4 4 0 0 0-5.7 0l-3 3a4 4 0 0 0 5.7 5.7l1-1" />);
export const IconChevron = ({ dir, size = 16 }: IconProps & { dir: "left" | "right" }) => svg(size, <path d={dir === "left" ? "M15 5l-7 7 7 7" : "M9 5l7 7-7 7"} />);
export const IconDb = ({ size = 14 }: IconProps) => svg(size, <><ellipse cx="12" cy="6" rx="8" ry="3" /><path d="M4 6v6c0 1.7 3.6 3 8 3s8-1.3 8-3V6M4 12v6c0 1.7 3.6 3 8 3s8-1.3 8-3v-6" /></>);

export function Eyebrow({ icon, children }: { icon: ReactNode; children: ReactNode }) {
  return <h2 className="eyebrow">{icon}{children}</h2>;
}

export function Pill({ status }: { status: string }) {
  return <span className={`pill ${status}`}>{statusLabel[status] ?? status}</span>;
}

export function Timeline({ run, running }: { run: RunView; running: boolean }) {
  return (
    <section className="card" aria-label="Steps">
      <Eyebrow icon={<IconSteps />}>Steps</Eyebrow>
      <ol className="timeline" aria-live="polite">
        {run.steps.map((s, i) => (
          <li key={i} className={s.kind}>
            {s.text}
          </li>
        ))}
        {running && <li className="working muted">Working…</li>}
      </ol>
      {!run.steps.length && !running && <p className="muted">Steps appear here as the analyst works.</p>}
    </section>
  );
}

export function SqlView({ sql }: { sql: SqlEvent }) {
  // highlight.js escapes the text it is given; the result is safe to insert as markup
  const html = hljs.highlight(sql.sql, { language: "sql" }).value;
  const lines = outlineLines(sql.outline);
  const [copied, setCopied] = useState(false);
  const copy = async () => {
    try {
      await navigator.clipboard.writeText(sql.sql);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      /* no clipboard in this context */
    }
  };
  return (
    <section className="card" aria-label="SQL">
      <Eyebrow icon={<IconCode />}>The query</Eyebrow>
      <div className="codebox">
        <pre className="sql" tabIndex={0}>
          <code dangerouslySetInnerHTML={{ __html: html }} />
        </pre>
        <button className="btn ghost" onClick={copy} aria-label="Copy the query">{copied ? "Copied" : "Copy"}</button>
      </div>
      {lines.length > 0 && (
        <>
          <p className="muted" style={{ margin: "14px 0 0", fontSize: "0.82rem", fontWeight: 600 }}>In words</p>
          <dl className="plain">
            {lines.map((l, i) => (
              <div key={i} style={{ display: "contents" }}>
                <dt>{l.label}</dt>
                <dd>{l.text}</dd>
              </div>
            ))}
          </dl>
        </>
      )}
    </section>
  );
}

const fmt = (v: unknown): string => (v === null || v === undefined ? "∅" : typeof v === "object" ? JSON.stringify(v) : String(v));

export function ResultTable({ rows }: { rows: RowsEvent }) {
  if (!rows.ok) {
    return (
      <section className="card" aria-label="Result">
        <Eyebrow icon={<IconTable />}>The rows</Eyebrow>
        <p className="callout bad"><IconInfo />The query did not return rows: {rows.error?.message ?? "an error"}.</p>
      </section>
    );
  }
  return (
    <section className="card" aria-label="Result">
      <Eyebrow icon={<IconTable />}>The rows it used</Eyebrow>
      <div className="table-wrap" tabIndex={0}>
        <table className="rows">
          <thead>
            <tr>{rows.columns.map((c, i) => <th key={c} scope="col" className={typeof rows.rows[0]?.[i] === "number" ? "num" : ""}>{c}</th>)}</tr>
          </thead>
          <tbody>
            {rows.rows.map((r, i) => (
              <tr key={i}>
                {r.map((v, j) => <td key={j} className={typeof v === "number" ? "num" : ""}>{fmt(v)}</td>)}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="note">
        {rows.rows.length} of {rows.fetched_rows}{rows.truncated ? "+" : ""} rows shown.
      </p>
    </section>
  );
}

export function ChartView({ spec, columns, rows }: { spec: Record<string, unknown>; columns: string[]; rows: unknown[][] }) {
  const el = useRef<HTMLDivElement>(null);
  const [problem, setProblem] = useState<string | null>(null);
  useEffect(() => {
    let view: { finalize: () => void } | undefined;
    let dead = false;
    (async () => {
      try {
        const embed = (await import("vega-embed")).default;
        const values = rows.map((r) => Object.fromEntries(columns.map((c, i) => [c, r[i]])));
        const dark = document.documentElement.getAttribute("data-theme") === "dark" ||
          (!document.documentElement.getAttribute("data-theme") && window.matchMedia("(prefers-color-scheme: dark)").matches);
        const res = await embed(el.current!, { ...spec, width: "container", data: { values } } as never, {
          actions: false, renderer: "svg", theme: dark ? "dark" : undefined,
          config: { background: "transparent", range: { category: ["#6366f1", "#14b8a6", "#f59e0b", "#ec4899", "#0ea5e9", "#84cc16"] } },
        } as never);
        view = res.view;
        if (dead) view.finalize();
      } catch (e) {
        setProblem(String(e));
      }
    })();
    return () => { dead = true; view?.finalize(); };
  }, [spec, columns, rows]);
  return (
    <section className="card" aria-label="Chart">
      <Eyebrow icon={<IconChart />}>Chart</Eyebrow>
      {problem ? <p className="callout"><IconInfo />The chart could not be drawn.</p> : <div ref={el} role="img" aria-label="Chart of the result rows" />}
    </section>
  );
}

export function Checks({ checks }: { checks: CheckLine[] }) {
  if (!checks.length) return null;
  return (
    <section className="card" aria-label="Checks">
      <Eyebrow icon={<IconShield />}>Automatic checks</Eyebrow>
      <ul className="checks">
        {checks.map((c) => (
          <li key={c.name} className={c.ok ? "" : "bad"}>
            <span className="ic">{c.ok ? <IconCheck /> : <IconX />}</span>
            <span>{c.text}</span>
          </li>
        ))}
      </ul>
    </section>
  );
}

const num = (x: number, d = 3) => (Math.abs(x) >= 100 ? x.toFixed(0) : x.toFixed(d));

export function StatisticsView({ s }: { s: Statistics }) {
  return (
    <section className="card" aria-label="Statistics">
      <Eyebrow icon={<IconSigma />}>Statistics (95% intervals)</Eyebrow>
      {s.groups && (
        <div className="table-wrap">
          <table className="rows">
            <thead><tr><th scope="col">Group</th><th scope="col">Units</th><th scope="col">Estimate</th><th scope="col">Interval</th></tr></thead>
            <tbody>
              {s.groups.map((g) => (
                <tr key={g.label}><td>{g.label}</td><td className="num">{g.n}</td><td className="num">{num(g.estimate)}</td><td className="num">{num(g.ci[0])} to {num(g.ci[1])}</td></tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      {s.comparisons?.map((c) => (
        <p key={c.group} className="stat-line">
          <strong>{c.group}</strong>: {c.measure} {num(c.estimate)} (interval {num(c.ci[0])} to {num(c.ci[1])}, p {c.p_value < 0.001 ? "< 0.001" : `= ${num(c.p_value)}`}). <span className="faint">{c.method}</span>
        </p>
      ))}
      {s.stratified && (
        <p className="stat-line">
          Within levels of {s.stratified.strata.join(", ")}: {num(s.stratified.adjusted.estimate)} (interval {num(s.stratified.adjusted.ci[0])} to {num(s.stratified.adjusted.ci[1])}); without them: {num(s.stratified.crude.estimate)}. The check says: <strong>{s.stratified.change}</strong>.
        </p>
      )}
      {s.warnings?.map((w, i) => (
        <p key={i} className="callout"><IconInfo />Warning: {w.kind === "binned_stratum" ? `${String(w.variable)} was cut into ${String(w.bins)} groups to compare within it` : w.kind.replace(/_/g, " ")}.</p>
      ))}
      <p className="note">The groups were not assigned at random: a difference or a trend here is an association, not proof of cause.</p>
    </section>
  );
}

export function ConfidenceMeter({ c, meter }: { c: Confidence; meter: MeterSummary }) {
  const w = meterWording(c, meter);
  const value = c.calibrated ?? c.stated ?? 0;
  const hasBar = !c.not_calibrated ? c.calibrated != null : c.stated != null;
  const kind = c.not_calibrated ? "model" : c.withheld ? "low" : "ok";
  const big = w.headline.match(/(\d+)%/);
  const r = w.record;
  const cal = meter.calibration;
  const ex = !c.not_calibrated && c.stated != null && c.calibrated != null ? workedExample(c.stated, meter) : null;
  return (
    <section className="card meter" aria-label="Confidence">
      <Eyebrow icon={<IconGauge />}>Confidence</Eyebrow>
      {big ? (
        <p className="big">{big[1]}%<small>{c.not_calibrated ? "stated by the model" : "calibrated"}</small></p>
      ) : (
        <p style={{ fontSize: "1.15rem", fontWeight: 650 }}>{w.headline}</p>
      )}
      <p className="sr-only">{w.headline}</p>
      {hasBar && (
        <>
          <div className="gauge" role="meter" aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.round(100 * value)} aria-label={w.headline}>
            <div className={`fill ${kind}`} style={{ width: `${100 * value}%` }} />
            {!c.not_calibrated && <div className="tick" data-label={`decline threshold ${Math.round(100 * meter.threshold)}%`} style={{ left: `${100 * meter.threshold}%` }} />}
          </div>
          <div className="scale"><span>0%</span><span>100%</span></div>
        </>
      )}
      {r && (
        <div className="record" aria-label={`Held-out accuracy at this confidence: ${pct(r.accuracy)}, 95% interval ${pct(r.low)} to ${pct(r.high)}, on ${r.n} questions`}>
          <p className="record-head">How often answers like this were right: <b>{pct(r.accuracy)}</b></p>
          <div className="band">
            <div className="span" style={{ left: `${100 * r.low}%`, width: `${100 * (r.high - r.low)}%` }} />
            <div className="dot" style={{ left: `${100 * r.accuracy}%` }} />
            <span className="end" style={{ left: `${100 * r.low}%` }}>{pct(r.low)}</span>
            <span className="end" style={{ left: `${100 * r.high}%` }}>{pct(r.high)}</span>
          </div>
          <p className="record-note">95% interval, from {r.n} held-out questions{r.thin ? " (few: treat as rough)" : ""}</p>
        </div>
      )}
      <p className="detail">{w.detail}</p>
      {!c.not_calibrated && (
        <details className="how">
          <summary>How is this calculated?</summary>
          <ol>
            <li><b>The model states how sure it is</b>{c.stated != null ? `: ${pct(c.stated)} for this answer.` : "."}</li>
            <li>
              <b>That number is recalibrated.</b> A model's own confidence is usually too high. So it is passed through a logistic curve
              (Platt scaling), fitted on {cal.fitted_on} calibration questions so that the result matches how often the model was actually right.
              {ex && (
                <span className="formula mono"> 1 ÷ (1 + e<sup>−({ex.slope.toFixed(2)} × {c.stated?.toFixed(2)} {ex.intercept < 0 ? "−" : "+"} {Math.abs(ex.intercept).toFixed(2)})</sup>) = {pct(ex.result)}</span>
              )}
            </li>
            <li>
              <b>The decline threshold ({pct(meter.threshold)}) comes from the same {cal.fitted_on} questions:</b> it is the lowest calibrated confidence above which
              answers were right at least {pct(cal.target_accuracy)} of the time.
            </li>
            <li>
              <b>Then it is tested on questions it never saw.</b> The bar above is how often answers at this confidence were right on the {meter.held_out.questions} held-out
              questions, with a 95% interval for how much that rate could move.
            </li>
          </ol>
        </details>
      )}
    </section>
  );
}

const pageLabel: Record<string, string> = { benchmark: "Benchmark", banking: "Banking", guardrail: "Comparison" };

/** The recorded questions, five to a page: swipe, scroll, use the arrows or the dots. */
export function QuestionPager({
  items, current, disabled, live, onPick,
}: {
  items: IndexEntry[];
  current: string | null | undefined;
  disabled: boolean;
  live: boolean;
  onPick: (q: IndexEntry) => void;
}) {
  const pages = paginate(items);
  const track = useRef<HTMLDivElement>(null);
  const [page, setPage] = useState(0);
  const go = (i: number) => {
    const el = track.current;
    if (!el) return;
    const n = Math.max(0, Math.min(pages.length - 1, i));
    el.scrollTo?.({ left: n * el.clientWidth, behavior: "smooth" });
    setPage(n);
  };
  // follow the selected run to its page
  useEffect(() => {
    const el = track.current;
    if (!el || !current) return;
    const n = pageOf(pages, current);
    if (n !== page) {
      el.scrollTo?.({ left: n * el.clientWidth, behavior: "smooth" });
      setPage(n);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [current]);
  if (!pages.length) return null;
  return (
    <div role="group" aria-roledescription="carousel" aria-label="Recorded questions, by page">
      <div className="pager-head">
        <div>
          <div className="pager-title">{pageTitle(pages[page])}</div>
          <div className="pager-count">Page {page + 1} of {pages.length} · {items.length} questions</div>
        </div>
        <div className="pager-arrows">
          <button className="icon-btn sm" onClick={() => go(page - 1)} disabled={page === 0} aria-label="Previous page"><IconChevron dir="left" /></button>
          <button className="icon-btn sm" onClick={() => go(page + 1)} disabled={page === pages.length - 1} aria-label="Next page"><IconChevron dir="right" /></button>
        </div>
      </div>
      <div
        className="pager-track" ref={track}
        onScroll={(e) => {
          const el = e.currentTarget;
          const n = Math.round(el.scrollLeft / Math.max(el.clientWidth, 1));
          if (n !== page) setPage(n);
        }}
      >
        {pages.map((p, pi) => (
          <div key={pi} className="pager-slide" role="group" aria-roledescription="slide" aria-label={`Page ${pi + 1} of ${pages.length}`}>
            <div className="qlist">
              {p.map((s) => (
                <button key={s.id} className="q" disabled={disabled} aria-current={current === s.id} onClick={() => onPick(s)}>
                  <span className={`dot ${s.status}`} aria-hidden="true" />
                  <span>
                    <span className="text">{s.question}</span>
                    <span className="meta">{pageLabel[s.kind] ?? s.kind} · {s.db_id}{live ? " · recorded run" : ""}</span>
                  </span>
                </button>
              ))}
            </div>
          </div>
        ))}
      </div>
      <div className="pager-dots" role="tablist" aria-label="Pages">
        {pages.map((_, i) => (
          <button key={i} role="tab" aria-selected={i === page} aria-label={`Go to page ${i + 1}`} className={i === page ? "on" : ""} onClick={() => go(i)} />
        ))}
      </div>
    </div>
  );
}
function MiniTable({ title, data }: { title: string; data: ResultTableData | null }) {
  return (
    <div className="mini">
      <div className="mini-title">{title}</div>
      {data ? (
        <>
          <div className="table-wrap" tabIndex={0}>
            <table className="rows">
              <thead>
                <tr>{data.columns.map((c) => <th key={c} scope="col">{c}</th>)}</tr>
              </thead>
              <tbody>
                {data.rows.map((r, i) => (
                  <tr key={i}>{r.map((v, j) => <td key={j}>{fmt(v)}</td>)}</tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="note">{data.rows.length} of {data.total_rows}{data.truncated ? "+" : ""} rows shown.</p>
        </>
      ) : (
        <p className="muted">No result.</p>
      )}
    </div>
  );
}

/** For a wrong answer: the expected result beside the analyst's, why it was scored wrong, and the expert's query. */
export function WhyWrong({ x, run }: { x: Explanation; run: RunView }) {
  const got: ResultTableData | null = run.rows && run.rows.ok
    ? { columns: run.rows.columns, rows: run.rows.rows, total_rows: run.rows.fetched_rows, truncated: run.rows.truncated }
    : null;
  return (
    <section className="why" aria-label="Why this was scored wrong">
      <h3>Why this was scored wrong</h3>
      {x.note ? (
        <p className="why-note">{x.note}</p>
      ) : (
        x.category_text && <p className="why-note">The automatic comparison finds the first difference here: {x.category_text}. It reads the structure of the queries, not their intent.</p>
      )}
      <div className="compare">
        <MiniTable title="The analyst's result" data={got} />
        <MiniTable title="The expected result" data={x.expected} />
      </div>
      {x.comparison && <p className="why-line">{x.comparison.text}</p>}
      <details className="disclose">
        <summary>Show the expert's query and the benchmark's hint</summary>
        {x.hint && <p className="muted" style={{ margin: "8px 0 0", fontSize: "0.86rem" }}><b>Hint given to the analyst:</b> {x.hint}</p>}
        {x.expected_sql && <pre className="sql" tabIndex={0} style={{ marginTop: 8 }}><code>{x.expected_sql}</code></pre>}
        <p className="note">The analyst never saw the expert's query; it is shown here, after the fact, to explain the score.</p>
      </details>
    </section>
  );
}
export function AnswerCard({ a, run, meter }: { a: AnswerEvent; run: RunView; meter: MeterSummary }) {
  const ev = run.done?.evaluation;
  const line = a.status === "declined" || a.status === "clarify" ? null : resultLine(run.rows);
  const [copiedMd, setCopiedMd] = useState(false);
  const copyMarkdown = async () => {
    try {
      await navigator.clipboard.writeText(toMarkdown(run, meter, window.location.origin));
      setCopiedMd(true);
      setTimeout(() => setCopiedMd(false), 1500);
    } catch {
      /* no clipboard in this context */
    }
  };
  const [shown, setShown] = useState(false);
  const withheld = a.status === "withheld";
  const text = a.text && a.status !== "declined" ? <p className="answer-text">{a.text}</p> : null;
  return (
    <div>
      <Pill status={a.status} />
      <p className="answer-q">{run.question}</p>
      {a.status === "declined" && <p className="callout"><IconInfo />{a.decline_reason ?? a.text}</p>}
      {a.status === "clarify" && <p className="callout"><IconInfo />The analyst asks: {a.clarifying_question}</p>}
      {withheld && (
        <p className="callout bad">
          <IconInfo />The analyst is not confident enough to stand behind this answer. What it found is shown below, to check for yourself.
        </p>
      )}
      {a.notice && <p className="callout"><IconInfo />{a.notice}</p>}
      {a.premise_correction && <p className="callout"><IconInfo />Premise corrected: {a.premise_correction}</p>}
      {withheld ? (
        text && (
          <details className="disclose" open={shown} onToggle={(e) => setShown((e.target as HTMLDetailsElement).open)}>
            <summary>Show the answer it would have given</summary>
            {text}
          </details>
        )
      ) : (
        text
      )}
      {line && !withheld && (
        <p className="resultline"><b>Result</b><code>{line}</code></p>
      )}
      {a.assumptions.length > 0 && <p className="muted" style={{ marginTop: 10, fontSize: "0.9rem" }}>Assumes: {a.assumptions.join("; ")}</p>}
      <div className="foot">
        {run.mode === "replay" && ev && ev.correct != null && (
          <span className={`evalbox ${ev.correct ? "ok" : "no"}`}>
            In the evaluation this answer was scored:{" "}
            <b className="verdict">{ev.correct ? <IconCheck /> : <IconX />}{ev.correct ? "correct" : "wrong"}</b>
            <small>({ev.rule})</small>
          </span>
        )}
        {run.mode === "replay" && ev && ev.reviewed_success != null && (
          <span className={`evalbox ${ev.reviewed_success ? "ok" : "no"}`}>
            Reviewed by the author:{" "}
            <b className="verdict">{ev.reviewed_success ? <IconCheck /> : <IconX />}{ev.reviewed_success ? "a good answer" : "falls short"}</b>
          </span>
        )}        {run.id && run.finished && !run.error && (
          <a href={`#/run/${run.id}`} style={{ display: "inline-flex", gap: 6, alignItems: "center" }}>
            <IconLink />Link to this run
          </a>
        )}
        {run.finished && !run.error && (
          <button className="btn ghost" onClick={copyMarkdown} aria-label="Copy the answer and its evidence as Markdown">
            {copiedMd ? "Copied" : "Copy as Markdown"}
          </button>
        )}
      </div>
      {run.mode === "replay" && ev && ev.correct === false && run.done?.explanation && (
        <WhyWrong x={run.done.explanation} run={run} />
      )}
    </div>
  );
}
