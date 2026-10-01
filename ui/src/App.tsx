import { useCallback, useEffect, useRef, useState } from "react";
import { AskError, ask, emptyRun, getMeta, getRun, reduce } from "./api";
import {
  AnswerCard, ChartView, Checks, ConfidenceMeter, Eyebrow, IconBars, IconDb, IconGauge, IconMoon, IconShield,
  IconSun, QuestionPager, ResultTable, SqlView, StatisticsView, Timeline,
} from "./components";
import { Connections } from "./Connections";
import type { Meta, RunView } from "./types";

type Theme = "light" | "dark" | "auto";

function storedTheme(): Theme {
  try {
    const t = localStorage.getItem("theme");
    return t === "light" || t === "dark" ? t : "auto";
  } catch {
    return "auto";
  }
}

function systemDark(): boolean {
  return window.matchMedia?.("(prefers-color-scheme: dark)").matches ?? false;
}

function runIdFromHash(): string | null {
  const m = window.location.hash.match(/^#\/run\/([A-Za-z0-9._-]+)$/);
  return m ? m[1] : null;
}


export default function App() {
  const [meta, setMeta] = useState<Meta | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [question, setQuestion] = useState("");
  const [run, setRun] = useState<RunView | null>(null);
  const [running, setRunning] = useState(false);
  const [problem, setProblem] = useState<string | null>(null);
  const [theme, setTheme] = useState<Theme>(storedTheme);
  const abort = useRef<AbortController | null>(null);

  useEffect(() => {
    if (theme === "auto") document.documentElement.removeAttribute("data-theme");
    else document.documentElement.setAttribute("data-theme", theme);
    try {
      if (theme === "auto") localStorage.removeItem("theme");
      else localStorage.setItem("theme", theme);
    } catch {
      /* the page works without storage */
    }
  }, [theme]);

  const reload = useCallback(() => {
    getMeta().then(setMeta).catch((e) => setLoadError(String(e.message ?? e)));
  }, []);
  useEffect(reload, [reload]);

  const stream = useCallback(async (body: { question?: string; run_id?: string }, label: string) => {
    abort.current?.abort();
    const ctl = new AbortController();
    abort.current = ctl;
    setProblem(null);
    setRunning(true);
    let current = emptyRun(label);
    setRun(current);
    try {
      await ask(body, (e) => {
        current = reduce(current, e);
        setRun(current);
        if (e.type === "done") window.history.replaceState(null, "", `#/run/${e.id}`);
      }, ctl.signal);
    } catch (e) {
      if ((e as Error).name !== "AbortError") setProblem(e instanceof AskError ? e.message : String(e));
    } finally {
      if (abort.current === ctl) setRunning(false);
    }
  }, []);

  // a permalink: show the stored run at once, without streaming it
  useEffect(() => {
    const open = () => {
      const id = runIdFromHash();
      if (id && id !== run?.id) {
        getRun(id).then(setRun).catch(() => setProblem("No run with this link."));
      }
    };
    open();
    window.addEventListener("hashchange", open);
    return () => window.removeEventListener("hashchange", open);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  if (loadError) {
    return <div className="app"><p className="callout bad" role="alert">The service is not reachable: {loadError}</p></div>;
  }
  if (!meta) return <div className="app"><p className="muted" style={{ padding: 40 }}>Loading…</p></div>;

  const live = meta.mode === "live";
  const dark = theme === "dark" || (theme === "auto" && systemDark());
  const submit = (e: React.FormEvent) => {
    e.preventDefault();
    if (question.trim() && !running) stream({ question: question.trim() }, question.trim());
  };

  return (
    <div className="app">
      <header className="bar">
        <div className="brand">
          <div className="logo" aria-hidden="true"><IconBars size={18} /></div>
          <div>
            <h1>AI Data Analyst</h1>
            <div className="tag">Answers you can audit</div>
          </div>
        </div>
        <div className="bar-right">
          <span className={`mode ${live ? "live" : "replay"}`}>
            <i aria-hidden="true" />
            <span data-testid="mode">{live ? "live" : "replay of recorded runs"}</span>
          </span>
          <button
            className="icon-btn" onClick={() => setTheme(dark ? "light" : "dark")}
            aria-label="Switch between light and dark theme" title={dark ? "Light theme" : "Dark theme"}
          >
            {dark ? <IconSun /> : <IconMoon />}
          </button>
        </div>
      </header>

      <div className="shell">
        <aside className="side">
          <section className="card ask" aria-label="Ask">
            <Eyebrow icon={<IconBars size={14} />}>Ask the analyst</Eyebrow>
            <form onSubmit={submit}>
              <label htmlFor="q" className="sr-only">Your question</label>
              <textarea
                id="q" value={question} maxLength={meta.max_question_chars}
                onChange={(e) => setQuestion(e.target.value)}
                onKeyDown={(e) => { if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) submit(e); }}
                placeholder={live ? "Ask a question about the data (Ctrl+Enter to send)" : "Pick a recorded question below"}
                disabled={!live}
              />
              <div className="actions">
                <button className="btn" type="submit" disabled={!live || running || !question.trim()}>Ask</button>
                {!live && <span className="hint">This demo replays recorded runs. Live questions work when run locally with an API key.</span>}
              </div>
            </form>
          </section>

          <section className="card" aria-label="Suggested questions">
            <Eyebrow icon={<IconDb />}>{live ? "Recorded runs to replay" : "Recorded questions"}</Eyebrow>
            <QuestionPager
              items={meta.suggested} current={run?.id} disabled={running} live={live}
              onPick={(s) => { setQuestion(s.question); stream({ run_id: s.id }, s.question); }}
            />
          </section>

          {live && meta.local_mode && <Connections meta={meta} onChange={reload} />}
        </aside>

        <main className="main">
          {problem && <p className="callout bad" role="alert">{problem}</p>}
          {run?.error && <p className="callout bad" role="alert">The run stopped: {run.error.message}</p>}

          {!run && !problem && (
            <section className="card empty" aria-label="Welcome">
              <div className="art" aria-hidden="true"><IconBars size={36} /></div>
              <h2>Ask a question. See how it got the answer.</h2>
              <p>Every answer comes with the exact query, the rows it used, the checks it ran, and a confidence that says how often answers like it were right on questions it had never been tuned on.</p>
              <div className="features">
                <div className="feature"><b>Auditable</b><span>The query, in SQL and in words, with the rows behind the answer.</span></div>
                <div className="feature"><b>Calibrated</b><span>Confidence measured against held-out questions, not just stated.</span></div>
                <div className="feature"><b>Honest</b><span>It declines or holds back what it cannot stand behind.</span></div>
              </div>
            </section>
          )}

          {run && (
            <>
              <div className="hero-grid">
                <section className="card" aria-label="Answer" aria-live="polite">
                  <Eyebrow icon={<IconShield />}>Answer</Eyebrow>
                  {run.answer ? (
                    <AnswerCard a={run.answer} run={run} meter={meta.meter} />
                  ) : (
                    <>
                      <p className="answer-q">{run.question}</p>
                      {running && !run.error && <p className="muted">Working…</p>}
                    </>
                  )}
                </section>
                {run.confidence && run.answer && run.answer.status !== "declined" ? (
                  <ConfidenceMeter c={run.confidence} meter={meta.meter} />
                ) : (
                  <section className="card" aria-label="Meter" style={{ display: run.answer ? "none" : undefined }}>
                    <Eyebrow icon={<IconGauge />}>Confidence</Eyebrow>
                    <p className="muted">Shown once the analyst has answered.</p>
                  </section>
                )}
              </div>

              {run.rows && <ResultTable rows={run.rows} />}
              {run.chart && run.rows && <ChartView spec={run.chart.spec} columns={run.chart.columns} rows={run.rows.rows} />}
              <div className="row2">
                <Timeline run={run} running={running} />
                {run.sql ? <SqlView sql={run.sql} /> : <div />}
              </div>
              <div className={run.checks.length && run.statistics ? "row-even" : ""}>
                <Checks checks={run.checks} />
                {run.statistics && <StatisticsView s={run.statistics} />}
              </div>
            </>
          )}
        </main>
      </div>
    </div>
  );
}
