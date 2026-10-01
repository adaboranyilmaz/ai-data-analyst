import { useState } from "react";
import { Eyebrow, IconDb } from "./components";
import type { Meta } from "./types";

/** Local mode only: point the analyst at your own PostgreSQL, read-only role required. */
export function Connections({ meta, onChange }: { meta: Meta; onChange: () => void }) {
  const [form, setForm] = useState({ host: "127.0.0.1", port: "5432", dbname: "", user: "", password: "", schema: "public" });
  const [problems, setProblems] = useState<string[]>([]);
  const [note, setNote] = useState<string | null>(null);
  const set = (k: keyof typeof form) => (e: React.ChangeEvent<HTMLInputElement>) => setForm({ ...form, [k]: e.target.value });

  const connect = async (e: React.FormEvent) => {
    e.preventDefault();
    setProblems([]); setNote(null);
    const res = await fetch("/connections", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ...form, port: Number(form.port) }),
    });
    const j = await res.json();
    if (res.ok) {
      setNote(`Connected: ${j.tables} tables. ${j.warnings.join(" ")} Confidence is not calibrated for this database.`);
      setForm({ ...form, password: "" });
      onChange();
    } else {
      setProblems(j.detail?.refused ?? [typeof j.detail === "string" ? j.detail : "The connection was refused."]);
    }
  };
  const disconnect = async () => {
    await fetch("/connections", { method: "DELETE" });
    setNote("Disconnected: questions use the demo database again.");
    onChange();
  };

  return (
    <section className="card conn" aria-label="Your own database">
      <Eyebrow icon={<IconDb />}>Your own PostgreSQL</Eyebrow>
      {meta.connection ? (
        <p>
          Connected to <strong>{String(meta.connection.dbname)}</strong> ({String(meta.connection.schema)}){" "}
          <button className="btn ghost" onClick={disconnect}>Disconnect</button>
        </p>
      ) : (
        <form onSubmit={connect}>
          <p className="muted" style={{ marginBottom: 10, fontSize: "0.86rem" }}>The role must only be able to read; it is checked before anything runs.</p>
          {(["host", "port", "dbname", "user", "password", "schema"] as const).map((k) => (
            <div key={k} className="field">
              <label htmlFor={`c-${k}`}>{k}</label>
              <input id={`c-${k}`} type={k === "password" ? "password" : "text"} value={form[k]} onChange={set(k)} autoComplete="off" />
            </div>
          ))}
          <button className="btn" type="submit">Connect</button>
        </form>
      )}
      {problems.length > 0 && <ul className="callout bad" role="alert">{problems.map((p, i) => <li key={i}>{p}</li>)}</ul>}
      {note && <p className="note" role="status">{note}</p>}
    </section>
  );
}
