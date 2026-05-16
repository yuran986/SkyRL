from __future__ import annotations

import argparse
import html
import json
import os
import sys
from pathlib import Path
from typing import Any


if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[3]))

from examples.train_integrations.arc_agi3.ft09_oracle_solution import (  # noqa: E402
    run_ft09_solution,
    write_trace_jsonl,
)


PALETTE = {
    0: "#ffffff",
    1: "#f7f7f7",
    2: "#d8d8d8",
    3: "#9a9a9a",
    4: "#2b2b2b",
    5: "#050505",
    6: "#ff00b8",
    7: "#ff8be8",
    8: "#e64141",
    9: "#1f5eff",
    10: "#7db5ff",
    11: "#f5dc3f",
    12: "#ff9d24",
    13: "#7c1f1f",
    14: "#39a845",
    15: "#7a48c7",
}


def read_trace_jsonl(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    with path.expanduser().open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSONL: {exc}") from exc
    return events


def build_trace(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.trace_jsonl and Path(args.trace_jsonl).expanduser().exists() and not args.regenerate:
        return read_trace_jsonl(Path(args.trace_jsonl))

    trace: list[dict[str, Any]] = []
    run_ft09_solution(
        environments_dir=args.environments_dir,
        operation_mode=args.operation_mode,
        legal_only=args.legal_only,
        verbose=not args.quiet,
        max_level_attempts=args.max_level_attempts,
        trace=trace,
    )
    if args.trace_jsonl:
        write_trace_jsonl(trace, args.trace_jsonl)
    return trace


def html_document(events: list[dict[str, Any]], title: str) -> str:
    data_json = json.dumps(events, ensure_ascii=False)
    palette_json = json.dumps(PALETTE)
    safe_title = html.escape(title)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{safe_title}</title>
  <style>
    :root {{
      --bg: #101214;
      --panel: #181c20;
      --panel-2: #22272e;
      --text: #ece7dc;
      --muted: #a8a096;
      --line: #343b43;
      --accent: #f0b84b;
      --good: #61c36d;
      --fix: #ff6b6b;
      --blue: #69a7ff;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background:
        radial-gradient(circle at 18% 12%, rgba(240,184,75,.18), transparent 28rem),
        radial-gradient(circle at 92% 8%, rgba(105,167,255,.12), transparent 24rem),
        linear-gradient(135deg, #0d1012, #171311 62%, #1a1814);
      color: var(--text);
      font-family: ui-sans-serif, "Avenir Next", "Segoe UI", sans-serif;
    }}
    header {{
      padding: 24px 28px 16px;
      border-bottom: 1px solid rgba(255,255,255,.08);
    }}
    h1 {{ margin: 0 0 8px; font-size: 28px; letter-spacing: .02em; }}
    .subtitle {{ color: var(--muted); font-size: 14px; }}
    .layout {{
      display: grid;
      grid-template-columns: minmax(260px, 370px) minmax(560px, 1fr);
      gap: 18px;
      padding: 18px;
    }}
    .panel {{
      background: rgba(24,28,32,.92);
      border: 1px solid rgba(255,255,255,.08);
      border-radius: 18px;
      box-shadow: 0 18px 60px rgba(0,0,0,.24);
      overflow: hidden;
    }}
    .timeline {{ max-height: calc(100vh - 122px); overflow: auto; padding: 10px; }}
    .event {{
      width: 100%;
      border: 1px solid transparent;
      border-radius: 13px;
      margin: 7px 0;
      padding: 10px 11px;
      color: inherit;
      background: var(--panel-2);
      text-align: left;
      cursor: pointer;
    }}
    .event:hover {{ border-color: rgba(240,184,75,.45); }}
    .event.active {{
      border-color: var(--accent);
      background: linear-gradient(135deg, rgba(240,184,75,.18), rgba(34,39,46,.96));
    }}
    .event-title {{ display: flex; justify-content: space-between; gap: 8px; font-weight: 700; }}
    .event-meta {{ margin-top: 5px; color: var(--muted); font-size: 12px; }}
    .stage {{ padding: 18px; }}
    .topbar {{ display: flex; gap: 10px; flex-wrap: wrap; align-items: center; margin-bottom: 14px; }}
    button.control {{
      border: 1px solid var(--line);
      color: var(--text);
      background: #20262c;
      border-radius: 11px;
      padding: 8px 12px;
      cursor: pointer;
    }}
    button.control:hover {{ border-color: var(--accent); }}
    input[type=range] {{ flex: 1 1 220px; accent-color: var(--accent); }}
    .badges {{ display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 14px; }}
    .badge {{
      border: 1px solid var(--line);
      background: rgba(255,255,255,.04);
      border-radius: 999px;
      padding: 6px 10px;
      font-size: 13px;
    }}
    .badge.good {{ border-color: rgba(97,195,109,.45); color: var(--good); }}
    .badge.fix {{ border-color: rgba(255,107,107,.5); color: var(--fix); }}
    .badge.click {{ border-color: rgba(105,167,255,.5); color: var(--blue); }}
    .frame-wrap {{
      display: grid;
      grid-template-columns: minmax(360px, 640px) minmax(260px, 1fr);
      gap: 18px;
      align-items: start;
    }}
    canvas {{
      width: min(640px, 100%);
      image-rendering: pixelated;
      background: #000;
      border: 1px solid var(--line);
      border-radius: 12px;
    }}
    .details {{
      background: #111417;
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 14px;
      min-height: 360px;
    }}
    .details h2 {{ margin: 0 0 8px; font-size: 18px; }}
    .kv {{ display: grid; grid-template-columns: 108px 1fr; gap: 7px 10px; font-size: 14px; }}
    .key {{ color: var(--muted); }}
    pre {{
      margin: 14px 0 0;
      white-space: pre-wrap;
      word-break: break-word;
      color: #d8d3c9;
      background: #0b0d0f;
      border-radius: 10px;
      padding: 12px;
      max-height: 280px;
      overflow: auto;
    }}
    @media (max-width: 900px) {{
      .layout, .frame-wrap {{ grid-template-columns: 1fr; }}
      .timeline {{ max-height: 260px; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>{safe_title}</h1>
    <div class="subtitle">Step through the deterministic ft09 oracle: legal clicks, internal fixes, level advances, and exact frame snapshots.</div>
  </header>
  <main class="layout">
    <aside class="panel timeline" id="timeline"></aside>
    <section class="panel stage">
      <div class="topbar">
        <button class="control" id="prev">Prev</button>
        <button class="control" id="next">Next</button>
        <input id="scrubber" type="range" min="0" max="0" value="0" />
      </div>
      <div class="badges" id="badges"></div>
      <div class="frame-wrap">
        <canvas id="frame" width="640" height="640"></canvas>
        <div class="details">
          <h2 id="eventName">Event</h2>
          <div class="kv" id="kv"></div>
          <pre id="raw"></pre>
        </div>
      </div>
    </section>
  </main>
  <script id="events" type="application/json">{data_json}</script>
  <script>
    const events = JSON.parse(document.getElementById("events").textContent);
    const palette = {palette_json};
    let index = 0;

    const timeline = document.getElementById("timeline");
    const canvas = document.getElementById("frame");
    const ctx = canvas.getContext("2d");
    const scrubber = document.getElementById("scrubber");
    const badges = document.getElementById("badges");
    const eventName = document.getElementById("eventName");
    const kv = document.getElementById("kv");
    const raw = document.getElementById("raw");

    function compactReason(event) {{
      if (event.reason) return event.reason;
      if (event.event === "click") return event.sprite_name || "click";
      return event.event;
    }}

    function eventClass(event) {{
      if (event.event === "internal_fix") return "fix";
      if (event.event === "click") return "click";
      if (event.event === "final") return "good";
      return "";
    }}

    function renderTimeline() {{
      timeline.innerHTML = "";
      events.forEach((event, i) => {{
        const button = document.createElement("button");
        button.className = `event ${{i === index ? "active" : ""}}`;
        button.innerHTML = `
          <div class="event-title">
            <span>#${{i}} ${{event.event}}</span>
            <span>L${{event.level_index ?? "-"}}</span>
          </div>
          <div class="event-meta">${{compactReason(event)}} · completed=${{event.levels_completed ?? 0}}</div>
        `;
        button.onclick = () => select(i);
        timeline.appendChild(button);
      }});
    }}

    function drawFrame(event) {{
      const frame = event.frame || [];
      const rows = frame.length || 64;
      const cols = frame[0]?.length || 64;
      const cellW = canvas.width / cols;
      const cellH = canvas.height / rows;
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      for (let y = 0; y < rows; y++) {{
        for (let x = 0; x < cols; x++) {{
          const value = frame[y]?.[x] ?? 0;
          ctx.fillStyle = palette[value] || "#00ffff";
          ctx.fillRect(x * cellW, y * cellH, Math.ceil(cellW), Math.ceil(cellH));
        }}
      }}
      ctx.strokeStyle = "rgba(255,255,255,.12)";
      ctx.lineWidth = 1;
      for (let i = 0; i <= 64; i += 4) {{
        const p = i * (canvas.width / 64);
        ctx.beginPath();
        ctx.moveTo(p, 0);
        ctx.lineTo(p, canvas.height);
        ctx.moveTo(0, p);
        ctx.lineTo(canvas.width, p);
        ctx.stroke();
      }}
      if (event.display) {{
        const x = event.display.x * (canvas.width / 64);
        const y = event.display.y * (canvas.height / 64);
        ctx.strokeStyle = "#f0b84b";
        ctx.lineWidth = 5;
        ctx.beginPath();
        ctx.arc(x + cellW / 2, y + cellH / 2, 22, 0, Math.PI * 2);
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(x - 18, y + cellH / 2);
        ctx.lineTo(x + 28, y + cellH / 2);
        ctx.moveTo(x + cellW / 2, y - 18);
        ctx.lineTo(x + cellW / 2, y + 28);
        ctx.stroke();
      }}
      if (event.grid && event.event === "internal_fix") {{
        const x = (event.grid.x * 2) * (canvas.width / 64);
        const y = (event.grid.y * 2) * (canvas.height / 64);
        ctx.strokeStyle = "#ff6b6b";
        ctx.lineWidth = 5;
        ctx.strokeRect(x, y, 60, 60);
      }}
    }}

    function renderBadges(event) {{
      const cls = eventClass(event);
      const items = [
        [`event`, event.event, cls],
        [`level`, `${{event.level_index}} ${{event.level_name || ""}}`, ""],
        [`state`, event.game_state, event.game_state === "WIN" ? "good" : ""],
        [`completed`, event.levels_completed, event.levels_completed >= 6 ? "good" : ""],
      ];
      if (event.sprite_name) items.push(["sprite", event.sprite_name, cls]);
      if (event.grid) items.push(["grid", `(${{event.grid.x}}, ${{event.grid.y}})`, cls]);
      if (event.display) items.push(["display", `(${{event.display.x}}, ${{event.display.y}})`, "click"]);
      badges.innerHTML = items.map(([key, value, extra]) =>
        `<span class="badge ${{extra}}">${{key}}: ${{value ?? "-"}}</span>`
      ).join("");
    }}

    function renderDetails(event) {{
      eventName.textContent = `#${{event.index}} ${{event.event}}`;
      const entries = [
        ["reason", event.reason],
        ["advanced", event.advanced_level],
        ["old center", event.old_center],
        ["new center", event.new_center],
        ["obs state", event.observation_state],
        ["obs completed", event.observation_levels_completed],
      ].filter(([, value]) => value !== undefined && value !== null);
      kv.innerHTML = entries.map(([key, value]) =>
        `<div class="key">${{key}}</div><div>${{String(value)}}</div>`
      ).join("");
      const clone = {{...event}};
      delete clone.frame;
      raw.textContent = JSON.stringify(clone, null, 2);
    }}

    function select(nextIndex) {{
      index = Math.max(0, Math.min(events.length - 1, nextIndex));
      scrubber.value = String(index);
      renderTimeline();
      const active = events[index] || {{}};
      drawFrame(active);
      renderBadges(active);
      renderDetails(active);
    }}

    document.getElementById("prev").onclick = () => select(index - 1);
    document.getElementById("next").onclick = () => select(index + 1);
    scrubber.max = String(Math.max(events.length - 1, 0));
    scrubber.oninput = () => select(Number(scrubber.value));
    window.addEventListener("keydown", (event) => {{
      if (event.key === "ArrowLeft") select(index - 1);
      if (event.key === "ArrowRight") select(index + 1);
    }});
    select(0);
  </script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a standalone ft09 oracle visualization.")
    parser.add_argument(
        "--environments_dir",
        default=os.getenv("ARC_AGI3_ENVIRONMENTS_DIR", "/home/users/yz1051/rlm/environment_files"),
    )
    parser.add_argument("--operation_mode", default=os.getenv("OPERATION_MODE", "OFFLINE"))
    parser.add_argument("--legal-only", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--max-level-attempts", type=int, default=16)
    parser.add_argument("--trace-jsonl", help="Read/write the oracle trace JSONL at this path.")
    parser.add_argument("--regenerate", action="store_true", help="Regenerate --trace-jsonl if it exists.")
    parser.add_argument(
        "-o",
        "--output",
        default="ft09_oracle_solution.html",
        help="Output HTML path.",
    )
    args = parser.parse_args()

    events = build_trace(args)
    if not events:
        raise RuntimeError("No trace events were produced.")

    output_path = Path(args.output).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html_document(events, "ft09 Oracle Solution Trace"), encoding="utf-8")
    print(f"wrote {output_path} with {len(events)} events")


if __name__ == "__main__":
    main()
