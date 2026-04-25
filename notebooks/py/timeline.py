"""HTML timeline helpers for CAL notebook simulations."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np

from max_plus import EPS, INF


def _finite(value: float) -> bool:
    return value != EPS and value != INF and math.isfinite(value)


def _reaction_names(count: int, names: Sequence[str] | None = None) -> list[str]:
    if names is not None:
        return list(names)
    return [chr(ord("A") + index) for index in range(count)]


def extract_timeline_data(
    x_history: Sequence[np.ndarray],
    logical_time_history: Sequence[float] | None = None,
    execution_time_history: Sequence[Sequence[float]] | None = None,
    reaction_names: Sequence[str] | None = None,
) -> dict[str, list[dict[str, float]]]:
    """Convert row-vector simulation history into per-reaction timeline events."""
    if not x_history:
        return {}

    names = _reaction_names(x_history[0].shape[1], reaction_names)
    reactions: dict[str, list[dict[str, float]]] = {name: [] for name in names}

    for tag_index, matrix in enumerate(x_history):
        row = matrix[0]
        logical_time = logical_time_history[tag_index] if logical_time_history and tag_index < len(logical_time_history) else 0

        for reaction_index, time in enumerate(row):
            if not _finite(float(time)):
                continue

            exec_time = 0
            if execution_time_history and tag_index < len(execution_time_history):
                exec_time = execution_time_history[tag_index][reaction_index]

            reactions[names[reaction_index]].append(
                {
                    "tagIndex": tag_index,
                    "time": float(time),
                    "logicalTime": float(logical_time),
                    "lag": float(time - logical_time),
                    "execTime": float(exec_time),
                }
            )

    return reactions


def generate_html_timeline(
    x_history: Sequence[np.ndarray],
    logical_time_history: Sequence[float] | None = None,
    execution_time_history: Sequence[Sequence[float]] | None = None,
    reaction_names: Sequence[str] | None = None,
    title: str = "CAL Timeline",
    subtitle: str = "Timeline showing reaction firing times, logical times, lag, and execution times.",
) -> str:
    """Generate a Vis.js HTML timeline like the TypeScript Delay notebook."""
    reaction_data = extract_timeline_data(x_history, logical_time_history, execution_time_history, reaction_names)
    colors = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12", "#9b59b6", "#1abc9c", "#e67e22", "#34495e"]

    items: list[dict[str, object]] = []
    groups: list[dict[str, object]] = []

    for index, (name, events) in enumerate(reaction_data.items()):
        if not events:
            continue
        color = colors[index % len(colors)]
        groups.append(
            {
                "id": name,
                "content": f"Reaction {name}",
                "style": f"background-color: white; color: #2c3e50; border-left: 4px solid {color};",
            }
        )

        for event in events:
            content = (
                f"k = {event['tagIndex']}<br/>"
                f"x = {event['time']:.3f}<br/>"
                f"t = {event['logicalTime']:.3f}<br/>"
                f"lag = {event['lag']:.3f}<br/>"
                f"e = {event['execTime']:.3f}"
            )
            items.append(
                {
                    "id": f"{name}_{event['tagIndex']}",
                    "content": content,
                    "start": event["time"] * 1000,
                    "group": name,
                    "title": (
                        f"Tag {event['tagIndex']}: Reaction {name} fires at {event['time']:.3f} "
                        f"(logical time: {event['logicalTime']:.3f}, lag: {event['lag']:.3f})"
                    ),
                    "style": f"background-color: {color}; border-color: {color}; color: white;",
                }
            )

    return f"""<!DOCTYPE HTML>
<html>
<head>
  <title>{title}</title>
  <style type="text/css">
    body, html {{
      font-family: sans-serif;
    }}
    .header {{
      background-color: #f8f9fa;
      padding: 20px;
      border-bottom: 2px solid #dee2e6;
      margin-bottom: 20px;
    }}
    .header h1 {{
      margin: 0;
      color: #2c3e50;
    }}
    .header p {{
      margin: 5px 0 0 0;
      color: #6c757d;
    }}
  </style>
  <script src="https://unpkg.com/vis-timeline@latest/standalone/umd/vis-timeline-graph2d.min.js"></script>
  <link href="https://unpkg.com/vis-timeline@latest/styles/vis-timeline-graph2d.min.css" rel="stylesheet" type="text/css" />
</head>
<body>
<div class="header">
  <h1>{title}</h1>
  <p>{subtitle}</p>
</div>
<div id="visualization"></div>

<script type="text/javascript">
  var container = document.getElementById('visualization');
  var items = new vis.DataSet({json.dumps(items)});
  var groups = new vis.DataSet({json.dumps(groups)});
  var options = {{
    format: {{
      minorLabels: function(date, scale, step) {{
        const seconds = (typeof date === 'number' ? date : new Date(date).getTime()) / 1000;
        return seconds.toFixed(1) + 's';
      }},
      majorLabels: function(date, scale, step) {{
        const seconds = (typeof date === 'number' ? date : new Date(date).getTime()) / 1000;
        return seconds.toFixed(2) + 's';
      }}
    }}
  }};
  var timeline = new vis.Timeline(container, items, groups, options);
</script>
</body>
</html>"""


def write_timeline_to_file(
    x_history: Sequence[np.ndarray],
    logical_time_history: Sequence[float] | None = None,
    execution_time_history: Sequence[Sequence[float]] | None = None,
    reaction_names: Sequence[str] | None = None,
    filename: str = "../html/reaction-timeline.html",
    title: str = "CAL Timeline",
    subtitle: str = "Timeline showing reaction firing times, logical times, lag, and execution times.",
) -> str:
    """Write a Vis.js timeline HTML file and return its path."""
    html = generate_html_timeline(
        x_history,
        logical_time_history,
        execution_time_history,
        reaction_names,
        title,
        subtitle,
    )
    path = Path(filename).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")
    print(f"Timeline saved to: {path}")
    return str(path)
