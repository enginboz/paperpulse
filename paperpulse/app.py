"""
PaperPulse Flask application.

Serves the daily paper digest as an HTMX-powered HTML fragment,
designed to be embedded as a widget in a dashboard.

Endpoints:
    GET /              — full page (for standalone use)
    GET /widgets/papers — HTML fragment for HTMX dashboard integration
"""

from datetime import date

from flask import Flask, render_template_string
from paperpulse.db import get_latest_digest, get_digest_for_date

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------------

WIDGET_TEMPLATE = """
<div id="papers-widget"
     hx-get="/widgets/papers"
     hx-trigger="every 21600s"
     hx-swap="outerHTML">
  <style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600&family=JetBrains+Mono:wght@300;400&display=swap');

    .pp { font-family: 'Syne', sans-serif; width: 100%; box-sizing: border-box; padding: 1.25rem 1.5rem; }

    .pp-head {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 1.25rem;
      padding-bottom: 0.75rem;
      border-bottom: 1px solid rgba(255,255,255,0.06);
    }

    .pp-wordmark {
      font-family: 'JetBrains Mono', monospace;
      font-size: 0.6rem;
      font-weight: 400;
      letter-spacing: 0.22em;
      text-transform: uppercase;
      color: #5a9e82;
    }

    .pp-meta {
      font-family: 'JetBrains Mono', monospace;
      font-size: 0.55rem;
      letter-spacing: 0.06em;
      color: rgba(255,255,255,0.18);
    }

    .pp-item {
      padding: 0.9rem 0;
      border-bottom: 1px solid rgba(255,255,255,0.04);
    }

    .pp-item:last-child { border-bottom: none; padding-bottom: 0; }

    .pp-top {
      display: flex;
      align-items: baseline;
      gap: 0.6rem;
      margin-bottom: 0.3rem;
    }

    .pp-num {
      font-family: 'JetBrains Mono', monospace;
      font-size: 0.55rem;
      color: #5a9e82;
      opacity: 0.6;
      flex-shrink: 0;
    }

    .pp-src {
      font-family: 'JetBrains Mono', monospace;
      font-size: 0.55rem;
      letter-spacing: 0.07em;
      text-transform: uppercase;
      color: rgba(255,255,255,0.22);
    }

    .pp-link {
      display: block;
      font-size: 0.82rem;
      font-weight: 500;
      line-height: 1.45;
      color: rgba(255,255,255,0.88);
      text-decoration: none;
      margin-bottom: 0.4rem;
      transition: color 0.15s ease;
    }

    .pp-link:hover { color: #5a9e82; }

    .pp-why {
      font-size: 0.72rem;
      line-height: 1.55;
      color: rgba(255,255,255,0.32);
      font-weight: 400;
      margin: 0;
    }

    .pp-empty {
      font-family: 'JetBrains Mono', monospace;
      font-size: 0.7rem;
      color: rgba(255,255,255,0.18);
      padding: 0.5rem 0;
    }
  </style>

  <div class="pp">
    <div class="pp-head">
      <span class="pp-wordmark">paperpulse</span>
      <span class="pp-meta">{{ digest_date }}</span>
    </div>

    {% if papers %}
      {% for paper in papers %}
      <div class="pp-item">
        <div class="pp-top">
          <span class="pp-num">0{{ loop.index }}</span>
          <span class="pp-src">{{ paper.journal }}</span>
        </div>
        <a class="pp-link" href="{{ paper.url }}" target="_blank">{{ paper.title }}</a>
        <p class="pp-why">{{ paper.reason }}</p>
      </div>
      {% endfor %}
    {% else %}
      <p class="pp-empty">no digest yet — run the pipeline first.</p>
    {% endif %}
  </div>
</div>
"""

PAGE_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>PaperPulse</title>
  <script src="https://unpkg.com/htmx.org@2.0.0"></script>
  <style>
    *, *::before, *::after {{ box-sizing: border-box; }}
    body {{
      background: #0e0e0e;
      margin: 0;
      padding: 3rem 1rem;
      min-height: 100vh;
      display: flex;
      justify-content: center;
    }}
    .wrap {{ width: 100%; max-width: 480px; }}
  </style>
</head>
<body>
  <div class="wrap">{widget_html}</div>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_digest_context() -> dict:
    """Fetch the latest digest from the database and return template context."""
    digest = get_latest_digest()
    if digest:
        return {
            "papers": digest.papers,
            "digest_date": digest.digest_date.isoformat(),
        }
    return {
        "papers": [],
        "digest_date": date.today().isoformat(),
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/widgets/papers")
def widget_papers():
    """
    HTMX endpoint — returns just the widget HTML fragment.
    This is what the dashboard polls every 6 hours.
    """
    context = _get_digest_context()
    return render_template_string(WIDGET_TEMPLATE, **context)


@app.route("/digest/<date_str>")
def digest_for_date(date_str: str):
    """
    View the digest for a specific date.
    Usage: /digest/2026-04-28
    """
    try:
        target = date.fromisoformat(date_str)
    except ValueError:
        return f"Invalid date format: '{date_str}'. Use YYYY-MM-DD.", 400

    digest = get_digest_for_date(target)
    if not digest:
        return f"No digest found for {date_str}.", 404

    widget_html = render_template_string(WIDGET_TEMPLATE,
        papers=digest.papers,
        digest_date=digest.digest_date.isoformat(),
    )
    return PAGE_TEMPLATE.format(widget_html=widget_html)


@app.route("/")
def index():
    """
    Standalone page — renders the full HTML page with the widget embedded.
    Useful for testing the widget outside of a dashboard.
    """
    context = _get_digest_context()
    widget_html = render_template_string(WIDGET_TEMPLATE, **context)
    return PAGE_TEMPLATE.format(widget_html=widget_html)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5001)