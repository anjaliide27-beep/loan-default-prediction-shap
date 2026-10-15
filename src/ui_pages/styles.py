"""
styles.py
---------
Custom CSS styling, color palettes, and card components for the Streamlit dashboard.
"""

def get_custom_css() -> str:
    return """
    <style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Primary Container Styling */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }

    /* Modern Card Container */
    .metric-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 1.25rem 1.5rem;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.05);
        margin-bottom: 1rem;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .metric-card:hover {
        box-shadow: 0 4px 12px rgba(15, 23, 42, 0.08);
    }

    .metric-title {
        font-size: 0.85rem;
        font-weight: 600;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.35rem;
    }

    .metric-value {
        font-size: 1.75rem;
        font-weight: 700;
        color: #0F172A;
    }

    .metric-sub {
        font-size: 0.8rem;
        color: #94A3B8;
        margin-top: 0.25rem;
    }

    /* Hero Banner */
    .hero-banner {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 50%, #0F766E 100%);
        border-radius: 16px;
        padding: 2.25rem 2.5rem;
        color: #FFFFFF;
        margin-bottom: 2rem;
        box-shadow: 0 10px 25px -5px rgba(15, 23, 42, 0.2);
    }

    .hero-title {
        font-size: 2.1rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        margin-bottom: 0.5rem;
        color: #FFFFFF;
    }

    .hero-sub {
        font-size: 1.05rem;
        color: #94A3B8;
        line-height: 1.5;
        max-width: 800px;
    }

    /* Badges */
    .badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    .badge-teal {
        background-color: #CCFBF1;
        color: #0F766E;
    }

    .badge-amber {
        background-color: #FEF3C7;
        color: #B45309;
    }

    .badge-rose {
        background-color: #FFE4E6;
        color: #BE123C;
    }

    /* Risk Result Box */
    .risk-box {
        border-radius: 12px;
        padding: 1.5rem;
        margin-top: 1rem;
        margin-bottom: 1rem;
        border: 2px solid;
    }

    /* Disclaimers */
    .disclaimer-box {
        background-color: #F8FAFC;
        border-left: 4px solid #0EA5E9;
        padding: 1rem 1.25rem;
        border-radius: 0 8px 8px 0;
        font-size: 0.85rem;
        color: #475569;
        margin-top: 1.5rem;
        margin-bottom: 1.5rem;
    }

    /* Section headers */
    .section-header {
        font-size: 1.4rem;
        font-weight: 700;
        color: #0F172A;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #F1F5F9;
        padding-bottom: 0.5rem;
    }

    /* Code & Mono */
    code {
        font-family: 'JetBrains Mono', monospace !important;
        background-color: #F1F5F9;
        padding: 0.15rem 0.4rem;
        border-radius: 4px;
        color: #0F766E;
        font-size: 0.88em;
    }

    /* Custom sidebar header */
    .sidebar-header {
        padding: 1rem 0;
        text-align: center;
        border-bottom: 1px solid #E2E8F0;
        margin-bottom: 1.5rem;
    }
    </style>
    """


def render_metric_card(title: str, value: str, sub: str = "", border_color: str = "#E2E8F0") -> str:
    return f"""
    <div class="metric-card" style="border-left: 4px solid {border_color};">
        <div class="metric-title">{title}</div>
        <div class="metric-value">{value}</div>
        {f'<div class="metric-sub">{sub}</div>' if sub else ''}
    </div>
    """


def render_disclaimer(text: str) -> str:
    return f"""
    <div class="disclaimer-box">
        <strong>Academic & Regulatory Notice:</strong> {text}
    </div>
    """
