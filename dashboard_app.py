"""Root Streamlit entrypoint for the trader dashboard.

Run with:
    streamlit run dashboard_app.py -- --config configs/trader/weekend_vol_btc.yaml
"""

from trader.dashboard import *  # noqa: F401,F403