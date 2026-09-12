import sqlite3
import os
import sys
import re
from typing import List, Dict, Any

# Ensure project root is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import config

class SQLEngineRetriever:
    """Structured SQL database retriever for financial metrics & performance data."""

    def __init__(self, db_path: str = None):
        self.db_path = db_path or config.SQLITE_DB_PATH
        self._initialize_database()

    def _initialize_database(self):
        """Creates tables and seeds financial metric records if not present."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS financial_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            company TEXT NOT NULL,
            fiscal_year TEXT NOT NULL,
            revenue_crores REAL,
            revenue_usd_billion REAL,
            ebitda_crores REAL,
            net_profit_crores REAL,
            net_debt_crores REAL,
            ev_market_share_pct REAL,
            key_highlights TEXT
        );
        """)

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS subsidiaries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            company TEXT NOT NULL,
            unit_name TEXT NOT NULL,
            sector TEXT,
            revenue_contribution_pct REAL,
            key_strategy TEXT
        );
        """)

        # Check if already seeded
        cursor.execute("SELECT COUNT(*) FROM financial_metrics")
        if cursor.fetchone()[0] == 0:
            print(f"[SQL Engine] Seeding initial financial records into {self.db_path}...", flush=True)
            
            # Seed Tata Motors metrics
            tata_metrics = [
                ('Tata Motors', 'FY22', 278454.0, 34.2, 34650.0, -11441.0, 48700.0, 65.0, 'EV transition initiated, Nexon EV leader'),
                ('Tata Motors', 'FY23', 345967.0, 42.1, 43200.0, 2414.0, 43700.0, 72.0, 'Returned to profitability, JLR recovery'),
                ('Tata Motors', 'FY24', 437939.0, 52.6, 62800.0, 31807.0, 1000.0, 73.0, 'Record profit & EBITDA, net debt reduced near zero'),
                ('Tata Motors', 'FY25', 465000.0, 55.8, 67500.0, 35200.0, 0.0, 75.0, 'Strategic de-merger into CV & PV/JLR separate entities')
            ]
            
            # Seed Reliance Industries metrics
            ril_metrics = [
                ('Reliance Industries', 'FY22', 792756.0, 95.0, 125687.0, 67845.0, 116300.0, 0.0, '5G roll-out launch, Retail expansion'),
                ('Reliance Industries', 'FY23', 976524.0, 118.6, 154263.0, 74088.0, 110200.0, 0.0, 'Consumer business EBITDA grew 30%+ YoY'),
                ('Reliance Industries', 'FY24', 1000122.0, 119.9, 178677.0, 79020.0, 116281.0, 0.0, 'Record EBITDA across O2C, Retail, and Jio'),
                ('Reliance Industries', 'FY25', 1060000.0, 127.0, 192000.0, 84500.0, 105000.0, 0.0, 'Green Energy Gigafactories commissioning in Jamnagar')
            ]

            cursor.executemany("""
            INSERT INTO financial_metrics (company, fiscal_year, revenue_crores, revenue_usd_billion, ebitda_crores, net_profit_crores, net_debt_crores, ev_market_share_pct, key_highlights)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, tata_metrics + ril_metrics)

            subsidiary_data = [
                ('Tata Motors', 'Jaguar Land Rover (JLR)', 'Automotive Luxury', 68.0, 'Reimagine Strategy, £15B electrification investment'),
                ('Tata Motors', 'Tata Passenger Electric Mobility (TPEM)', 'EV Passenger', 12.0, 'Market leader in Indian EVs, TPG Rise Climate backing'),
                ('Tata Motors', 'Commercial Vehicles (CV)', 'Trucks & Buses', 20.0, 'Fleet Edge telematics, Hydrogen & LNG zero emission trucks'),
                ('Reliance Industries', 'Reliance Retail', 'Retail', 31.0, 'Omnichannel dominance with 18,000+ stores'),
                ('Reliance Industries', 'Jio Platforms', 'Telecom & Digital', 14.0, '5G leader in India, Cloud & AI services'),
                ('Reliance Industries', 'Oil to Chemicals (O2C)', 'Energy & Petrochem', 50.0, 'Downstream integration & transition to Green Hydrogen')
            ]

            cursor.executemany("""
            INSERT INTO subsidiaries (company, unit_name, sector, revenue_contribution_pct, key_strategy)
            VALUES (?, ?, ?, ?, ?)
            """, subsidiary_data)

            conn.commit()
            print("[SQL Engine] Financial tables seeded successfully.", flush=True)

        conn.close()

    def query(self, query_text: str) -> List[Dict[str, Any]]:
        """Interprets natural query into SQL query or structured lookup."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        query_lower = query_text.lower()
        results = []

        try:
            # Flexible company & fiscal year detection
            target_company = None
            if "tata" in query_lower:
                target_company = "Tata Motors"
            elif "reliance" in query_lower or "ril" in query_lower:
                target_company = "Reliance Industries"

            target_years = []
            if "2024" in query_lower or "fy24" in query_lower:
                target_years.append("FY24")
            if "2025" in query_lower or "fy25" in query_lower:
                target_years.append("FY25")

            if target_company and target_years:
                placeholders = ",".join(["?"] * len(target_years))
                sql = f"SELECT * FROM financial_metrics WHERE company = ? AND fiscal_year IN ({placeholders}) ORDER BY fiscal_year DESC"
                params = [target_company] + target_years
                cursor.execute(sql, params)
            elif target_company:
                sql = "SELECT * FROM financial_metrics WHERE company = ? ORDER BY fiscal_year DESC"
                cursor.execute(sql, (target_company,))
            else:
                sql = "SELECT * FROM financial_metrics ORDER BY fiscal_year DESC LIMIT 8"
                cursor.execute(sql)

            rows = cursor.fetchall()

            for r in rows:
                dict_row = dict(r)
                results.append({
                    "source": "SQL_Database",
                    "sql_query": sql,
                    "content": f"{dict_row.get('company')} ({dict_row.get('fiscal_year', 'N/A')}): Net Profit = Rs {dict_row.get('net_profit_crores', 0):,.0f} Cr, EBITDA = Rs {dict_row.get('ebitda_crores', 0):,.0f} Cr, Revenue = Rs {dict_row.get('revenue_crores', 0):,.0f} Cr (${dict_row.get('revenue_usd_billion', 0)}B), Net Debt = Rs {dict_row.get('net_debt_crores', 0):,.0f} Cr, EV Share = {dict_row.get('ev_market_share_pct', 0)}%. Highlights: {dict_row.get('key_highlights', '')}",
                    "details": dict_row
                })

        except Exception as e:
            results.append({"source": "SQL_Database", "content": f"SQL query execution error: {e}", "details": {}})
        finally:
            conn.close()

        return results

if __name__ == "__main__":
    sql_engine = SQLEngineRetriever()
    res = sql_engine.query("profits in financial year 2024/2025 for tata motors?")
    for r in res:
        print(r['content'], flush=True)
