#!/bin/bash
cd /home/jmt/dev/StockMarketReportRag
source venv/bin/activate
DATE=$(date +%Y-%m-%d)
python3 generate_report.py --date $DATE --market AR --out /home/jmt/cierre-jornada/cierre_ar.txt
