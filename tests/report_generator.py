import os
import json
import logging
from typing import List, Dict, Any

# Set up logging
logger = logging.getLogger(__name__)

def generate_pie_chart(passed: int, failed: int, filepath: str) -> None:
    """Generates a pure-SVG pass/fail pie chart representation."""
    total = passed + failed
    pass_percent = (passed / total * 100) if total > 0 else 0
    # Circumference of radius 50 is ~314.16
    stroke_offset = 314.16 - (314.16 * pass_percent / 100)
    
    svg = f"""<svg width="300" height="200" viewBox="0 0 300 200" xmlns="http://www.w3.org/2000/svg">
  <rect width="100%" height="100%" fill="#0d1117" rx="8"/>
  <text x="15" y="30" fill="#c9d1d9" font-family="Segoe UI, Arial" font-size="14" font-weight="bold">Pass vs Fail Ratio</text>
  
  <circle cx="90" cy="110" r="50" fill="none" stroke="#21262d" stroke-width="18"/>
  <circle cx="90" cy="110" r="50" fill="none" stroke="#2ea44f" stroke-width="18" 
          stroke-dasharray="314.16" stroke-dashoffset="{stroke_offset:.2f}" 
          transform="rotate(-90 90 110)"/>
          
  <text x="90" y="116" fill="#f0f6fc" font-family="Segoe UI, Arial" font-size="16" font-weight="bold" text-anchor="middle">{pass_percent:.1f}%</text>
  
  <!-- Legend -->
  <rect x="180" y="85" width="12" height="12" fill="#2ea44f" rx="2"/>
  <text x="200" y="96" fill="#8b949e" font-family="Segoe UI, Arial" font-size="12">Passed: {passed}</text>
  
  <rect x="180" y="115" width="12" height="12" fill="#f85149" rx="2"/>
  <text x="200" y="126" fill="#8b949e" font-family="Segoe UI, Arial" font-size="12">Failed: {failed}</text>
</svg>"""
    
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(svg)

def generate_bar_chart(labels: List[str], values: List[float], filepath: str, title: str, fill_color: str = "#58a6ff") -> None:
    """Generates a responsive SVG bar chart based on labels and values."""
    width = 500
    height = 300
    padding_left = 120
    padding_right = 30
    padding_top = 50
    padding_bottom = 40
    
    chart_width = width - padding_left - padding_right
    chart_height = height - padding_top - padding_bottom
    
    bar_height = min(30, int(chart_height / max(1, len(labels)) * 0.7))
    spacing = int((chart_height - (bar_height * len(labels))) / max(1, len(labels) + 1))
    
    bars_xml = []
    labels_xml = []
    
    for idx, (label, val) in enumerate(zip(labels, values)):
        y = padding_top + spacing + idx * (bar_height + spacing)
        # Scale value width to percentage
        w = int(chart_width * val / 100) if val > 0 else 2
        
        # Highlight values based on scores
        color = fill_color
        if "Accuracy" in title or "Score" in title:
            if val < 60:
                color = "#f85149"  # Red
            elif val < 85:
                color = "#d29922"  # Amber
                
        bars_xml.append(f"""
    <rect x="{padding_left}" y="{y}" width="{w}" height="{bar_height}" fill="{color}" rx="3"/>
    <text x="{padding_left + w + 8}" y="{y + bar_height//2 + 4}" fill="#c9d1d9" font-family="Segoe UI, Arial" font-size="11" font-weight="bold">{val:.1f}%</text>
        """)
        
        # Handle label wrap or display
        short_label = label if len(label) <= 16 else label[:14] + ".."
        labels_xml.append(f"""
    <text x="{padding_left - 10}" y="{y + bar_height//2 + 4}" fill="#8b949e" font-family="Segoe UI, Arial" font-size="11" text-anchor="end">{short_label}</text>
        """)
        
    bars_str = "\n".join(bars_xml)
    labels_str = "\n".join(labels_xml)
    
    svg = f"""<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg">
  <rect width="100%" height="100%" fill="#0d1117" rx="8"/>
  <text x="15" y="30" fill="#c9d1d9" font-family="Segoe UI, Arial" font-size="14" font-weight="bold">{title}</text>
  
  <!-- Y Axis line -->
  <line x1="{padding_left}" y1="{padding_top}" x2="{padding_left}" y2="{height - padding_bottom}" stroke="#30363d" stroke-width="1.5"/>
  
  <!-- Bars -->
  {bars_str}
  
  <!-- Labels -->
  {labels_str}
</svg>"""
    
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(svg)

def generate_report(results: List[Dict[str, Any]], failed_cases: List[Dict[str, Any]], metrics: Dict[str, Any], output_dir: str = "reports") -> None:
    """Compiles markdown, HTML reports and saves JSON metrics."""
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Save JSON records
    with open(os.path.join(output_dir, "evaluation_results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
        
    with open(os.path.join(output_dir, "failed_cases.json"), "w", encoding="utf-8") as f:
        json.dump(failed_cases, f, indent=2, default=str)
        
    # 2. Render SVGs
    passed_count = metrics["passed_count"]
    failed_count = metrics["failed_count"]
    generate_pie_chart(passed_count, failed_count, os.path.join(output_dir, "pass_fail_pie.svg"))
    
    # Accuracy components
    acc_labels = ["Intent", "Complexity", "Routing", "Citations", "Answers", "Memory"]
    acc_vals = [
        metrics["intent_accuracy"] * 100,
        metrics["complexity_accuracy"] * 100,
        metrics["routing_accuracy"] * 100,
        metrics["citation_accuracy"] * 100,
        metrics["answer_accuracy"] * 100,
        metrics["memory_accuracy"] * 100
    ]
    generate_bar_chart(acc_labels, acc_vals, os.path.join(output_dir, "category_scores_bar.svg"), "Pipeline Layer Accuracies", "#58a6ff")
    
    # Latencies
    lat_labels = ["Simple", "Medium", "Complex", "Avg Latency"]
    lat_vals = [
        metrics.get("latency_simple", 0) / 100, # scaled dynamically for visual bars
        metrics.get("latency_medium", 0) / 100,
        metrics.get("latency_complex", 0) / 100,
        metrics["avg_latency_ms"] / 100
    ]
    # Standardize labels and units
    generate_bar_chart(lat_labels, [min(100.0, l) for l in lat_vals], os.path.join(output_dir, "latency_dist_bar.svg"), "Avg Latency Factor (ms/100)", "#bc8cff")

    # 3. Generate Markdown Report
    md_content = f"""# RAG Pipeline Evaluation Report

## Executive Summary

| Metrics | Value |
|---|---|
| **Total Test Cases** | {metrics["total_count"]} |
| **Passed Cases** | {metrics["passed_count"]} |
| **Failed Cases** | {metrics["failed_count"]} |
| **Overall Score** | {metrics["overall_score"] * 100:.1f}% |
| **Average Latency** | {metrics["avg_latency_ms"]:.0f} ms |
| **Average Confidence** | {metrics["avg_confidence"] * 100:.1f}% |
| **Average Compression Ratio** | {metrics["avg_compression_ratio"] * 100:.1f}% |

---

## Visualizations

### Pass / Fail Ratio
![Pass Fail Ratio](pass_fail_pie.svg)

### Pipeline Accuracies
![Pipeline Accuracies](category_scores_bar.svg)

---

## Layer Metrics Detail

- **Intent Accuracy**: {metrics["intent_accuracy"] * 100:.1f}%
- **Complexity Accuracy**: {metrics["complexity_accuracy"] * 100:.1f}%
- **Routing Accuracy**: {metrics["routing_accuracy"] * 100:.1f}%
- **Citation Accuracy**: {metrics["citation_accuracy"] * 100:.1f}%
- **Answer Accuracy**: {metrics["answer_accuracy"] * 100:.1f}%
- **Memory Accuracy**: {metrics["memory_accuracy"] * 100:.1f}%

---

## Failed Cases List ({len(failed_cases)})

"""
    if failed_cases:
        for idx, fc in enumerate(failed_cases):
            md_content += f"""### [{idx + 1}] Query: "{fc['query']}"
- **Expected Intent**: {fc.get('expected_intent')} | **Got**: {fc.get('predicted_intent')}
- **Expected Complexity**: {fc.get('expected_complexity')} | **Got**: {fc.get('predicted_complexity')}
- **Routed Docs**: Expected: {fc.get('expected_routed')} | Actual: {fc.get('actual_routed')}
- **Failure Reason**: `{fc['reason']}`
- **Suggested Fix**: {fc.get('suggested_fix', 'Investigate retriever payload mapping or routing weights.')}

"""
    else:
        md_content += "_All test cases completed successfully!_\n"
        
    with open(os.path.join(output_dir, "evaluation_report.md"), "w", encoding="utf-8") as f:
        f.write(md_content)
        
    # 4. Generate HTML Report
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RAG Evaluation Report</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
            background-color: #0d1117;
            color: #c9d1d9;
            margin: 0;
            padding: 40px;
        }}
        .container {{
            max-width: 900px;
            margin: 0 auto;
        }}
        h1, h2, h3 {{
            color: #f0f6fc;
            border-bottom: 1px fill #30363d;
            padding-bottom: 8px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border: 1px solid #30363d;
        }}
        th {{
            background-color: #161b22;
        }}
        tr:nth-child(even) {{
            background-color: #0d1117;
        }}
        .score {{
            font-size: 24px;
            font-weight: bold;
            color: #2ea44f;
        }}
        .charts {{
            display: flex;
            flex-wrap: wrap;
            gap: 20px;
            margin: 30px 0;
        }}
        .chart-box {{
            background-color: #161b22;
            border: 1px solid #30363d;
            border-radius: 6px;
            padding: 10px;
        }}
        .failed-card {{
            background-color: #161b22;
            border: 1px solid #f85149;
            border-radius: 6px;
            padding: 20px;
            margin: 20px 0;
        }}
        .failed-title {{
            color: #f85149;
            font-weight: bold;
            margin-top: 0;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Enterprise RAG Framework Evaluation</h1>
        <h2>Executive Summary</h2>
        <table>
            <tr>
                <th>Metric</th>
                <th>Result</th>
            </tr>
            <tr>
                <td>Overall Evaluation Score</td>
                <td class="score">{metrics["overall_score"] * 100:.1f}%</td>
            </tr>
            <tr>
                <td>Total Test Cases Run</td>
                <td>{metrics["total_count"]}</td>
            </tr>
            <tr>
                <td>Passed / Failed</td>
                <td>{passed_count} / {failed_count}</td>
            </tr>
            <tr>
                <td>Average Latency</td>
                <td>{metrics["avg_latency_ms"]:.0f} ms</td>
            </tr>
            <tr>
                <td>Average Confidence</td>
                <td>{metrics["avg_confidence"] * 100:.1f}%</td>
            </tr>
            <tr>
                <td>Avg Context Compression Ratio</td>
                <td>{metrics["avg_compression_ratio"] * 100:.1f}%</td>
            </tr>
        </table>
        
        <h2>Visualizations</h2>
        <div class="charts">
            <div class="chart-box">
                <embed src="pass_fail_pie.svg" type="image/svg+xml" />
            </div>
            <div class="chart-box">
                <embed src="category_scores_bar.svg" type="image/svg+xml" />
            </div>
            <div class="chart-box">
                <embed src="latency_dist_bar.svg" type="image/svg+xml" />
            </div>
        </div>
        
        <h2>Failed Cases Detail ({len(failed_cases)})</h2>
        
        {"<p>All test cases completed successfully!</p>" if not failed_cases else ""}
        
        {"".join(f'''
        <div class="failed-card">
            <div class="failed-title">[{idx+1}] Query: "{fc['query']}"</div>
            <p><strong>Failure Reason:</strong> <code>{fc['reason']}</code></p>
            <p><strong>Expected Intent:</strong> {fc.get('expected_intent')} | <strong>Predicted:</strong> {fc.get('predicted_intent')}</p>
            <p><strong>Expected Complexity:</strong> {fc.get('expected_complexity')} | <strong>Predicted:</strong> {fc.get('predicted_complexity')}</p>
            <p><strong>Expected Routing:</strong> {fc.get('expected_routed')} | <strong>Actual Routing:</strong> {fc.get('actual_routed')}</p>
            <p><strong>Suggested Fix:</strong> {fc.get('suggested_fix')}</p>
        </div>
        ''' for idx, fc in enumerate(failed_cases))}
        
    </div>
</body>
</html>"""
    
    with open(os.path.join(output_dir, "evaluation_report.html"), "w", encoding="utf-8") as f:
        f.write(html_content)
        
    logger.info(f"RAG Evaluation reports generated in '{output_dir}/' folder.")
