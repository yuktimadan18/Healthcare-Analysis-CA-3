"""
test_dashboard_ui.py
--------------------
Automated test suite for Milestone 8 Stage 3: Dashboard UI Implementation.
Validates:
1. Presence and integrity of index.html, styles.css, app.js, and components.
2. Section container structure for all 6 modular sections (A-F).
3. Script dependency loading order and CDN references.
4. Strict exclusion of clinical calculators or personal risk symptom checkers.
5. Invariant preservation of the locked threshold (0.080) in the UI markup.
6. Relative asset referencing for standalone local hosting.
"""

import unittest
from pathlib import Path
import re
import json

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DASHBOARD_DIR = PROJECT_ROOT / "dashboard"
COMPONENTS_DIR = DASHBOARD_DIR / "components"
DATA_DIR = DASHBOARD_DIR / "data"


class TestDashboardUI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.index_html_path = DASHBOARD_DIR / "index.html"
        cls.styles_css_path = DASHBOARD_DIR / "styles.css"
        cls.app_js_path = DASHBOARD_DIR / "app.js"
        cls.views_js_path = COMPONENTS_DIR / "views.js"
        cls.charts_js_path = COMPONENTS_DIR / "charts.js"
        cls.loader_js_path = COMPONENTS_DIR / "data_loader.js"

        cls.assertTrue(cls.index_html_path.exists(), "index.html is missing")
        cls.assertTrue(cls.styles_css_path.exists(), "styles.css is missing")
        cls.assertTrue(cls.app_js_path.exists(), "app.js is missing")
        cls.assertTrue(cls.views_js_path.exists(), "views.js is missing")
        cls.assertTrue(cls.charts_js_path.exists(), "charts.js is missing")
        cls.assertTrue(cls.loader_js_path.exists(), "data_loader.js is missing")

        with open(cls.index_html_path, "r", encoding="utf-8") as f:
            cls.index_html = f.read()
        with open(cls.styles_css_path, "r", encoding="utf-8") as f:
            cls.styles_css = f.read()
        with open(cls.app_js_path, "r", encoding="utf-8") as f:
            cls.app_js = f.read()
        with open(cls.views_js_path, "r", encoding="utf-8") as f:
            cls.views_js = f.read()
        with open(cls.charts_js_path, "r", encoding="utf-8") as f:
            cls.charts_js = f.read()

    def test_html_structure_and_title(self):
        """Verifies title tag, semantic structure, and charset encoding."""
        self.assertIn("<title>Population Health Analytics & Explainable ML — CDC BRFSS 2024</title>", self.index_html)
        self.assertIn('<meta charset="UTF-8">', self.index_html)
        self.assertIn('<meta name="viewport"', self.index_html)
        self.assertIn('<main class="app-main"', self.index_html)
        self.assertIn('<aside class="app-sidebar"', self.index_html)

    def test_all_six_sections_present_in_dom(self):
        """Verifies container divs for Sections A through F exist in index.html."""
        expected_sections = [
            'id="section-overview"',
            'id="section-population"',
            'id="section-performance"',
            'id="section-threshold"',
            'id="section-xai"',
            'id="section-limitations"'
        ]
        for sec in expected_sections:
            self.assertIn(sec, self.index_html, f"Missing section container: {sec}")

    def test_script_loading_sequence(self):
        """Verifies that components and scripts are loaded in strict dependency order."""
        idx_charts = self.index_html.find('src="components/charts.js"')
        idx_views = self.index_html.find('src="components/views.js"')
        idx_loader = self.index_html.find('src="components/data_loader.js"')
        idx_app = self.index_html.find('src="app.js"')

        self.assertGreater(idx_charts, -1, "charts.js script tag missing")
        self.assertGreater(idx_views, -1, "views.js script tag missing")
        self.assertGreater(idx_loader, -1, "data_loader.js script tag missing")
        self.assertGreater(idx_app, -1, "app.js script tag missing")

        # Order check: charts -> views -> loader -> app
        self.assertLess(idx_charts, idx_views, "charts.js must precede views.js")
        self.assertLess(idx_views, idx_app, "views.js must precede app.js")
        self.assertLess(idx_loader, idx_app, "data_loader.js must precede app.js")

    def test_locked_threshold_displayed_and_immutable(self):
        """Verifies that locked threshold 0.080 is explicitly marked in HTML, views, and charts."""
        self.assertIn("0.080", self.index_html)
        self.assertIn("0.080", self.views_js)
        self.assertIn("0.08", self.charts_js)
        # Verify no form action to mutate model threshold
        self.assertNotIn("update_threshold_model", self.app_js)
        self.assertNotIn("save_threshold", self.app_js)

    def test_exclusion_of_clinical_prediction_calculator(self):
        """Verifies strict adherence to rule: NO individual symptom checker or clinical calculator."""
        forbidden_terms = [
            "symptom_checker",
            "calculate_my_risk",
            "personal_risk_calculator",
            "predict_patient",
            "enter_your_symptoms",
            "patient_form"
        ]
        for term in forbidden_terms:
            self.assertNotIn(term, self.index_html.lower())
            self.assertNotIn(term, self.views_js.lower())
            self.assertNotIn(term, self.app_js.lower())

    def test_charts_methods_coverage(self):
        """Verifies that DashboardCharts provides all required visualization methods."""
        required_methods = [
            "renderCandidateOutcomes",
            "renderPopulationDistribution",
            "renderModelCurves",
            "renderCalibrationCurve",
            "renderThresholdTradeoff",
            "renderGlobalImportance",
            "renderIndicatorImportance",
            "renderPartialDependence",
            "renderLocalAttribution"
        ]
        for m in required_methods:
            self.assertIn(m, self.charts_js, f"Missing method in DashboardCharts: {m}")

    def test_views_methods_coverage(self):
        """Verifies that DashboardViews provides renderers for all 6 sections."""
        required_views = [
            "renderOverview",
            "renderPopulationHealth",
            "renderModelPerformance",
            "renderThresholdAnalysis",
            "renderExplainableAi",
            "renderLimitations"
        ]
        for v in required_views:
            self.assertIn(v, self.views_js, f"Missing view in DashboardViews: {v}")

    def test_disclaimers_prominently_rendered(self):
        """Verifies that public health and research disclaimers are present."""
        self.assertIn("Public Health Surveillance Notice", self.views_js)
        self.assertIn("Mandatory Research Disclaimer", self.views_js)
        self.assertIn("Permissible Academic & Public Health Uses", self.views_js)
        self.assertIn("Strictly Prohibited Clinical & Commercial Uses", self.views_js)

    def test_css_design_system_tokens(self):
        """Verifies that styles.css defines required color tokens and responsive styles."""
        self.assertIn("--primary-navy:", self.styles_css)
        self.assertIn("--primary-teal:", self.styles_css)
        self.assertIn("--color-locked:", self.styles_css)
        self.assertIn("@media (max-width:", self.styles_css)

    def test_population_health_schema_and_views_compatibility(self):
        """Regression test: verifies that population_health.json schema matches views.js expectations without unhandled property access."""
        pop_file = DATA_DIR / "population_health.json"
        self.assertTrue(pop_file.exists(), "population_health.json must exist")
        with open(pop_file, "r", encoding="utf-8") as f:
            pop_data = json.load(f)

        self.assertIn("multivariable_odds_ratios", pop_data)
        self.assertIn("continuous_risk_factor_stats", pop_data)
        self.assertIn("distributions", pop_data)

        # Verify multivariable odds ratios structure
        for row in pop_data["multivariable_odds_ratios"]:
            self.assertIn("covariate", row)
            self.assertTrue("odds_ratio" in row or "aor" in row, "Row must have odds_ratio or aor")
            self.assertTrue("ci_lower_95" in row or "ci_lower" in row, "Row must have CI lower")
            self.assertTrue("ci_upper_95" in row or "ci_upper" in row, "Row must have CI upper")

        # Verify continuous risk factors structure
        for row in pop_data["continuous_risk_factor_stats"]:
            self.assertTrue("variable_label" in row or "variable" in row, "Row must identify variable")
            self.assertIn("mean", row)
            self.assertIn("median", row)
            self.assertIn("iqr", row)

        # Verify views.js defines safeToFixed and does not call unshielded .toFixed on missing properties
        self.assertIn("function safeToFixed(", self.views_js)
        self.assertIn("DashboardViews.safeToFixed = safeToFixed", self.views_js)

    def test_population_health_node_rendering_regression(self):
        """Regression test: executes node to render Population Health Explorer and all views, confirming no .toFixed or undefined errors."""
        import subprocess

        node_script = """
const fs = require('fs');
const viewsCode = fs.readFileSync('dashboard/components/views.js', 'utf8');
const overviewData = JSON.parse(fs.readFileSync('dashboard/data/overview_summary.json', 'utf8'));
const popData = JSON.parse(fs.readFileSync('dashboard/data/population_health.json', 'utf8'));
const perfData = JSON.parse(fs.readFileSync('dashboard/data/model_performance.json', 'utf8'));
const threshData = JSON.parse(fs.readFileSync('dashboard/data/threshold_analysis.json', 'utf8'));
const xaiData = JSON.parse(fs.readFileSync('dashboard/data/explainable_ai.json', 'utf8'));

const mockElement = {
    innerHTML: '',
    value: '_AGE80',
    addEventListener: () => {},
    querySelectorAll: () => [],
    querySelector: () => null,
    appendChild: () => {},
    classList: { add: () => {}, remove: () => {} },
    setAttribute: () => {},
    getAttribute: (attr) => (attr === 'data-category' ? 'typical_high_risk' : '')
};

const DashboardCharts = {
    renderPopulationDistribution: () => {},
    renderCandidateOutcomes: () => {},
    renderModelCurves: () => {},
    renderCalibrationCurve: () => {},
    renderThresholdTradeoff: () => {},
    renderGlobalImportance: () => {},
    renderIndicatorImportance: () => {},
    renderPartialDependence: () => {},
    renderLocalAttribution: () => {}
};
const window = { DashboardCharts };
const document = {
    getElementById: (id) => mockElement,
    querySelectorAll: () => [mockElement],
    querySelector: () => mockElement,
    addEventListener: () => {}
};

eval(viewsCode);

// Test safeToFixed helper specifically
if (window.DashboardViews.safeToFixed(0, 2) !== '0.00') throw new Error('safeToFixed(0) failed');
if (window.DashboardViews.safeToFixed(null, 2) !== 'N/A') throw new Error('safeToFixed(null) failed');
if (window.DashboardViews.safeToFixed(undefined, 2) !== 'N/A') throw new Error('safeToFixed(undefined) failed');
if (window.DashboardViews.safeToFixed(NaN, 2) !== 'N/A') throw new Error('safeToFixed(NaN) failed');
if (window.DashboardViews.safeToFixed('12.3456', 2) !== '12.35') throw new Error('safeToFixed string failed');

// Render Section B (Population Health Explorer)
const popContainer = { innerHTML: '' };
window.DashboardViews.renderPopulationHealth(popContainer, popData);
if (!popContainer.innerHTML.includes('Population Health Explorer')) {
    throw new Error('renderPopulationHealth produced unexpected output');
}

// Render all other sections to ensure zero regressions
window.DashboardViews.renderOverview({ innerHTML: '' }, overviewData);
window.DashboardViews.renderModelPerformance({ innerHTML: '' }, perfData);
window.DashboardViews.renderThresholdAnalysis({ innerHTML: '' }, threshData);
window.DashboardViews.renderExplainableAi({ innerHTML: '' }, xaiData);
window.DashboardViews.renderLimitations({ innerHTML: '' });

console.log('ALL_SECTIONS_RENDER_SUCCESS');
"""
        proc = subprocess.run(
            ["node", "-e", node_script],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True
        )
        self.assertEqual(
            proc.returncode, 0,
            f"Node rendering check failed: {proc.stderr}\nStdout: {proc.stdout}"
        )
        self.assertIn("ALL_SECTIONS_RENDER_SUCCESS", proc.stdout)


if __name__ == "__main__":
    unittest.main()

