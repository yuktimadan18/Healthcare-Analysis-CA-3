/**
 * charts.js
 * ---------
 * Modular Plotly.js chart generation engine for the
 * "Population Health Analytics & Explainable ML — CDC BRFSS 2024" dashboard.
 * Provides reusable, theme-consistent visual components across all sections.
 */

const DashboardCharts = {
    // Curated clinical surveillance color palette
    theme: {
        navy: "#0b2545",
        teal: "#0e8388",
        tealLight: "#2eafb5",
        slate: "#2c3e50",
        danger: "#d62728",
        success: "#2ca02c",
        warning: "#ff7f0e",
        purple: "#8e44ad",
        gray: "#7f8c8d",
        gridColor: "rgba(0, 0, 0, 0.06)",
        fontFamily: "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif"
    },

    /**
     * Section A: Candidate outcomes prevalence comparison bar chart.
     */
    renderCandidateOutcomes(containerId, outcomesList) {
        const sorted = [...outcomesList].sort((a, b) => a.unweighted_prev_pct - b.unweighted_prev_pct);
        const yLabels = sorted.map(d => `${d.condition_name} (${d.variable})`);
        const unweighted = sorted.map(d => d.unweighted_prev_pct);
        const weighted = sorted.map(d => d.weighted_prev_pct);

        const traceUnweighted = {
            y: yLabels,
            x: unweighted,
            name: "Unweighted Sample Prevalence (%)",
            type: "bar",
            orientation: "h",
            marker: { color: "#2e86ab" }
        };

        const traceWeighted = {
            y: yLabels,
            x: weighted,
            name: "Survey-Weighted US Prevalence (%)",
            type: "bar",
            orientation: "h",
            marker: { color: "#0e8388" }
        };

        const layout = {
            title: "<b>Candidate Chronic Condition Prevalences (CDC BRFSS 2024, N=452,464)</b>",
            barmode: "group",
            xaxis: { title: "Prevalence Percentage (%)", gridcolor: this.theme.gridColor },
            yaxis: { automargin: true },
            legend: { orientation: "h", y: -0.2 },
            margin: { l: 280, r: 40, t: 60, b: 60 },
            font: { family: this.theme.fontFamily }
        };

        Plotly.newPlot(containerId, [traceUnweighted, traceWeighted], layout, { responsive: true });
    },

    /**
     * Section B: Population Health Explorer distribution and stratified prevalence.
     */
    renderPopulationDistribution(containerId, distData, mode = "weighted") {
        const categories = distData.categories;
        const labels = categories.map(c => c.label);
        const percentages = mode === "weighted" ? categories.map(c => c.weighted_pct) : categories.map(c => c.unweighted_pct);
        const prevalences = mode === "weighted" ? categories.map(c => c.weighted_prevalence_pct) : categories.map(c => c.unweighted_prevalence_pct);

        const traceDist = {
            x: labels,
            y: percentages,
            name: mode === "weighted" ? "Population Distribution (% of US Adults)" : "Sample Distribution (% of Cohort)",
            type: "bar",
            marker: { color: "#13293d" },
            yaxis: "y"
        };

        const tracePrev = {
            x: labels,
            y: prevalences,
            name: "Reported CAD/MI Prevalence (%)",
            type: "scatter",
            mode: "lines+markers",
            marker: { color: this.theme.danger, size: 8 },
            line: { width: 3 },
            yaxis: "y2"
        };

        const layout = {
            title: `<b>${distData.display_name} (${distData.variable_name}) — Distribution & Stratified CAD/MI Prevalence</b>`,
            xaxis: { automargin: true, tickangle: -25 },
            yaxis: { title: mode === "weighted" ? "Population Proportion (% of US)" : "Sample Proportion (%)", gridcolor: this.theme.gridColor },
            yaxis2: {
                title: "Reported CAD/MI Prevalence (%)",
                overlaying: "y",
                side: "right",
                showgrid: false,
                rangemode: "tozero"
            },
            legend: { orientation: "h", y: -0.3 },
            margin: { l: 60, r: 60, t: 60, b: 90 },
            font: { family: this.theme.fontFamily }
        };

        Plotly.newPlot(containerId, [traceDist, tracePrev], layout, { responsive: true });
    },

    /**
     * Section C: Model ROC and Precision-Recall Curves.
     */
    renderModelCurves(rocContainerId, prContainerId, curvesData, modelMetadata) {
        const colors = {
            "hist_gradient_boosting_unweighted": "#0e8388", // Champion Teal
            "hist_gradient_boosting_balanced": "#ff7f0e",
            "logistic_regression_unweighted": "#1f77b4",
            "logistic_regression_balanced": "#d62728"
        };

        const rocTraces = [];
        const prTraces = [];

        for (const [key, curve] of Object.entries(curvesData)) {
            const mMeta = modelMetadata.find(m => m.model_key === key);
            const mName = mMeta ? mMeta.model_name : key;
            const isChamp = key === "hist_gradient_boosting_unweighted";

            rocTraces.push({
                x: curve.roc.fpr,
                y: curve.roc.tpr,
                mode: "lines",
                name: `${mName} (ROC-AUC = ${mMeta.val_roc_auc})`,
                line: { color: colors[key], width: isChamp ? 3.5 : 2 }
            });

            prTraces.push({
                x: curve.pr.recall,
                y: curve.pr.precision,
                mode: "lines",
                name: `${mName} (PR-AUC = ${mMeta.val_pr_auc})`,
                line: { color: colors[key], width: isChamp ? 3.5 : 2 }
            });
        }

        // Add ROC reference diagonal
        rocTraces.push({
            x: [0, 1],
            y: [0, 1],
            mode: "lines",
            name: "Chance (AUC = 0.50)",
            line: { dash: "dash", color: "#95a5a6", width: 1.5 }
        });

        // Add PR prevalence baseline
        prTraces.push({
            x: [0, 1],
            y: [0.0936, 0.0936],
            mode: "lines",
            name: "Prevalence Baseline (9.36%)",
            line: { dash: "dash", color: "#95a5a6", width: 1.5 }
        });

        const rocLayout = {
            title: "<b>Validation ROC Curves (Discrimination)</b>",
            xaxis: { title: "False Positive Rate (1 - Specificity)", range: [0, 1], gridcolor: this.theme.gridColor },
            yaxis: { title: "True Positive Rate (Sensitivity)", range: [0, 1.02], gridcolor: this.theme.gridColor },
            legend: { orientation: "h", y: -0.28 },
            margin: { l: 60, r: 30, t: 50, b: 90 },
            font: { family: this.theme.fontFamily }
        };

        const prLayout = {
            title: "<b>Validation Precision-Recall Curves (Class Imbalance Primary)</b>",
            xaxis: { title: "Recall (Sensitivity)", range: [0, 1], gridcolor: this.theme.gridColor },
            yaxis: { title: "Precision (Positive Predictive Value)", range: [0, 0.7], gridcolor: this.theme.gridColor },
            legend: { orientation: "h", y: -0.28 },
            margin: { l: 60, r: 30, t: 50, b: 90 },
            font: { family: this.theme.fontFamily }
        };

        Plotly.newPlot(rocContainerId, rocTraces, rocLayout, { responsive: true });
        Plotly.newPlot(prContainerId, prTraces, prLayout, { responsive: true });
    },

    /**
     * Section C: Model Calibration Deciles Chart.
     */
    renderCalibrationCurve(containerId, curvesData, modelMetadata) {
        const traces = [];

        // Perfect calibration diagonal
        traces.push({
            x: [0, 1],
            y: [0, 1],
            mode: "lines",
            name: "Perfect Calibration",
            line: { dash: "dash", color: "#7f8c8d", width: 1.5 }
        });

        for (const [key, curve] of Object.entries(curvesData)) {
            const mMeta = modelMetadata.find(m => m.model_key === key);
            const mName = mMeta ? mMeta.model_name : key;
            const isChamp = key === "hist_gradient_boosting_unweighted";
            const color = isChamp ? "#0e8388" : (key.includes("balanced") ? "#ff7f0e" : "#1f77b4");

            traces.push({
                x: curve.calibration.predicted_prob,
                y: curve.calibration.observed_prob,
                mode: "lines+markers",
                name: `${mName} (Brier = ${mMeta.val_brier_score})`,
                marker: { size: 7 },
                line: { color: color, width: isChamp ? 3 : 1.8 }
            });
        }

        const layout = {
            title: "<b>Probability Calibration Curves (Empirical Deciles, N=67,870)</b>",
            xaxis: { title: "Mean Predicted Probability", range: [0, 1], gridcolor: this.theme.gridColor },
            yaxis: { title: "Observed Event Rate (Fraction of Positives)", range: [0, 1], gridcolor: this.theme.gridColor },
            legend: { orientation: "h", y: -0.28 },
            margin: { l: 60, r: 30, t: 50, b: 90 },
            font: { family: this.theme.fontFamily }
        };

        Plotly.newPlot(containerId, traces, layout, { responsive: true });
    },

    /**
     * Section D: Threshold Trade-Off Analysis Curve with locked threshold and dynamic exploration marker.
     */
    renderThresholdTradeoff(containerId, gridPoints, lockedThreshold = 0.08, currentThreshold = null) {
        const ths = gridPoints.map(p => p.threshold);
        const sens = gridPoints.map(p => p.sensitivity * 100);
        const spec = gridPoints.map(p => p.specificity * 100);
        const ppv = gridPoints.map(p => p.precision_ppv * 100);
        const npv = gridPoints.map(p => p.npv * 100);
        const youden = gridPoints.map(p => p.youden_j * 100);

        const traces = [
            { x: ths, y: sens, name: "Sensitivity (Recall) %", mode: "lines", line: { color: "#27ae60", width: 2.5 } },
            { x: ths, y: spec, name: "Specificity %", mode: "lines", line: { color: "#2980b9", width: 2.5 } },
            { x: ths, y: ppv, name: "Precision (PPV) %", mode: "lines", line: { color: "#e67e22", width: 2 } },
            { x: ths, y: npv, name: "Negative Pred Value (NPV) %", mode: "lines", line: { color: "#8e44ad", width: 2 } },
            { x: ths, y: youden, name: "Youden's J Index (*100)", mode: "lines", line: { color: "#c0392b", dash: "dot", width: 2 } }
        ];

        const shapes = [
            {
                type: "line",
                x0: lockedThreshold,
                x1: lockedThreshold,
                y0: 0,
                y1: 100,
                line: { color: "#8e44ad", width: 2.5, dash: "dash" }
            }
        ];

        const annotations = [
            {
                x: lockedThreshold,
                y: 95,
                text: `Locked Operational Cutoff: τ = ${lockedThreshold}`,
                showarrow: true,
                arrowhead: 2,
                ax: 60,
                ay: 0,
                bgcolor: "#f4ecf7",
                bordercolor: "#8e44ad",
                font: { size: 11, color: "#8e44ad" }
            }
        ];

        // Add dynamic exploration marker if different from locked
        if (currentThreshold !== null && Math.abs(currentThreshold - lockedThreshold) > 0.005) {
            shapes.push({
                type: "line",
                x0: currentThreshold,
                x1: currentThreshold,
                y0: 0,
                y1: 100,
                line: { color: "#34495e", width: 2, dash: "dot" }
            });
            annotations.push({
                x: currentThreshold,
                y: 75,
                text: `Explored Cutoff: τ = ${currentThreshold.toFixed(2)}`,
                showarrow: true,
                arrowhead: 2,
                ax: -60,
                ay: 0,
                bgcolor: "#ebf5fb",
                bordercolor: "#2980b9",
                font: { size: 11, color: "#2980b9" }
            });
        }

        const layout = {
            title: "<b>Decision Threshold Trade-Offs (Validation Partition, N=67,870)</b>",
            xaxis: { title: "Decision Threshold Cutoff (τ)", range: [0.02, 0.65], gridcolor: this.theme.gridColor },
            yaxis: { title: "Performance Metric (%)", range: [0, 105], gridcolor: this.theme.gridColor },
            shapes: shapes,
            annotations: annotations,
            legend: { orientation: "h", y: -0.22 },
            margin: { l: 60, r: 40, t: 60, b: 75 },
            font: { family: this.theme.fontFamily }
        };

        Plotly.newPlot(containerId, traces, layout, { responsive: true });
    },

    /**
     * Section E: Global Grouped Feature Importance Bar Chart.
     */
    renderGlobalImportance(containerId, globalList, topN = 20) {
        const top = [...globalList].slice(0, topN).reverse();
        const yLabels = top.map(d => `${d.source_variable} (${d.num_transformed_columns} col${d.num_transformed_columns > 1 ? 's' : ''})`);
        const drops = top.map(d => d.mean_pr_auc_drop);
        const errs = top.map(d => d.std_pr_auc_drop);

        const trace = {
            y: yLabels,
            x: drops,
            type: "bar",
            orientation: "h",
            marker: { color: "#0e8388" },
            error_x: {
                type: "data",
                array: errs,
                visible: true,
                color: "#0b2545"
            }
        };

        const layout = {
            title: `<b>Top ${topN} Original Survey Predictors Ranked by Grouped Permutation Importance</b><br><sup>Primary Metric: Validation PR-AUC Decrease (± 1 SD across 10 repeats, Seed 42)</sup>`,
            xaxis: { title: "Mean PR-AUC Degradation", gridcolor: this.theme.gridColor },
            yaxis: { automargin: true },
            margin: { l: 220, r: 30, t: 70, b: 50 },
            font: { family: this.theme.fontFamily }
        };

        Plotly.newPlot(containerId, [trace], layout, { responsive: true });
    },

    /**
     * Section E: Transformed Indicator Feature Importance Bar Chart.
     */
    renderIndicatorImportance(containerId, indicatorList, topN = 25) {
        const top = [...indicatorList].slice(0, topN).reverse();
        const yLabels = top.map(d => `${d.transformed_feature_name}`);
        const drops = top.map(d => d.mean_pr_auc_drop);
        const errs = top.map(d => d.std_pr_auc_drop);

        const trace = {
            y: yLabels,
            x: drops,
            type: "bar",
            orientation: "h",
            marker: { color: "#2e86ab" },
            error_x: {
                type: "data",
                array: errs,
                visible: true,
                color: "#13293d"
            }
        };

        const layout = {
            title: `<b>Top ${topN} Transformed Indicators Ranked by Permutation Importance</b><br><sup>Demonstrating Dummy-Dilution Dynamics across 190 Transformed Features</sup>`,
            xaxis: { title: "Mean PR-AUC Degradation", gridcolor: this.theme.gridColor },
            yaxis: { automargin: true },
            margin: { l: 260, r: 30, t: 70, b: 50 },
            font: { family: this.theme.fontFamily }
        };

        Plotly.newPlot(containerId, [trace], layout, { responsive: true });
    },

    /**
     * Section E: Unscaled Partial Dependence Profile.
     */
    renderPartialDependence(containerId, pdpData, varName = "_AGE80") {
        const pdp = pdpData[varName];
        if (!pdp) return;

        const trace = {
            x: pdp.grid_unscaled,
            y: pdp.marginal_probability,
            mode: "lines+markers",
            marker: { size: 5, color: "#0e8388" },
            line: { color: "#0e8388", width: 3 },
            name: "Marginal Predicted Risk"
        };

        const traces = [trace];

        const layout = {
            title: `<b>Partial Dependence Profile: ${pdp.display_name}</b>`,
            xaxis: { title: `${pdp.display_name} (${pdp.unit})`, gridcolor: this.theme.gridColor },
            yaxis: {
                title: "Marginal Predicted Risk (Probability)",
                range: [0, Math.max(0.25, Math.max(...pdp.marginal_probability) * 1.15)],
                gridcolor: this.theme.gridColor
            },
            shapes: [
                {
                    type: "line",
                    x0: Math.min(...pdp.grid_unscaled),
                    x1: Math.max(...pdp.grid_unscaled),
                    y0: 0.08,
                    y1: 0.08,
                    line: { color: "#8e44ad", width: 1.8, dash: "dash" }
                },
                {
                    type: "line",
                    x0: Math.min(...pdp.grid_unscaled),
                    x1: Math.max(...pdp.grid_unscaled),
                    y0: 0.0936,
                    y1: 0.0936,
                    line: { color: "#c0392b", width: 1.8, dash: "dot" }
                }
            ],
            annotations: [
                {
                    x: Math.min(...pdp.grid_unscaled) + (Math.max(...pdp.grid_unscaled) - Math.min(...pdp.grid_unscaled)) * 0.15,
                    y: 0.086,
                    text: "Screening Cutoff (0.080)",
                    showarrow: false,
                    font: { size: 10, color: "#8e44ad" }
                },
                {
                    x: Math.min(...pdp.grid_unscaled) + (Math.max(...pdp.grid_unscaled) - Math.min(...pdp.grid_unscaled)) * 0.15,
                    y: 0.100,
                    text: "Cohort Prevalence (9.36%)",
                    showarrow: false,
                    font: { size: 10, color: "#c0392b" }
                }
            ],
            margin: { l: 60, r: 30, t: 50, b: 50 },
            font: { family: this.theme.fontFamily }
        };

        Plotly.newPlot(containerId, traces, layout, { responsive: true });
    },

    /**
     * Section E: Local Case Directional Attribution Bar Chart.
     */
    renderLocalAttribution(containerId, records, caseInfo, topK = 10) {
        // Sort by absolute delta p
        const sorted = [...records].sort((a, b) => Math.abs(b.marginal_risk_delta_dp) - Math.abs(a.marginal_risk_delta_dp)).slice(0, topK).reverse();

        const yLabels = sorted.map(r => r.source_variable);
        const deltas = sorted.map(r => r.marginal_risk_delta_dp);
        const colors = deltas.map(d => d >= 0 ? this.theme.danger : this.theme.teal);

        const trace = {
            y: yLabels,
            x: deltas,
            type: "bar",
            orientation: "h",
            marker: { color: colors },
            text: deltas.map(d => (d >= 0 ? `+${d.toFixed(4)}` : `${d.toFixed(4)}`)),
            textposition: "auto"
        };

        const maxAbs = Math.max(...deltas.map(d => Math.abs(d)), 0.05);

        const layout = {
            title: `<b>${caseInfo.category} (${caseInfo.case_id}) — Top ${topK} Feature Risk Attributions</b><br><sup>Actual: ${caseInfo.actual_target === 1 ? 'Positive CAD/MI' : 'Negative'} | Predicted Probability: ${caseInfo.predicted_probability} (Locked Cutoff: 0.080)</sup>`,
            xaxis: {
                title: "Marginal Risk Attribution (Δp relative to background cohort distribution)",
                range: [-maxAbs * 1.25, maxAbs * 1.25],
                gridcolor: this.theme.gridColor
            },
            yaxis: { automargin: true },
            margin: { l: 150, r: 40, t: 70, b: 60 },
            font: { family: this.theme.fontFamily }
        };

        Plotly.newPlot(containerId, [trace], layout, { responsive: true });
    }
};

window.DashboardCharts = DashboardCharts;
