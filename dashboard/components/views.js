/**
 * views.js
 * --------
 * Modular UI View component renderers for the
 * "Population Health Analytics & Explainable ML — CDC BRFSS 2024" dashboard.
 * Renders HTML structures and initiates chart bindings for Sections A through F.
 */
/**
 * Safe numeric formatter that preserves valid 0 values, handles null/undefined/NaN,
 * and formats to specified decimal places without throwing runtime exceptions.
 */
function safeToFixed(val, digits = 2, fallback = "N/A") {
    if (val === null || val === undefined || val === '') {
        return fallback;
    }
    const num = Number(val);
    if (!Number.isFinite(num)) {
        return fallback;
    }
    return num.toFixed(digits);
}

/**
 * Formatter for multivariable regression covariate names into clean human-readable labels.
 */
function formatCovariateName(name) {
    if (!name) return 'N/A';
    const labels = {
        'age_65plus': 'Age 65+ Years',
        'male_sex': 'Male Sex',
        'bmi_overweight_obese': 'Overweight / Obese (BMI ≥ 25)',
        'current_smoker': 'Current Smoker',
        'physical_inactivity': 'Physical Inactivity (No Leisure Exercise)',
        'fair_poor_health': 'Fair / Poor General Health',
        'diabetes': 'Diagnosed Diabetes',
        'kidney_disease': 'Diagnosed Kidney Disease'
    };
    return labels[name] || String(name).replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
}

const DashboardViews = {
    /**
     * Section A: Overview & Project Scope.
     */
    renderOverview(container, overviewData) {
        const dims = overviewData.cohort_dimensions;
        const champ = overviewData.champion_model;

        container.innerHTML = `
            <div class="section-header">
                <h2>1. Overview & Project Scope</h2>
                <p class="subtitle">Cardiovascular Disease Surveillance & Explainable Machine Learning (CDC BRFSS 2024)</p>
            </div>

            <div class="alert-banner warning">
                <div class="alert-icon">⚠️</div>
                <div class="alert-content">
                    <strong>Public Health Surveillance Notice:</strong> This interactive application communicates statistical patterns and machine learning classifications of self-reported coronary heart disease history in cross-sectional surveillance data. <strong>It does not provide clinical diagnoses, predict future disease onset, or offer personal medical advice.</strong>
                </div>
            </div>

            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-label">Analytical Cohort Size</div>
                    <div class="metric-value">${dims.total_survey_respondents.toLocaleString()}</div>
                    <div class="metric-caption">Validated respondents (from ${dims.raw_total_records.toLocaleString()} raw records; ${dims.missing_excluded_cases.toLocaleString()} excluded)</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Target Prevalence (_MICHD)</div>
                    <div class="metric-value">${dims.unweighted_prevalence_pct}%</div>
                    <div class="metric-caption">Survey-Weighted US Population: <strong>${dims.survey_weighted_prevalence_pct}%</strong> (Imbalance: ${dims.imbalance_ratio}:1)</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Champion Model</div>
                    <div class="metric-value">Unweighted GBDT</div>
                    <div class="metric-caption">HistGradientBoosting (Validation PR-AUC: <strong>${champ.validation_pr_auc}</strong>, ROC-AUC: <strong>${champ.validation_roc_auc}</strong>)</div>
                </div>
                <div class="metric-card locked">
                    <div class="metric-label">Locked Decision Cutoff</div>
                    <div class="metric-value">τ = ${champ.locked_threshold}</div>
                    <div class="metric-caption">Validation Youden's J criterion; preserves ${champ.locked_threshold} invariant</div>
                </div>
            </div>

            <!-- Methodology Pipeline Diagram -->
            <div class="card methodology-card" style="margin-top: 1.5rem;">
                <h3>Research Methodology & Analytics Pipeline</h3>
                <div class="pipeline-flow">
                    <div class="pipeline-step">
                        <div class="step-num">Step 1</div>
                        <div class="step-title">CDC BRFSS 2024</div>
                        <div class="step-desc">457,670 raw multi-state surveillance records</div>
                    </div>
                    <div class="pipeline-arrow">➔</div>
                    <div class="pipeline-step">
                        <div class="step-num">Step 2</div>
                        <div class="step-title">Eligibility & Leakage Audit</div>
                        <div class="step-desc">Excluded direct constituents (CVDINFR4, CVDCRHD4); N=452,464</div>
                    </div>
                    <div class="pipeline-arrow">➔</div>
                    <div class="pipeline-step">
                        <div class="step-num">Step 3</div>
                        <div class="step-title">Stratified Partitioning</div>
                        <div class="step-desc">Train (70%), Val (15%), Test (15%) preserving prevalence</div>
                    </div>
                    <div class="pipeline-arrow">➔</div>
                    <div class="pipeline-step">
                        <div class="step-num">Step 4</div>
                        <div class="step-title">Supervised Modeling</div>
                        <div class="step-desc">Linear vs. non-linear GBDT, unweighted vs. class-weighted</div>
                    </div>
                    <div class="pipeline-arrow">➔</div>
                    <div class="pipeline-step highlight">
                        <div class="step-num">Step 5</div>
                        <div class="step-title">Threshold Locking</div>
                        <div class="step-desc">Validation Youden's J locks cutoff at τ = 0.080</div>
                    </div>
                    <div class="pipeline-arrow">➔</div>
                    <div class="pipeline-step">
                        <div class="step-num">Step 6</div>
                        <div class="step-title">Explainable AI Audit</div>
                        <div class="step-desc">Grouped permutation, PDP, and local Ridge archetypes</div>
                    </div>
                </div>
            </div>

            <div class="chart-container" id="chart-outcomes-prevalence" style="height: 480px; margin-top: 1.5rem;"></div>

            <div class="card" style="margin-top: 1.5rem;">
                <h3>Candidate Chronic Condition Feasibility Assessment</h3>
                <div class="table-responsive">
                    <table class="data-table">
                        <thead>
                            <tr>
                                <th>Outcome Variable</th>
                                <th>Condition Name</th>
                                <th>Category</th>
                                <th>Positive Cases</th>
                                <th>Unweighted Prev (%)</th>
                                <th>Weighted US Prev (%)</th>
                                <th>Imbalance Ratio</th>
                            </tr>
                        </thead>
                        <tbody>
                            ${overviewData.candidate_outcomes_feasibility.map(o => `
                                <tr class="${o.variable === '_MICHD' ? 'highlight-row' : ''}">
                                    <td><strong>${o.variable}</strong> ${o.variable === '_MICHD' ? '<span class="badge-champ">Selected Target</span>' : ''}</td>
                                    <td>${o.condition_name}</td>
                                    <td>${o.category}</td>
                                    <td>${o.valid_positive.toLocaleString()}</td>
                                    <td>${safeToFixed(o.unweighted_prev_pct, 2)}%</td>
                                    <td>${safeToFixed(o.weighted_prev_pct, 2)}%</td>
                                    <td>${safeToFixed(o.imbalance_ratio, 2)}:1</td>
                                </tr>
                            `).join('')}
                        </tbody>
                    </table>
                </div>
            </div>

            <div class="card" style="margin-top: 1.5rem;">
                <h3>Core Methodological Safeguards</h3>
                <ul class="styled-list">
                    ${overviewData.methodological_disclaimers.map(d => `<li>${d}</li>`).join('')}
                </ul>
            </div>
        `;

        DashboardCharts.renderCandidateOutcomes('chart-outcomes-prevalence', overviewData.candidate_outcomes_feasibility);
    },

    /**
     * Section B: Population Health Explorer.
     */
    renderPopulationHealth(container, popData) {
        const dists = popData.distributions;
        const varKeys = Object.keys(dists);

        container.innerHTML = `
            <div class="section-header">
                <h2>2. Population Health Explorer</h2>
                <p class="subtitle">Interactive Strata Distributions, Complex Survey Weighting, and Risk Factor Covariates</p>
            </div>

            <div class="card controls-card">
                <div class="controls-row">
                    <div class="control-group">
                        <label for="select-pop-var"><strong>Select Health / Demographic Construct:</strong></label>
                        <select id="select-pop-var" class="form-select">
                            ${varKeys.map(k => `<option value="${k}">${dists[k].display_name} (${dists[k].variable_name} — ${dists[k].domain})</option>`).join('')}
                        </select>
                    </div>
                    <div class="control-group">
                        <label><strong>Weighting Mode:</strong></label>
                        <div class="btn-group">
                            <button class="btn btn-sm btn-primary active" id="btn-weighted">Survey-Weighted (US Adult Pop %)</button>
                            <button class="btn btn-sm btn-outline" id="btn-unweighted">Unweighted Sample Counts</button>
                        </div>
                    </div>
                </div>
            </div>

            <div class="chart-container" id="chart-population-distribution" style="height: 520px; margin-top: 1.5rem;"></div>

            <!-- Dynamic Strata Breakdown Table -->
            <div class="card" style="margin-top: 1.5rem;">
                <h3 id="table-strata-title">Strata Breakdown</h3>
                <div class="table-responsive">
                    <table class="data-table" id="table-strata">
                        <thead>
                            <tr>
                                <th>Category / Level</th>
                                <th>Unweighted Count</th>
                                <th>Unweighted Proportion (%)</th>
                                <th>Survey-Weighted Proportion (% of US)</th>
                                <th>Reported CAD/MI Prevalence (%)</th>
                            </tr>
                        </thead>
                        <tbody id="table-strata-body"></tbody>
                    </table>
                </div>
            </div>

            <!-- Continuous Risk Factors Table -->
            <div class="card" style="margin-top: 1.5rem;">
                <h3>Continuous Population Health Covariates Summary</h3>
                <p class="table-caption">Distribution statistics across the validated analytical cohort (N = 452,464)</p>
                <div class="table-responsive">
                    <table class="data-table">
                        <thead>
                            <tr>
                                <th>Covariate Construct</th>
                                <th>Valid Responses</th>
                                <th>Mean</th>
                                <th>Std Dev</th>
                                <th>Median</th>
                                <th>Min</th>
                                <th>Max</th>
                            </tr>
                        </thead>
                        <tbody>
                            ${(popData.continuous_risk_factor_stats || []).map(c => `
                                <tr>
                                    <td><strong>${c.variable_label || c.variable || c.name || 'N/A'}</strong></td>
                                    <td>${c.count ? c.count.toLocaleString() : 'N/A'}</td>
                                    <td>${safeToFixed(c.mean, 2)}</td>
                                    <td>${safeToFixed(c.std, 2)}</td>
                                    <td>${safeToFixed(c.median, 2)}</td>
                                    <td>${safeToFixed(c.min, 2)}</td>
                                    <td>${safeToFixed(c.max, 2)}</td>
                                </tr>
                            `).join('')}
                        </tbody>
                    </table>
                </div>
            </div>

            <!-- Multivariable Logistic Regression Table -->
            <div class="card" style="margin-top: 1.5rem;">
                <h3>Multivariable Survey-Weighted Odds Ratios (aOR) for Reported CAD/MI</h3>
                <p class="table-caption">Independent association estimates derived via survey-weighted logistic regression in Milestone 4</p>
                <div class="table-responsive">
                    <table class="data-table">
                        <thead>
                            <tr>
                                <th>Risk Factor Covariate</th>
                                <th>Adjusted Odds Ratio (aOR)</th>
                                <th>95% Confidence Interval</th>
                                <th>p-value</th>
                                <th>Epidemiological Association</th>
                            </tr>
                        </thead>
                        <tbody>
                            ${(popData.multivariable_odds_ratios || []).map(r => {
                                const orVal = r.odds_ratio !== undefined ? r.odds_ratio : (r.aor !== undefined ? r.aor : null);
                                const ciLow = r.ci_lower_95 !== undefined ? r.ci_lower_95 : (r.ci_lower !== undefined ? r.ci_lower : null);
                                const ciHigh = r.ci_upper_95 !== undefined ? r.ci_upper_95 : (r.ci_upper !== undefined ? r.ci_upper : null);
                                const pVal = r.p_value !== undefined ? r.p_value : null;

                                const orDisplay = safeToFixed(orVal, 2);
                                const ciDisplay = (ciLow !== null && ciHigh !== null)
                                    ? `[${safeToFixed(ciLow, 2)}, ${safeToFixed(ciHigh, 2)}]`
                                    : 'N/A';
                                const pDisplay = pVal !== null
                                    ? (pVal < 0.001 ? '< 0.001' : safeToFixed(pVal, 3))
                                    : 'N/A';

                                const isRisk = orVal !== null && orVal > 1.2;
                                const isProtective = orVal !== null && orVal < 0.85;
                                const badgeClass = isRisk ? 'badge-risk' : (isProtective ? 'badge-protective' : 'badge-neutral');
                                const badgeText = isRisk ? 'Elevated Odds' : (isProtective ? 'Inversely Associated' : 'Moderate');

                                return `
                                    <tr>
                                        <td><strong>${formatCovariateName(r.covariate)}</strong></td>
                                        <td><strong>${orDisplay}</strong></td>
                                        <td>${ciDisplay}</td>
                                        <td>${pDisplay}</td>
                                        <td><span class="badge ${badgeClass}">${badgeText}</span></td>
                                    </tr>
                                `;
                            }).join('')}
                        </tbody>
                    </table>
                </div>
            </div>

            <div class="alert-banner info" style="margin-top: 1.5rem;">
                <div class="alert-icon">ℹ️</div>
                <div class="alert-content">
                    <strong>Survey Weighting Note:</strong> Survey-weighted metrics incorporate CDC sample expansion weights (<code>_LLCPWT</code>) to project estimates onto the non-institutionalized US adult population. Formal variance estimation requires primary sampling unit (<code>_PSU</code>) and stratification (<code>_STSTR</code>) design adjustments.
                </div>
            </div>
        `;

        let currentVar = varKeys[0];
        let currentMode = "weighted";

        const updateTable = () => {
            const vData = dists[currentVar];
            if (!vData || !vData.categories) return;
            document.getElementById('table-strata-title').textContent = `${vData.display_name} (${vData.variable_name}) — Detailed Strata Metrics`;
            const tbody = document.getElementById('table-strata-body');
            tbody.innerHTML = vData.categories.map(c => `
                <tr>
                    <td><strong>${c.label}</strong></td>
                    <td>${c.unweighted_count ? c.unweighted_count.toLocaleString() : '0'}</td>
                    <td>${safeToFixed(c.unweighted_pct, 2)}%</td>
                    <td>${safeToFixed(c.weighted_pct, 2)}%</td>
                    <td><strong>${safeToFixed(c.weighted_prevalence_pct, 2)}%</strong></td>
                </tr>
            `).join('');
        };

        const updateView = () => {
            DashboardCharts.renderPopulationDistribution('chart-population-distribution', dists[currentVar], currentMode);
            updateTable();
        };

        document.getElementById('select-pop-var').addEventListener('change', (e) => {
            currentVar = e.target.value;
            updateView();
        });

        document.getElementById('btn-weighted').addEventListener('click', () => {
            currentMode = "weighted";
            document.getElementById('btn-weighted').classList.add('active');
            document.getElementById('btn-unweighted').classList.remove('active');
            updateView();
        });

        document.getElementById('btn-unweighted').addEventListener('click', () => {
            currentMode = "unweighted";
            document.getElementById('btn-unweighted').classList.add('active');
            document.getElementById('btn-weighted').classList.remove('active');
            updateView();
        });

        updateView();
    },

    /**
     * Section C: Model Performance & Comparative Evaluation.
     */
    renderModelPerformance(container, perfData) {
        container.innerHTML = `
            <div class="section-header">
                <h2>3. Model Performance & Comparative Evaluation</h2>
                <p class="subtitle">Discrimination, Calibration, and Class-Imbalance Auditing across Supervised Architectures</p>
            </div>

            <div class="card">
                <h3>Candidate Supervised Model Comparison (Validation Partition, N=67,870)</h3>
                <div class="table-responsive">
                    <table class="data-table">
                        <thead>
                            <tr>
                                <th>Candidate Architecture</th>
                                <th>Validation PR-AUC</th>
                                <th>Validation ROC-AUC</th>
                                <th>Brier Score</th>
                                <th>Sensitivity @ 0.50</th>
                                <th>Specificity @ 0.50</th>
                                <th>Survey-Weighted PR-AUC</th>
                                <th>Survey-Weighted ROC-AUC</th>
                            </tr>
                        </thead>
                        <tbody>
                            ${perfData.validation_models.map(m => `
                                <tr class="${m.model_key === 'hist_gradient_boosting_unweighted' ? 'highlight-row' : ''}">
                                    <td><strong>${m.model_name}</strong> ${m.model_key === 'hist_gradient_boosting_unweighted' ? '<span class="badge-champ">Champion</span>' : ''}</td>
                                    <td><strong>${safeToFixed(m.val_pr_auc, 4)}</strong></td>
                                    <td>${safeToFixed(m.val_roc_auc, 4)}</td>
                                    <td>${safeToFixed(m.val_brier_score, 4)}</td>
                                    <td>${safeToFixed(m.sens_at_050 * 100, 1)}%</td>
                                    <td>${safeToFixed(m.spec_at_050 * 100, 1)}%</td>
                                    <td>${safeToFixed(m.survey_weighted_pr_auc, 4)}</td>
                                    <td>${safeToFixed(m.survey_weighted_roc_auc, 4)}</td>
                                </tr>
                            `).join('')}
                        </tbody>
                    </table>
                </div>
            </div>

            <!-- Help text cards explaining metrics -->
            <div class="help-grid" style="display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 1rem; margin-top: 1.5rem;">
                <div class="help-card">
                    <h4>PR-AUC (Primary Metric)</h4>
                    <p>Precision-Recall Area evaluates positive case retrieval under severe 10:1 class imbalance. Unlike ROC-AUC, PR-AUC is sensitive to false-positive alarms in rare-outcome settings.</p>
                </div>
                <div class="help-card">
                    <h4>ROC-AUC (Discrimination)</h4>
                    <p>Receiver Operating Characteristic Area measures overall pairwise rank-ordering between positive and negative respondents across all possible cutoffs. (Champion: 0.8449).</p>
                </div>
                <div class="help-card">
                    <h4>Brier Score (Calibration)</h4>
                    <p>Mean squared difference between predicted risk probabilities and true binary labels. Lower values indicate superior calibration and reliability. (Champion: 0.0700).</p>
                </div>
                <div class="help-card">
                    <h4>Probabilities vs. Cutoffs</h4>
                    <p>Predicted risk is a continuous probability [0, 1]. Classification metrics (Sensitivity, Specificity) depend entirely on the operational threshold applied to these probabilities.</p>
                </div>
            </div>

            <!-- Discrimination Curves -->
            <div class="charts-row" style="display: flex; flex-wrap: wrap; gap: 1.5rem; margin-top: 1.5rem;">
                <div class="chart-container" id="chart-model-roc" style="flex: 1; min-width: 320px; height: 460px;"></div>
                <div class="chart-container" id="chart-model-pr" style="flex: 1; min-width: 320px; height: 460px;"></div>
            </div>

            <!-- Calibration Curve -->
            <div class="chart-container" id="chart-model-calib" style="height: 460px; margin-top: 1.5rem;"></div>

            <!-- Final Holdout Test Summary Table -->
            <div class="card" style="margin-top: 1.5rem;">
                <h3>Final Holdout Test Evaluation (N=67,870, Single Unbiased Forward Pass)</h3>
                <p class="table-caption">Conducted strictly after model and threshold selection were locked. Survey-weighted metrics represent population-weighted point estimates.</p>
                <div class="table-responsive">
                    <table class="data-table">
                        <thead>
                            <tr>
                                <th>Evaluation Mode</th>
                                <th>Threshold Cutoff</th>
                                <th>Sensitivity</th>
                                <th>Specificity</th>
                                <th>Precision (PPV)</th>
                                <th>Negative Pred Value</th>
                                <th>PR-AUC</th>
                                <th>ROC-AUC</th>
                            </tr>
                        </thead>
                        <tbody>
                            ${perfData.test_holdout_summary.map(t => `
                                <tr class="${Math.abs(t.threshold - 0.08) < 0.005 ? 'highlight-row' : ''}">
                                    <td>${t.evaluation_mode}</td>
                                    <td><strong>${t.threshold_label}</strong> ${Math.abs(t.threshold - 0.08) < 0.005 ? '<span class="badge-locked">Locked</span>' : ''}</td>
                                    <td><strong>${safeToFixed(t.sensitivity * 100, 2)}%</strong></td>
                                    <td>${safeToFixed(t.specificity * 100, 2)}%</td>
                                    <td>${safeToFixed(t.precision_ppv * 100, 2)}%</td>
                                    <td><strong>${safeToFixed(t.npv * 100, 2)}%</strong></td>
                                    <td>${safeToFixed(t.pr_auc, 4)}</td>
                                    <td>${safeToFixed(t.roc_auc, 4)}</td>
                                </tr>
                            `).join('')}
                        </tbody>
                    </table>
                </div>
            </div>
        `;

        DashboardCharts.renderModelCurves('chart-model-roc', 'chart-model-pr', perfData.curves, perfData.validation_models);
        DashboardCharts.renderCalibrationCurve('chart-model-calib', perfData.curves, perfData.validation_models);
    },

    /**
     * Section D: Threshold Optimization & Trade-Off Analysis.
     */
    renderThresholdAnalysis(container, thData) {
        const grid = thData.threshold_grid;
        const locked = thData.confusion_matrix_at_locked_080;
        const lockedTh = thData.locked_operational_threshold;

        container.innerHTML = `
            <div class="section-header">
                <h2>4. Decision Threshold Optimization & Trade-Off Analysis</h2>
                <p class="subtitle">Balancing Sensitivity and False-Positive Trade-Offs for Population Surveillance Risk Stratification</p>
            </div>

            <div class="alert-banner info">
                <div class="alert-icon">ℹ️</div>
                <div class="alert-content">
                    <strong>Threshold Policy:</strong> The operational decision threshold is strictly locked at <strong>τ = 0.080</strong> (established via Youden's J criterion on validation data prior to holdout test evaluation). The interactive slider below facilitates hypothetical visual exploration without altering the saved model or the frozen operational protocol.
                </div>
            </div>

            <!-- Interactive Slider Controls -->
            <div class="card controls-card" style="margin-top: 1.5rem;">
                <div class="slider-row">
                    <div style="flex: 1;">
                        <label for="threshold-slider">
                            <strong>Explore Decision Cutoff (τ): </strong>
                            <span id="slider-val-display" style="font-size: 1.25rem; font-weight: bold; color: #0e8388;">0.080</span>
                        </label>
                        <input type="range" id="threshold-slider" min="0.02" max="0.65" step="0.01" value="0.08" class="form-range" style="width: 100%; margin-top: 0.5rem;">
                        <div class="slider-ticks" style="display: flex; justify-content: space-between; font-size: 0.8rem; color: #64748b; margin-top: 0.25rem;">
                            <span>0.02 (Max Recall)</span>
                            <span style="font-weight: bold; color: #8e44ad;">▲ Locked (0.080)</span>
                            <span>0.30</span>
                            <span>0.50 (Standard Default)</span>
                            <span>0.65</span>
                        </div>
                    </div>
                    <div style="margin-left: 1.5rem; display: flex; align-items: flex-end;">
                        <button class="btn btn-outline" id="btn-reset-threshold">Reset to Locked (0.080)</button>
                    </div>
                </div>
            </div>

            <!-- Dynamic Confusion Matrix vs Locked Comparison -->
            <div class="cm-comparison-grid" style="display: flex; flex-wrap: wrap; gap: 1.5rem; margin-top: 1.5rem;">
                <!-- Explored Cutoff Card -->
                <div class="card cm-card" style="flex: 1; min-width: 320px;">
                    <div class="cm-card-header">
                        <h3>Explored Cutoff: <span id="cm-explored-title">τ = 0.080</span></h3>
                        <span class="badge" id="cm-explored-badge">Current Selection</span>
                    </div>
                    <div class="cm-matrix">
                        <div class="cm-cell tp">
                            <div class="cell-label">True Positive (TP)</div>
                            <div class="cell-value" id="cm-tp">${locked.tp.toLocaleString()}</div>
                        </div>
                        <div class="cm-cell fp">
                            <div class="cell-label">False Positive (FP)</div>
                            <div class="cell-value" id="cm-fp">${locked.fp.toLocaleString()}</div>
                        </div>
                        <div class="cm-cell fn">
                            <div class="cell-label">False Negative (FN)</div>
                            <div class="cell-value" id="cm-fn">${locked.fn.toLocaleString()}</div>
                        </div>
                        <div class="cm-cell tn">
                            <div class="cell-label">True Negative (TN)</div>
                            <div class="cell-value" id="cm-tn">${locked.tn.toLocaleString()}</div>
                        </div>
                    </div>
                    <div class="cm-metrics-list">
                        <div class="metric-row"><span>Sensitivity (Recall):</span><strong id="cm-sens">${safeToFixed(locked.sensitivity * 100, 2)}%</strong></div>
                        <div class="metric-row"><span>Specificity:</span><strong id="cm-spec">${safeToFixed(locked.specificity * 100, 2)}%</strong></div>
                        <div class="metric-row"><span>Precision (PPV):</span><strong id="cm-ppv">${safeToFixed(locked.precision_ppv * 100, 2)}%</strong></div>
                        <div class="metric-row"><span>Negative Pred Value (NPV):</span><strong id="cm-npv">${safeToFixed(locked.npv * 100, 2)}%</strong></div>
                        <div class="metric-row"><span>Youden's J Index:</span><strong id="cm-youden">${safeToFixed(locked.youden_j, 4)}</strong></div>
                    </div>
                </div>

                <!-- Locked Reference Card -->
                <div class="card cm-card locked-border" style="flex: 1; min-width: 320px;">
                    <div class="cm-card-header">
                        <h3>Locked Operational Benchmark (τ = 0.080)</h3>
                        <span class="badge-locked">Locked Invariant</span>
                    </div>
                    <div class="cm-matrix">
                        <div class="cm-cell tp">
                            <div class="cell-label">True Positive (TP)</div>
                            <div class="cell-value">${locked.tp.toLocaleString()}</div>
                        </div>
                        <div class="cm-cell fp">
                            <div class="cell-label">False Positive (FP)</div>
                            <div class="cell-value">${locked.fp.toLocaleString()}</div>
                        </div>
                        <div class="cm-cell fn">
                            <div class="cell-label">False Negative (FN)</div>
                            <div class="cell-value">${locked.fn.toLocaleString()}</div>
                        </div>
                        <div class="cm-cell tn">
                            <div class="cell-label">True Negative (TN)</div>
                            <div class="cell-value">${locked.tn.toLocaleString()}</div>
                        </div>
                    </div>
                    <div class="cm-metrics-list">
                        <div class="metric-row"><span>Sensitivity (Recall):</span><strong>${safeToFixed(locked.sensitivity * 100, 2)}%</strong></div>
                        <div class="metric-row"><span>Specificity:</span><strong>${safeToFixed(locked.specificity * 100, 2)}%</strong></div>
                        <div class="metric-row"><span>Precision (PPV):</span><strong>${safeToFixed(locked.precision_ppv * 100, 2)}%</strong></div>
                        <div class="metric-row"><span>Negative Pred Value (NPV):</span><strong>${safeToFixed(locked.npv * 100, 2)}%</strong></div>
                        <div class="metric-row"><span>Youden's J Index:</span><strong>${safeToFixed(locked.youden_j, 4)}</strong></div>
                    </div>
                </div>
            </div>

            <!-- Plotly Interactive Trade-Off Chart -->
            <div class="chart-container" id="chart-threshold-tradeoffs" style="height: 480px; margin-top: 1.5rem;"></div>

            <!-- Epidemiological Rationale Callout -->
            <div class="card" style="margin-top: 1.5rem; border-left: 4px solid #0e8388;">
                <h3>Epidemiological Rationale for the Locked Research Cutoff (0.080)</h3>
                <p>In population surveillance and epidemiological risk stratification research, failing to identify a respondent with self-reported disease history (False Negative) limits the utility of secondary risk-reduction outreach relative to re-contacting an unaffected respondent (False Positive). At the standard default threshold of <strong>0.50</strong>, the model identifies only 10.7% of positive validation cases, failing to identify 89.3% of positive respondents. Setting <strong>τ = 0.080</strong> based on validation Youden's J optimizes the statistical trade-off, achieving an <strong>83.3% sensitivity</strong> and a <strong>97.6% NPV</strong> within this surveillance cohort while accepting a 29.7% false-positive rate. This cutoff is an epidemiological research baseline and does not constitute a validated clinical recommendation.</p>
            </div>
        `;

        const slider = document.getElementById('threshold-slider');
        const sliderDisplay = document.getElementById('slider-val-display');

        const updateThresholdView = (thVal) => {
            sliderDisplay.textContent = safeToFixed(thVal, 3);

            // Find closest precomputed grid point
            let closest = grid[0];
            let minDiff = Math.abs(grid[0].threshold - thVal);
            for (const pt of grid) {
                const diff = Math.abs(pt.threshold - thVal);
                if (diff < minDiff) {
                    minDiff = diff;
                    closest = pt;
                }
            }

            // Update explored CM values
            document.getElementById('cm-explored-title').textContent = `τ = ${safeToFixed(closest.threshold, 3)}`;
            document.getElementById('cm-tp').textContent = closest.tp.toLocaleString();
            document.getElementById('cm-fp').textContent = closest.fp.toLocaleString();
            document.getElementById('cm-fn').textContent = closest.fn.toLocaleString();
            document.getElementById('cm-tn').textContent = closest.tn.toLocaleString();
            document.getElementById('cm-sens').textContent = `${safeToFixed(closest.sensitivity * 100, 2)}%`;
            document.getElementById('cm-spec').textContent = `${safeToFixed(closest.specificity * 100, 2)}%`;
            document.getElementById('cm-ppv').textContent = `${safeToFixed(closest.precision_ppv * 100, 2)}%`;
            document.getElementById('cm-npv').textContent = `${safeToFixed(closest.npv * 100, 2)}%`;
            document.getElementById('cm-youden').textContent = safeToFixed(closest.youden_j, 4);

            // Re-render chart with dynamic marker
            DashboardCharts.renderThresholdTradeoff('chart-threshold-tradeoffs', grid, lockedTh, closest.threshold);
        };

        slider.addEventListener('input', (e) => {
            updateThresholdView(parseFloat(e.target.value));
        });

        document.getElementById('btn-reset-threshold').addEventListener('click', () => {
            slider.value = 0.08;
            updateThresholdView(0.08);
        });

        updateThresholdView(0.08);
    },

    /**
     * Section E: Explainable AI (XAI).
     */
    renderExplainableAi(container, xaiData) {
        const archetypes = xaiData.case_archetypes_summary;

        container.innerHTML = `
            <div class="section-header">
                <h2>5. Explainable AI (XAI) Audits</h2>
                <p class="subtitle">Global Grouped Permutation, Transformed Indicators, Partial Dependence, and Case Archetypes</p>
            </div>

            <!-- Global Feature Importance Controls & Chart -->
            <div class="card controls-card">
                <div class="controls-row">
                    <div>
                        <label><strong>Importance Granularity:</strong></label>
                        <div class="btn-group">
                            <button class="btn btn-sm btn-primary active" id="btn-imp-grouped">Original Survey Constructs (Grouped, N=38)</button>
                            <button class="btn btn-sm btn-outline" id="btn-imp-indicator">Transformed Indicators (Top 25 of 190)</button>
                        </div>
                    </div>
                </div>
            </div>

            <div class="chart-container" id="chart-global-importance" style="height: 540px; margin-top: 1.5rem;"></div>

            <!-- Partial Dependence Section -->
            <div class="card" style="margin-top: 2rem;">
                <div class="controls-row" style="margin-bottom: 1rem;">
                    <div>
                        <label for="select-pdp-var"><strong>Select Continuous Risk Covariate for Partial Dependence:</strong></label>
                        <select id="select-pdp-var" class="form-select">
                            <option value="_AGE80">Continuous Age (_AGE80, 18-80 years)</option>
                            <option value="PHYSHLTH_DAYS">Poor Physical Health Days (PHYSHLTH_DAYS, 0-30 days)</option>
                            <option value="BMI_NUMERIC">Body Mass Index (BMI_NUMERIC, kg/m²)</option>
                        </select>
                    </div>
                </div>
                <div class="chart-container" id="chart-partial-dependence" style="height: 480px;"></div>
            </div>

            <!-- Case Archetypes Section -->
            <div class="card" style="margin-top: 2rem;">
                <div class="section-subheader">
                    <h3>Representative Validation Case Archetype Explanations</h3>
                    <p>Auditing directional feature attributions (Δp) and local Ridge surrogate models (R² > 0.85) across the four confusion quadrants at τ = 0.080.</p>
                </div>

                <div class="controls-row" style="margin-top: 1rem;">
                    <label><strong>Select Case Archetype to Audit:</strong></label>
                    <div class="btn-group" id="archetype-btn-group" style="flex-wrap: wrap;">
                        ${archetypes.map((a, i) => `
                            <button class="btn btn-sm ${i === 0 ? 'btn-primary active' : 'btn-outline'}" data-category="${a.category}">
                                ${a.category} (${a.case_id})
                            </button>
                        `).join('')}
                    </div>
                </div>

                <!-- Case Demographics & Clinical Profile -->
                <div id="case-profile-card" class="case-profile-card" style="margin-top: 1.5rem;"></div>

                <div class="chart-container" id="chart-local-attribution" style="height: 480px; margin-top: 1.5rem;"></div>
            </div>

            <div class="alert-banner info" style="margin-top: 1.5rem;">
                <div class="alert-icon">ℹ️</div>
                <div class="alert-content">
                    <strong>Attribution Methodology Disclosure:</strong> Features represent custom grouped marginal perturbation attributions (Δp) and local Ridge surrogate models. <strong>These are custom model-agnostic interpretability methods, NOT SHAP or LIME.</strong> Attributions describe statistical model associations, not causal clinical interventions.
                </div>
            </div>
        `;

        // Bind Global Importance toggling
        let impMode = "grouped";
        const updateImportanceChart = () => {
            if (impMode === "grouped") {
                DashboardCharts.renderGlobalImportance('chart-global-importance', xaiData.global_grouped_importance, 20);
            } else {
                DashboardCharts.renderIndicatorImportance('chart-global-importance', xaiData.indicator_importance_top30, 25);
            }
        };

        document.getElementById('btn-imp-grouped').addEventListener('click', () => {
            impMode = "grouped";
            document.getElementById('btn-imp-grouped').classList.add('active');
            document.getElementById('btn-imp-indicator').classList.remove('active');
            updateImportanceChart();
        });

        document.getElementById('btn-imp-indicator').addEventListener('click', () => {
            impMode = "indicator";
            document.getElementById('btn-imp-indicator').classList.add('active');
            document.getElementById('btn-imp-grouped').classList.remove('active');
            updateImportanceChart();
        });

        updateImportanceChart();

        // Bind Partial Dependence
        const selectPdp = document.getElementById('select-pdp-var');
        selectPdp.addEventListener('change', (e) => {
            DashboardCharts.renderPartialDependence('chart-partial-dependence', xaiData.partial_dependence_profiles, e.target.value);
        });
        DashboardCharts.renderPartialDependence('chart-partial-dependence', xaiData.partial_dependence_profiles, '_AGE80');

        // Bind Case Archetype selection
        let currentCat = archetypes[0].category;

        const updateCaseView = () => {
            const caseInfo = archetypes.find(a => a.category === currentCat);
            const records = xaiData.case_explanations[currentCat];
            const profileContainer = document.getElementById('case-profile-card');

            profileContainer.innerHTML = `
                <div class="profile-grid" style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; padding: 1rem; background: #f8fafc; border-radius: 6px; border: 1px solid #e2e8f0;">
                    <div><span class="label">Case ID:</span> <strong>${caseInfo.case_id}</strong></div>
                    <div><span class="label">Actual Target:</span> <strong>${caseInfo.actual_target === 1 ? 'Positive (CAD/MI History)' : 'Negative (No CAD/MI)'}</strong></div>
                    <div><span class="label">Predicted Probability:</span> <strong>${safeToFixed(caseInfo.predicted_probability, 4)}</strong></div>
                    <div><span class="label">Classification (@ 0.080):</span> <strong>${caseInfo.predicted_class === 1 ? 'Positive Flag' : 'Negative'}</strong></div>
                    <div><span class="label">Reported Age:</span> <strong>${safeToFixed(caseInfo.age_years, 0)} years</strong></div>
                    <div><span class="label">General Health:</span> <strong>Code ${caseInfo.general_health_code}</strong></div>
                    <div><span class="label">Poor Physical Days:</span> <strong>${caseInfo.physical_unhealthy_days} days/month</strong></div>
                    <div><span class="label">Body Mass Index:</span> <strong>${safeToFixed(caseInfo.bmi_numeric, 1)} kg/m²</strong></div>
                </div>
                <div style="margin-top: 0.75rem; font-size: 0.9rem; color: #475569;">
                    <strong>Archetype Context:</strong> ${caseInfo.archetype_description}
                </div>
            `;

            DashboardCharts.renderLocalAttribution('chart-local-attribution', records, caseInfo, 10);
        };

        const buttons = document.querySelectorAll('#archetype-btn-group button');
        buttons.forEach(btn => {
            btn.addEventListener('click', () => {
                buttons.forEach(b => { b.classList.remove('btn-primary', 'active'); b.classList.add('btn-outline'); });
                btn.classList.add('btn-primary', 'active');
                btn.classList.remove('btn-outline');
                currentCat = btn.getAttribute('data-category');
                updateCaseView();
            });
        });

        updateCaseView();
    },

    /**
     * Section F: Limitations & Responsible Use Guidelines.
     */
    renderLimitations(container) {
        container.innerHTML = `
            <div class="section-header">
                <h2>6. Limitations & Responsible Use Guidelines</h2>
                <p class="subtitle">Ethical, Methodological, and Public Health Guardrails for AI Surveillance</p>
            </div>

            <div class="limitations-grid" style="display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 1.5rem; margin-top: 1rem;">
                <div class="card limitation-card">
                    <h3>1. Cross-Sectional Surveillance Boundary</h3>
                    <p>The CDC BRFSS is a retrospective, cross-sectional telephone surveillance survey. The target variable (<code>_MICHD</code>) records whether an individual has <strong>ever received a physician diagnosis</strong> of CAD/MI. The model is an epidemiological classifier of historical reporting, not a prospective prognostic incidence tool predicting future cardiovascular events.</p>
                </div>

                <div class="card limitation-card">
                    <h3>2. No Clinical Diagnostic Validity</h3>
                    <p>Although the champion GBDT achieves an 83.9% sensitivity and 97.7% negative predictive value within this surveillance sample, <strong>clinical screening utility and rule-out capability have not been established.</strong> Clinical deployment would require prospective randomized diagnostic trials in actual healthcare settings.</p>
                </div>

                <div class="card limitation-card">
                    <h3>3. Complex Survey Weighting Constraints</h3>
                    <p>Models were trained by minimizing unweighted sample loss. Reported survey-weighted population metrics (incorporating <code>_LLCPWT</code>) are descriptive population point estimates without Taylor series linearization for primary sampling units (<code>_PSU</code>) and stratification (<code>_STSTR</code>).</p>
                </div>

                <div class="card limitation-card">
                    <h3>4. Demographic and Fairness Disclosures</h3>
                    <p>Predictive adjustments associated with biological sex (<code>_SEX</code>) and race/ethnicity (<code>_RACE</code>) capture empirical disparities in disease reporting, diagnostic access, and social determinants of health. These demographic adjustments describe historical epidemiological patterns and must <strong>never be used to justify denial of clinical care or healthcare access rationing.</strong></p>
                </div>
            </div>

            <div class="card ethical-use-box" style="margin-top: 2rem; border-left: 5px solid #0e8388;">
                <h3>Responsible Use Framework for Researchers & Public Health Authorities</h3>
                <div class="governance-grid" style="display: flex; flex-wrap: wrap; gap: 2rem; margin-top: 1rem;">
                    <div style="flex: 1; min-width: 280px;">
                        <h4 style="color: #2e7d32; display: flex; align-items: center; gap: 0.5rem;">
                            <span>✔</span> Permissible Academic & Public Health Uses:
                        </h4>
                        <ul class="styled-list">
                            <li>Population health surveillance and prevalence benchmarking across US regions.</li>
                            <li>Epidemiological risk factor clustering and hypothesis generation.</li>
                            <li>Evaluating non-linear risk gradients in public health data.</li>
                            <li>Methodological research in explainable AI for complex survey data.</li>
                        </ul>
                    </div>
                    <div style="flex: 1; min-width: 280px;">
                        <h4 style="color: #c62828; display: flex; align-items: center; gap: 0.5rem;">
                            <span>✖</span> Strictly Prohibited Clinical & Commercial Uses:
                        </h4>
                        <ul class="styled-list">
                            <li>Individual patient clinical diagnosis or medical triage.</li>
                            <li>Insurance underwriting or actuarial pricing decisions.</li>
                            <li>Clinical rule-out without physician diagnostic evaluation.</li>
                            <li>Consumer-facing automated symptom checking or medical advice.</li>
                        </ul>
                    </div>
                </div>
            </div>

            <div class="alert-banner warning" style="margin-top: 2rem;">
                <div class="alert-icon">⚠️</div>
                <div class="alert-content">
                    <strong>Mandatory Research Disclaimer:</strong> This web application was developed strictly as an academic research platform for population health analytics and machine learning auditing. It is not a medical device, has not been reviewed or evaluated by the U.S. FDA or any regulatory agency, and is strictly prohibited for clinical screening, individual diagnosis, or personal health decision-making.
                </div>
            </div>
        `;
    }
};

DashboardViews.safeToFixed = safeToFixed;
DashboardViews.formatCovariateName = formatCovariateName;

window.DashboardViews = DashboardViews;
