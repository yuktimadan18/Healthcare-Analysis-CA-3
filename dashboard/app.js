/**
 * app.js
 * ------
 * Main application coordinator for the
 * "Population Health Analytics & Explainable ML — CDC BRFSS 2024" dashboard.
 * Manages data preloading, client-side routing, and view lifecycle.
 */

document.addEventListener('DOMContentLoaded', async () => {
    const loadingOverlay = document.getElementById('loading-overlay');
    const loadingText = document.getElementById('loading-text');
    const errorContainer = document.getElementById('error-container');
    const navLinks = document.querySelectorAll('.nav-link');
    const sections = {
        'overview': document.getElementById('section-overview'),
        'population': document.getElementById('section-population'),
        'performance': document.getElementById('section-performance'),
        'threshold': document.getElementById('section-threshold'),
        'xai': document.getElementById('section-xai'),
        'limitations': document.getElementById('section-limitations')
    };

    let dashboardData = null;
    const renderedSections = new Set();

    // Check Plotly availability
    const isPlotlyAvailable = typeof Plotly !== 'undefined';
    if (!isPlotlyAvailable) {
        console.warn("Plotly.js is not detected. Internet access is required to render interactive charts from CDN.");
        const banner = document.createElement('div');
        banner.className = 'alert-banner warning';
        banner.style.margin = '1rem 0';
        banner.innerHTML = `
            <div class="alert-icon">⚠️</div>
            <div class="alert-content">
                <strong>Plotly.js CDN Notice:</strong> Interactive charts require external Plotly.js from CDN. If offline, static data tables remain fully functional, but charts may not render.
            </div>
        `;
        document.querySelector('.app-main').prepend(banner);
    }

    /**
     * Renders a specific section if not already rendered.
     */
    function activateSection(sectionId) {
        if (!sections[sectionId]) sectionId = 'overview';

        // Update nav links
        navLinks.forEach(link => {
            if (link.getAttribute('data-section') === sectionId) {
                link.classList.add('active');
            } else {
                link.classList.remove('active');
            }
        });

        // Hide all sections, show active
        Object.keys(sections).forEach(id => {
            if (id === sectionId) {
                sections[id].classList.add('active');
            } else {
                sections[id].classList.remove('active');
            }
        });

        // Render on demand if not yet rendered
        if (dashboardData && !renderedSections.has(sectionId)) {
            try {
                switch (sectionId) {
                    case 'overview':
                        DashboardViews.renderOverview(sections['overview'], dashboardData.overview);
                        break;
                    case 'population':
                        DashboardViews.renderPopulationHealth(sections['population'], dashboardData.population);
                        break;
                    case 'performance':
                        DashboardViews.renderModelPerformance(sections['performance'], dashboardData.performance);
                        break;
                    case 'threshold':
                        DashboardViews.renderThresholdAnalysis(sections['threshold'], dashboardData.threshold);
                        break;
                    case 'xai':
                        DashboardViews.renderExplainableAi(sections['xai'], dashboardData.xai);
                        break;
                    case 'limitations':
                        DashboardViews.renderLimitations(sections['limitations']);
                        break;
                }
                renderedSections.add(sectionId);
            } catch (err) {
                console.error(`Error rendering section ${sectionId}:`, err);
                sections[sectionId].innerHTML = `
                    <div class="alert-banner warning">
                        <strong>Rendering Error:</strong> Unable to render section ${sectionId}. Details: ${err.message}
                    </div>
                `;
            }
        }

        // Trigger Plotly relayout/resize for active charts
        if (isPlotlyAvailable) {
            setTimeout(() => {
                const charts = sections[sectionId].querySelectorAll('.chart-container');
                charts.forEach(c => {
                    try { Plotly.Plots.resize(c); } catch (e) {}
                });
            }, 100);
        }

        // Update URL hash
        if (window.location.hash !== `#${sectionId}`) {
            history.pushState(null, null, `#${sectionId}`);
        }
    }

    // Set up click handlers on navigation links
    navLinks.forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            const targetSection = link.getAttribute('data-section');
            activateSection(targetSection);
        });
    });

    // Handle back/forward hash changes
    window.addEventListener('hashchange', () => {
        const hash = window.location.hash.replace('#', '') || 'overview';
        activateSection(hash);
    });

    // Handle window resize for charts
    window.addEventListener('resize', () => {
        if (!isPlotlyAvailable) return;
        const activeSec = document.querySelector('.dashboard-section.active');
        if (activeSec) {
            activeSec.querySelectorAll('.chart-container').forEach(c => {
                try { Plotly.Plots.resize(c); } catch (e) {}
            });
        }
    });

    // Main initialization sequence
    try {
        if (loadingText) loadingText.textContent = "Loading precomputed population health data...";
        dashboardData = await DataLoader.preloadAll();

        if (loadingOverlay) {
            loadingOverlay.style.opacity = '0';
            setTimeout(() => { loadingOverlay.style.display = 'none'; }, 300);
        }

        // Determine initial section from URL hash or default to overview
        const initialHash = window.location.hash.replace('#', '') || 'overview';
        activateSection(initialHash);

    } catch (err) {
        console.error("Critical Dashboard Startup Error:", err);
        if (loadingOverlay) loadingOverlay.style.display = 'none';
        if (errorContainer) {
            errorContainer.style.display = 'block';
            errorContainer.innerHTML = `
                <div class="alert-banner warning" style="max-width: 800px; margin: 3rem auto; padding: 2rem;">
                    <div class="alert-icon" style="font-size: 2rem;">⚠️</div>
                    <div class="alert-content">
                        <h3 style="margin-bottom: 0.5rem; color: #92400e;">Dashboard Startup Error</h3>
                        <p style="margin-bottom: 1rem;">Failed to load precomputed data bundles. Please ensure the local HTTP server is running from the project root or the <code>dashboard/</code> directory.</p>
                        <p style="font-family: monospace; font-size: 0.85rem; background: #fff; padding: 0.75rem; border-radius: 4px; border: 1px solid #fde68a;">
                            ${err.message}
                        </p>
                        <div style="margin-top: 1.5rem;">
                            <strong>Local Launch Instructions:</strong>
                            <pre style="background: #1e293b; color: #f8fafc; padding: 0.75rem; border-radius: 4px; margin-top: 0.5rem; font-size: 0.85rem;">
cd "C:\\Users\\Asus\\Desktop\\7 SEM\\7-HA\\CA-3\\Project"
.venv\\Scripts\\python.exe -m http.server 8501 --directory dashboard</pre>
                        </div>
                    </div>
                </div>
            `;
        }
    }
});
