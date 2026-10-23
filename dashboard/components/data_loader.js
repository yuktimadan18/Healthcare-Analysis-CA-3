/**
 * data_loader.js
 * --------------
 * Asynchronous data loader and client-side caching module for the
 * "Population Health Analytics & Explainable ML — CDC BRFSS 2024" dashboard.
 * Loads precomputed JSON bundles from the data/ directory with path fallbacks.
 */

const DataLoader = {
    _cache: {},

    /**
     * Fetches and caches a JSON resource by name with path fallbacks.
     * @param {string} filename
     * @returns {Promise<Object>}
     */
    async fetchJson(filename) {
        if (this._cache[filename]) {
            return this._cache[filename];
        }

        // Candidate paths accommodating various local server root configurations
        const candidatePaths = [
            `./data/${filename}`,
            `data/${filename}`,
            `./dashboard/data/${filename}`,
            `dashboard/data/${filename}`,
            `../data/${filename}`
        ];

        let lastError = null;
        for (const url of candidatePaths) {
            try {
                const response = await fetch(url);
                if (response.ok) {
                    const data = await response.json();
                    this._cache[filename] = data;
                    return data;
                }
            } catch (err) {
                lastError = err;
            }
        }

        const msg = `DataLoader Error: Unable to load ${filename} from paths [${candidatePaths.join(', ')}]. ${lastError ? lastError.message : ''}`;
        console.error(msg);
        throw new Error(msg);
    },

    async getOverviewSummary() {
        return this.fetchJson('overview_summary.json');
    },

    async getPopulationHealth() {
        return this.fetchJson('population_health.json');
    },

    async getModelPerformance() {
        return this.fetchJson('model_performance.json');
    },

    async getThresholdAnalysis() {
        return this.fetchJson('threshold_analysis.json');
    },

    async getExplainableAi() {
        return this.fetchJson('explainable_ai.json');
    },

    /**
     * Preloads all bundles in parallel.
     */
    async preloadAll() {
        const [overview, population, performance, threshold, xai] = await Promise.all([
            this.getOverviewSummary(),
            this.getPopulationHealth(),
            this.getModelPerformance(),
            this.getThresholdAnalysis(),
            this.getExplainableAi()
        ]);
        return { overview, population, performance, threshold, xai };
    }
};

window.DataLoader = DataLoader;
