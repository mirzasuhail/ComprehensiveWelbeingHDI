document.addEventListener("DOMContentLoaded", () => {
    // Form & Action Elements
    const form = document.getElementById("hdiForm");
    const submitBtn = document.getElementById("submitBtn");
    const btnText = document.getElementById("btnText");
    const errorBanner = document.getElementById("errorBanner");
    const errorText = document.getElementById("errorText");
    const resetBtn = document.getElementById("resetBtn");
    const resultSection = document.getElementById("resultSection");

    // Result Elements
    const scoreValue = document.getElementById("scoreValue");
    const tierBadge = document.getElementById("tierBadge");
    const resultSpectrumFill = document.getElementById("resultSpectrumFill");
    const resultPointerMarker = document.getElementById("resultPointerMarker");
    const markerLabel = document.getElementById("markerLabel");

    // Profile Elements
    const resLife = document.getElementById("resLife");
    const resExp = document.getElementById("resExp");
    const resMean = document.getElementById("resMean");
    const resGni = document.getElementById("resGni");

    // Form Submission Handler
    form.addEventListener("submit", async (e) => {
        e.preventDefault();
        hideError();

        // 1. Extract Numeric Inputs
        const lifeExp = parseFloat(document.getElementById("life_expectancy").value);
        const expSch = parseFloat(document.getElementById("expected_schooling").value);
        const meanSch = parseFloat(document.getElementById("mean_schooling").value);
        const gni = parseFloat(document.getElementById("gni_per_capita").value);

        // 2. Client-side Quick Validation
        if (isNaN(lifeExp) || isNaN(expSch) || isNaN(meanSch) || isNaN(gni)) {
            showError("Please enter valid numeric values for all four development indicators.");
            return;
        }

        if (meanSch > expSch) {
            showError(`Logical Inconsistency: Mean schooling (${meanSch} yrs) cannot exceed Expected schooling (${expSch} yrs).`);
            return;
        }

        // 3. Multi-phase Micro-Loading Animation
        setLoadingState(true, "Analyzing indicators...");

        setTimeout(async () => {
            setLoadingState(true, "Generating estimate...");

            try {
                const response = await fetch("/predict", {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json"
                    },
                    body: JSON.stringify({
                        life_expectancy: lifeExp,
                        expected_schooling: expSch,
                        mean_schooling: meanSch,
                        gni_per_capita: gni
                    })
                });

                const data = await response.json();

                if (!data.success) {
                    showError(data.error || "Prediction request failed.");
                    setLoadingState(false, "Estimate HDI");
                    return;
                }

                // Render Results with NLPxAI Count-up Animation
                renderResult(data);
                setLoadingState(false, "Estimate HDI");

            } catch (err) {
                showError("Network error: Could not connect to backend server.");
                setLoadingState(false, "Estimate HDI");
            }

        }, 350);
    });

    // Reset Button Handler
    resetBtn.addEventListener("click", () => {
        // Hide result section
        resultSection.classList.add("hidden");
        hideError();

        // Scroll back to workspace form smoothly
        const workspace = document.getElementById("workspace");
        if (workspace) {
            workspace.scrollIntoView({ behavior: "smooth", block: "start" });
        }
    });

    // Helper: Render Result View & Count-up Score
    function renderResult(data) {
        // Reveal result section
        resultSection.classList.remove("hidden");

        const targetHdi = parseFloat(data.hdi);
        const formattedHdi = data.hdi_formatted || targetHdi.toFixed(3);

        // 1. Count-up Number Animation (0.000 -> targetHdi)
        animateCountUp(targetHdi, formattedHdi);

        // 2. Tier Badge
        tierBadge.textContent = data.tier.toUpperCase();
        tierBadge.className = `tier-badge ${data.tier_code}`;

        // 3. Spectrum Marker Position
        const percentage = Math.min(100, Math.max(0, targetHdi * 100));
        resultSpectrumFill.style.width = `${percentage}%`;
        resultPointerMarker.style.left = `${percentage}%`;
        markerLabel.textContent = formattedHdi;

        // 4. Input Profile Table
        resLife.textContent = `${data.inputs.life_expectancy} yrs`;
        resExp.textContent = `${data.inputs.expected_schooling} yrs`;
        resMean.textContent = `${data.inputs.mean_schooling} yrs`;
        resGni.textContent = `$${data.inputs.gni_per_capita} PPP`;

        // Smooth Scroll to Result Section
        setTimeout(() => {
            resultSection.scrollIntoView({ behavior: "smooth", block: "start" });
        }, 100);
    }

    // Realistic Count-Up Animation (NLPxAI Output Transition)
    function animateCountUp(targetValue, finalFormatted) {
        let current = 0.000;
        const duration = 600; // ms
        const steps = 30;
        const increment = targetValue / steps;
        const stepTime = duration / steps;

        const timer = setInterval(() => {
            current += increment;
            if (current >= targetValue) {
                scoreValue.textContent = finalFormatted;
                clearInterval(timer);
            } else {
                scoreValue.textContent = current.toFixed(3);
            }
        }, stepTime);
    }

    // Helper: Button Loading State
    function setLoadingState(isLoading, text) {
        if (isLoading) {
            submitBtn.disabled = true;
            btnText.textContent = text;
            submitBtn.style.opacity = "0.75";
        } else {
            submitBtn.disabled = false;
            btnText.textContent = text;
            submitBtn.style.opacity = "1.0";
        }
    }

    // Helper: Error Banner
    function showError(msg) {
        errorText.textContent = msg;
        errorBanner.classList.remove("hidden");
    }

    function hideError() {
        errorBanner.classList.add("hidden");
    }
});
