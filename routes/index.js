import { Router } from "express";

const router = Router();

// home page
router.get("/", (req, res) => {
  res.render("index");
});

// analyze submission
router.post("/analyze", async (req, res) => {
  const code = (req.body.code || "").trim();
  const language = (req.body.language || "python").trim().toLowerCase();

  if (!code) {
    return res.status(400).render("index", {
      error: "Please paste some code before submitting.",
    });
  }

  // prevent mega-pastes from nuking your demo
  if (code.length > 60_000) {
    return res.status(413).render("index", {
      error:
        "That code is too large for the demo. Please paste a smaller sample (under ~60k chars).",
    });
  }

  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 8000); // 8 seconds

  try {
    const pyUrl = process.env.PY_SERVICE_URL || "http://127.0.0.1:8000";

    const response = await fetch(`${pyUrl}/analyze`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ code, language }),
      signal: controller.signal,
    });

    clearTimeout(timeout);

    if (!response.ok) {
      const text = await response.text().catch(() => "");
      return res.status(502).render("results", {
        code,
        report: {
          riskLevel: "Error",
          riskScore: null,
          mlRiskLevel: null,
          mlConfidence: null,
          findings: [],
          metrics: {},
          notes: `Python service returned ${response.status}. ${
            text ? "Details: " + text : ""
          }`,
        },
      });
    }

    const pyReport = await response.json();

    // Optional: debug once, then remove
    // console.log("Python summary:", pyReport.summary);

    const report = {
      // Rule-based summary
      riskLevel: pyReport.summary?.riskLevel ?? "N/A",
      riskScore: pyReport.summary?.riskScore ?? null,

      // ML summary (THIS is what you were missing)
      mlRiskLevel: pyReport.summary?.mlRiskLevel ?? null,
      mlConfidence: pyReport.summary?.mlConfidence ?? null,

      findings: pyReport.findings ?? [],
      metrics: pyReport.metrics ?? {},
      features: pyReport.features ?? {},
      featureInfo: pyReport.featureInfo ?? {},
      notes: pyReport.notes ?? "",
    };

    res.render("results", { code, report });
  } catch (err) {
    clearTimeout(timeout);

    const isTimeout = err?.name === "AbortError";
    res.status(500).render("results", {
      code,
      report: {
        riskLevel: "Error",
        riskScore: null,
        mlRiskLevel: null,
        mlConfidence: null,
        findings: [],
        metrics: {},
        notes: isTimeout
          ? "Analysis timed out. Is the Python service running and responsive?"
          : "Could not reach Python analysis service.",
      },
    });
  }
});

export default router;
