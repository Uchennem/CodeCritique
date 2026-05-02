from fastapi import FastAPI
from pydantic import BaseModel
from typing import Any, Dict, List, Optional
import ast
import re
import joblib
from pathlib import Path
import pandas as pd

# The exact order the ML model expects features in.
FEATURE_ORDER = [
    "loc",
    "max_nesting",
    "branch_points",
    "avg_fn_len",
    "max_fn_len",
    "num_errors",
    "num_warnings",
    "comment_ratio",
]

FEATURE_DESCRIPTIONS = {
    "loc": "Total lines of code in the submitted snippet.",
    "max_nesting": "Deepest nesting level found in Python control structures.",
    "branch_points": "Approximate count of decision points and branches.",
    "avg_fn_len": "Average function length in lines.",
    "max_fn_len": "Length of the largest function in lines.",
    "num_errors": "Number of high-severity findings, such as unsafe execution patterns.",
    "num_warnings": "Number of medium-severity maintainability or safety findings.",
    "comment_ratio": "Ratio of comment lines to total lines of code.",
}

# Load model safely (service should still run without it).
try:
    model = joblib.load(Path(__file__).with_name("risk_model.joblib"))
    print("ML model loaded:", model is not None)
except Exception:
    model = None

app = FastAPI()


class AnalyzeRequest(BaseModel):
    code: str
    language: Optional[str] = "python"  # default to python for now


def _basic_text_metrics(code: str) -> Dict[str, Any]:
    lines = code.splitlines()
    loc = len(lines)
    non_empty = [ln for ln in lines if ln.strip()]
    blank_lines = loc - len(non_empty)

    # simple comment detection (Python-ish + JS-ish)
    comment_lines = 0
    for ln in lines:
        s = ln.strip()
        if s.startswith("#") or s.startswith("//"):
            comment_lines += 1

    comment_ratio = (comment_lines / loc) if loc else 0.0
    return {
        "loc": loc,
        "blankLines": blank_lines,
        "commentLines": comment_lines,
        "commentRatio": round(comment_ratio, 3),
    }


class NestingAndComplexityVisitor(ast.NodeVisitor):
    """
    Computes:
      - max nesting depth (if/for/while/try/with/function/class)
      - branch count (rough complexity proxy)
      - function lengths + counts
    """

    def __init__(self):
        self.current_depth = 0
        self.max_depth = 0

        self.branch_points = 0
        self.function_count = 0
        self.class_count = 0
        self.function_lengths: List[int] = []

        self._function_stack: List[int] = []  # track start line of current function

    def _enter_block(self):
        self.current_depth += 1
        self.max_depth = max(self.max_depth, self.current_depth)

    def _exit_block(self):
        self.current_depth -= 1

    def generic_visit(self, node):
        super().generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef):
        self.function_count += 1
        self._enter_block()
        self._function_stack.append(getattr(node, "lineno", 0))

        self.generic_visit(node)

        start = self._function_stack.pop() if self._function_stack else 0
        end = getattr(node, "end_lineno", getattr(node, "lineno", start))
        length = max(0, (end - start + 1))
        self.function_lengths.append(length)

        self._exit_block()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        # treat async like normal
        self.visit_FunctionDef(node)  # type: ignore

    def visit_ClassDef(self, node: ast.ClassDef):
        self.class_count += 1
        self._enter_block()
        self.generic_visit(node)
        self._exit_block()

    # blocks that increase nesting
    def visit_If(self, node: ast.If):
        self.branch_points += 1
        self._enter_block()
        self.generic_visit(node)
        self._exit_block()

    def visit_For(self, node: ast.For):
        self.branch_points += 1
        self._enter_block()
        self.generic_visit(node)
        self._exit_block()

    def visit_While(self, node: ast.While):
        self.branch_points += 1
        self._enter_block()
        self.generic_visit(node)
        self._exit_block()

    def visit_Try(self, node: ast.Try):
        self.branch_points += 1
        self._enter_block()
        self.generic_visit(node)
        self._exit_block()

    def visit_With(self, node: ast.With):
        self._enter_block()
        self.generic_visit(node)
        self._exit_block()

    def visit_BoolOp(self, node: ast.BoolOp):
        # a && b && c adds branching-ish decision points
        self.branch_points += max(0, len(node.values) - 1)
        self.generic_visit(node)

    def visit_Compare(self, node: ast.Compare):
        # chained comparisons a < b < c adds decision points
        self.branch_points += max(0, len(node.ops) - 1)
        self.generic_visit(node)


def _python_ast_metrics(code: str) -> Dict[str, Any]:
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return {
            "parseOk": False,
            "syntaxError": {
                "message": str(e),
                "line": getattr(e, "lineno", None),
                "column": getattr(e, "offset", None),
            },
        }

    v = NestingAndComplexityVisitor()
    v.visit(tree)

    avg_fn_len = round(sum(v.function_lengths) / len(v.function_lengths), 2) if v.function_lengths else 0.0
    max_fn_len = max(v.function_lengths) if v.function_lengths else 0

    return {
        "parseOk": True,
        "functions": v.function_count,
        "classes": v.class_count,
        "avgFunctionLength": avg_fn_len,
        "maxFunctionLength": max_fn_len,
        "maxNestingDepth": v.max_depth,
        "branchPoints": v.branch_points,
    }


def _security_findings(code: str) -> List[Dict[str, Any]]:
    findings = []
    patterns = [
        (r"\beval\s*\(", "error", "Use of eval() detected. This can execute arbitrary code."),
        (r"\bexec\s*\(", "error", "Use of exec() detected. This can execute arbitrary code."),
        (r"\bsubprocess\.(Popen|run|call)\b", "warning", "subprocess usage detected. Ensure inputs are validated."),
        (r"shell\s*=\s*True", "error", "shell=True detected. High risk of command injection with untrusted input."),
        (r"\bpickle\.loads\s*\(", "warning", "pickle.loads() detected. Unsafe on untrusted data."),
    ]

    for pat, severity, msg in patterns:
        if re.search(pat, code):
            findings.append({"severity": severity, "message": msg})

    return findings


def _maintainability_findings(metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
    findings = []

    loc = metrics.get("loc", 0)
    depth = metrics.get("maxNestingDepth", 0)
    max_fn = metrics.get("maxFunctionLength", 0)
    branches = metrics.get("branchPoints", 0)
    comment_ratio = metrics.get("commentRatio", 0.0)

    if loc > 300:
        findings.append({"severity": "warning", "message": f"Large file ({loc} LOC). Consider splitting into modules."})

    if depth >= 5:
        findings.append({"severity": "warning", "message": f"Deep nesting detected (depth {depth}). Consider refactoring into smaller functions."})

    if max_fn >= 60:
        findings.append({"severity": "warning", "message": f"Very long function detected ({max_fn} lines). Consider extracting helper functions."})

    if branches >= 30:
        findings.append({"severity": "warning", "message": f"High branching/decision points ({branches}). Complexity may be high; consider simplifying logic."})

    if loc >= 80 and comment_ratio < 0.05:
        findings.append({"severity": "info", "message": "Low comment density for a non-trivial file. Consider adding docstrings/comments for key logic."})

    return findings


def _score_risk(metrics: Dict[str, Any], findings: List[Dict[str, Any]]) -> Dict[str, Any]:
    score = 0.10  # base

    loc = metrics.get("loc", 0)
    depth = metrics.get("maxNestingDepth", 0)
    avg_fn = metrics.get("avgFunctionLength", 0.0)
    branches = metrics.get("branchPoints", 0)

    if loc > 80:
        score += 0.10
    if loc > 300:
        score += 0.15

    if depth >= 4:
        score += 0.15
    if depth >= 6:
        score += 0.10

    if avg_fn >= 25:
        score += 0.10
    if branches >= 20:
        score += 0.10
    if branches >= 40:
        score += 0.10

    for f in findings:
        sev = f.get("severity")
        if sev == "error":
            score += 0.35
        elif sev == "warning":
            score += 0.15
        elif sev == "info":
            score += 0.05

    score = max(0.0, min(1.0, score))
    if score >= 0.70:
        level = "High"
    elif score >= 0.40:
        level = "Medium"
    else:
        level = "Low"

    return {"riskScore": round(score, 2), "riskLevel": level}


def extract_feature_vector(metrics: Dict[str, Any], findings: List[Dict[str, Any]]) -> Dict[str, float]:
    return {
        "loc": float(metrics.get("loc", 0)),
        "max_nesting": float(metrics.get("maxNestingDepth", 0)),
        "branch_points": float(metrics.get("branchPoints", 0)),
        "avg_fn_len": float(metrics.get("avgFunctionLength", 0.0)),
        "max_fn_len": float(metrics.get("maxFunctionLength", 0)),
        "num_errors": float(sum(1 for f in findings if f.get("severity") == "error")),
        "num_warnings": float(sum(1 for f in findings if f.get("severity") == "warning")),
        "comment_ratio": float(metrics.get("commentRatio", 0.0)),
    }


@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    code = (req.code or "").strip()
    if not code:
        return {
            "summary": {"riskLevel": "N/A", "riskScore": None, "mlRiskLevel": None, "mlConfidence": None},
            "metrics": {"loc": 0},
            "features": {},
            "featureInfo": FEATURE_DESCRIPTIONS,
            "findings": [],
            "notes": "No code provided.",
        }

    # 1) text metrics (works for any language)
    metrics: Dict[str, Any] = _basic_text_metrics(code)

    # 2) python AST metrics (only if python)
    if (req.language or "python").lower() == "python":
        ast_metrics = _python_ast_metrics(code)
        metrics.update(ast_metrics)

        if ast_metrics.get("parseOk") is False:
            findings = [{
                "severity": "error",
                "message": "Python syntax error: unable to parse code.",
                "line": ast_metrics.get("syntaxError", {}).get("line"),
            }]
            summary = _score_risk(metrics, findings)
            return {
                "summary": {"riskLevel": summary["riskLevel"], "riskScore": summary["riskScore"], "mlRiskLevel": None, "mlConfidence": None},
                "metrics": metrics,
                "features": {},
                "featureInfo": FEATURE_DESCRIPTIONS,
                "findings": findings,
                "notes": "Parsing failed.",
            }

    # 3) findings
    findings: List[Dict[str, Any]] = []
    findings.extend(_security_findings(code))
    findings.extend(_maintainability_findings(metrics))

    # 4) rule-based score
    summary = _score_risk(metrics, findings)

    # 5) ML feature extraction + prediction (safe)
    features = extract_feature_vector(metrics, findings)

    # Ensure all required features exist before building X
    missing = [name for name in FEATURE_ORDER if name not in features]
    ml_prediction = None
    ml_confidence = None

    if model is not None and not missing:
        X = pd.DataFrame([{name: features[name] for name in FEATURE_ORDER}])

        try:
            ml_prediction = model.predict(X)[0]
            if isinstance(ml_prediction, str):
                ml_prediction = ml_prediction.capitalize()

            if hasattr(model, "predict_proba"):
                ml_confidence = float(max(model.predict_proba(X)[0]))
        except Exception:
            # If ML fails for any reason, keep the API working.
            ml_prediction = None
            ml_confidence = None

    return {
        "summary": {
            "riskLevel": summary["riskLevel"],
            "riskScore": summary["riskScore"],
            "mlRiskLevel": ml_prediction,
            "mlConfidence": round(ml_confidence, 2) if ml_confidence is not None else None,
        },
        "metrics": metrics,
        "features": features,
        "featureInfo": FEATURE_DESCRIPTIONS,
        "findings": findings,
        "notes": "Rule-based scoring remains primary while the extracted feature set supports ML training and comparison.",
    }
