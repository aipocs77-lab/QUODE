"""
eval_metrics.py — QODE evaluation framework for context quality before LLM processing.

Evaluates incoming context against QODE propensities, readiness scores, and engineering
practice definitions. Ensures all graph data sent to LLM meets quality thresholds.

Evaluation gates (run before LLM context injection):
  - evaluate_graph_context() → validates propensity scores + readiness + entity completeness
  - evaluate_engineering_practices() → verifies practice definitions + role mappings
  - evaluate_propensities() → checks Practice, Technology, Collaboration scores
  - evaluate_readiness_score() → validates RAG (Red/Amber/Green) thresholds
  - evaluate_people_patterns() → maps propensity combinations to valid patterns
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


# ==============================================================================
# Enums and Data Classes
# ==============================================================================

class RAGColor(Enum):
    """Readiness Assessment (RAG) color scheme from QODE methodologies."""
    RED = 1      # 🔴 Little or no readiness; needs substantial improvement
    YELLOW = 2   # 🟡 Partial readiness; needs improvement and focus
    GREEN = 3    # 🟢 Substantial readiness; well established


class PropensityLevel(Enum):
    """Three-tier propensity scoring."""
    LOW = 1       # Managerial / Documentation-Based / Silo-Based
    MEDIUM = 2    # Engineering / Configuration-Centric / Intra-Group
    HIGH = 3      # High Automation / Coding-Centric / Inter-Group


@dataclass
class PropensityScore:
    """Scores for Practice, Technology Usage, and Collaboration propensities."""
    practice: PropensityLevel
    technology_usage: PropensityLevel
    collaboration: PropensityLevel

    def to_dict(self) -> dict[str, int]:
        return {
            "practice": self.practice.value,
            "technology_usage": self.technology_usage.value,
            "collaboration": self.collaboration.value,
        }

    def pattern_key(self) -> str:
        """Return (p, t, c) pattern key for people pattern lookup."""
        return f"{self.practice.value}-{self.technology_usage.value}-{self.collaboration.value}"


@dataclass
class EvaluationResult:
    """Outcome of a single evaluation gate."""
    gate_name: str
    passed: bool
    message: str
    severity: str = "INFO"  # INFO, WARNING, ERROR
    details: dict[str, Any] | None = None

    def __str__(self) -> str:
        status = "✓ PASS" if self.passed else "✗ FAIL"
        return f"[{self.gate_name}] {status}: {self.message}"


# ==============================================================================
# Engineering Practices Evaluator
# ==============================================================================

ENGINEERING_PRACTICES = {
    "Requirements Engineering": {
        "roles": ["Product Owner", "Business Analyst", "IT Lead"],
        "tools": ["Jira", "Confluence", "Azure DevOps Boards"],
        "description": "Capturing, translating, categorizing, and integrating requirements; traceability automation",
    },
    "Code Engineering": {
        "roles": ["Developer", "Tech Lead"],
        "tools": ["GitHub", "GitLab", "Bitbucket", "SonarQube"],
        "description": "Modularity, microservices architecture, in-line code quality, model-driven code generation",
    },
    "Data Engineering": {
        "roles": ["Data Architect", "Data Modeler"],
        "tools": ["DBaaS", "ETL Tools", "Data Pipeline"],
        "description": "Schema optimization, unstructured data management, data migration, data security",
    },
    "Quality Engineering": {
        "roles": ["QA", "Tester", "QA Lead"],
        "tools": ["Selenium", "JUnit", "pytest", "TestNG", "Cypress"],
        "description": "TDD/BDD, in-sprint automation, regression and performance test automation, shift-left testing",
    },
    "Build & Release Engineering": {
        "roles": ["DevOps Lead", "Release Manager", "DevOps Engineer"],
        "tools": ["Jenkins", "GitHub Actions", "CircleCI", "Nexus", "JFrog"],
        "description": "CI/CD frameworks, toolchain integration, zero-downtime deployment, release pipeline reliability",
    },
    "Environment Engineering": {
        "roles": ["Ops-Infra", "Cloud Engineer", "SRE"],
        "tools": ["Terraform", "Ansible", "Puppet", "CloudFormation", "Helm"],
        "description": "Infrastructure-as-code, environment provisioning automation, configuration management, containerization",
    },
    "Service Operations Engineering": {
        "roles": ["Ops-Rel", "Support Lead", "Production Support"],
        "tools": ["ServiceNow", "Jira", "Splunk", "ELK Stack"],
        "description": "Production monitoring, automated incident detection and resolution, feedback loops to SDLC",
    },
    "Security Engineering": {
        "roles": ["Security Engineer", "IT Security"],
        "tools": ["Snyk", "Checkmarx", "OWASP ZAP", "Twistlock"],
        "description": "SAST, DAST, SCA, secrets scanning, container scanning, DevSecOps",
    },
    "Reliability Engineering": {
        "roles": ["SRE", "Ops-Rel"],
        "tools": ["Prometheus", "Grafana", "Datadog", "Dynatrace"],
        "description": "Monitoring-based failure detection, self-healing, chaos engineering, SLO/SLA management",
    },
}


def evaluate_engineering_practices(
    practice_name: str,
    roles_found: list[str],
    tools_found: list[str],
) -> EvaluationResult:
    """Validate that a practice definition includes roles and tools.

    Args:
        practice_name: Name of the engineering practice (e.g. "Code Engineering")
        roles_found: Roles referenced in current context
        tools_found: Tools referenced in current context

    Returns:
        EvaluationResult indicating whether the practice is properly scoped.
    """
    if practice_name not in ENGINEERING_PRACTICES:
        return EvaluationResult(
            gate_name="EngineeringPracticesValidator",
            passed=False,
            message=f"Unknown practice: {practice_name}",
            severity="ERROR",
            details={"available_practices": list(ENGINEERING_PRACTICES.keys())}
        )

    practice = ENGINEERING_PRACTICES[practice_name]
    expected_roles = set(r.lower() for r in practice["roles"])
    expected_tools = set(t.lower() for t in practice["tools"])

    found_roles = set(r.lower() for r in roles_found)
    found_tools = set(t.lower() for t in tools_found)

    missing_roles = expected_roles - found_roles
    missing_tools = expected_tools - found_tools

    if not found_roles and not found_tools:
        return EvaluationResult(
            gate_name="EngineeringPracticesValidator",
            passed=False,
            message=f"{practice_name}: no roles or tools found",
            severity="ERROR",
        )

    passed = len(missing_roles) == 0 and len(missing_tools) == 0
    severity = "INFO" if passed else "WARNING"

    message = f"{practice_name}: "
    if found_roles:
        message += f"roles={list(found_roles)[:3]} "
    if found_tools:
        message += f"tools={list(found_tools)[:3]}"
    if missing_roles or missing_tools:
        message += f" [missing: roles={list(missing_roles)} tools={list(missing_tools)}]"

    return EvaluationResult(
        gate_name="EngineeringPracticesValidator",
        passed=passed,
        message=message,
        severity=severity,
        details={
            "found_roles": list(found_roles),
            "found_tools": list(found_tools),
            "missing_roles": list(missing_roles),
            "missing_tools": list(missing_tools),
        }
    )


# ==============================================================================
# Propensity Evaluator
# ==============================================================================

def evaluate_propensities(
    practice_score: int | PropensityLevel,
    technology_score: int | PropensityLevel,
    collaboration_score: int | PropensityLevel,
) -> EvaluationResult:
    """Validate propensity scores are in valid range (1, 2, or 3).

    Args:
        practice_score: Practice propensity (Low=1, Medium=2, High=3)
        technology_score: Technology usage propensity
        collaboration_score: Collaboration propensity

    Returns:
        EvaluationResult with propensity summary.
    """
    scores = []
    valid = True

    for label, score in [
        ("Practice", practice_score),
        ("Technology", technology_score),
        ("Collaboration", collaboration_score),
    ]:
        val = score.value if isinstance(score, PropensityLevel) else score
        if val not in (1, 2, 3):
            valid = False
            logger.warning(f"{label} propensity {val} out of valid range [1,3]")
        scores.append(val)

    if valid:
        try:
            propensity = PropensityScore(
                practice=PropensityLevel(scores[0]),
                technology_usage=PropensityLevel(scores[1]),
                collaboration=PropensityLevel(scores[2]),
            )
            return EvaluationResult(
                gate_name="PropensitiesValidator",
                passed=True,
                message=f"Propensities valid: {propensity.pattern_key()}",
                severity="INFO",
                details=propensity.to_dict(),
            )
        except ValueError as e:
            return EvaluationResult(
                gate_name="PropensitiesValidator",
                passed=False,
                message=f"Invalid propensity values: {e}",
                severity="ERROR",
            )

    return EvaluationResult(
        gate_name="PropensitiesValidator",
        passed=False,
        message=f"Propensity scores out of range: P={scores[0]}, T={scores[1]}, C={scores[2]}",
        severity="ERROR",
    )


# ==============================================================================
# People Patterns Evaluator
# ==============================================================================

PEOPLE_PATTERNS = {
    "1-1-1": ("Traditional Waterfall + Ops Silo", "Waterfall dev, manual testing, completely separate teams"),
    "1-1-2": ("Collaborative Waterfall", "Teams collaborate but low automation; Ops independent"),
    "1-2-2": ("Tool-Rich but Engineering-Weak", "Coding-centric tools, high collaboration, weak automation mindset"),
    "3-3-3": ("Mature DevOps / NoOps Emerging", "Unified Dev+Ops, coded pipelines, SRE active"),
    "3-3-1": ("Platform Engineering", "High automation, coding-centric, independent sub-teams"),
    "3-1-1": ("Niche IT-for-IT Team", "Advanced automation/research, no tools/collab, innovation-only"),
    "3-1-2": ("Script-Heavy Collaborative", "High practice + collab + doc/config; TDD/BDD + microservices"),
    "1-2-1": ("Tool-Overloaded Siloed Team", "High tool investment, silo culture, no engineering mindset"),
}


def evaluate_people_patterns(
    pattern_key: str,
) -> EvaluationResult:
    """Validate that a propensity combination maps to a known people pattern.

    Args:
        pattern_key: String like "3-3-3" (practice-technology-collaboration)

    Returns:
        EvaluationResult with pattern name and characteristics.
    """
    if pattern_key not in PEOPLE_PATTERNS:
        return EvaluationResult(
            gate_name="PeoplePatternsValidator",
            passed=False,
            message=f"Unknown pattern: {pattern_key}",
            severity="ERROR",
            details={"available_patterns": list(PEOPLE_PATTERNS.keys())}
        )

    name, description = PEOPLE_PATTERNS[pattern_key]
    return EvaluationResult(
        gate_name="PeoplePatternsValidator",
        passed=True,
        message=f"Pattern {pattern_key}: {name}",
        severity="INFO",
        details={"pattern_name": name, "description": description}
    )


# ==============================================================================
# Readiness Score Evaluator
# ==============================================================================

def evaluate_readiness_score(
    score: int | RAGColor,
) -> EvaluationResult:
    """Validate readiness score is in valid RAG range (1, 2, or 3).

    Args:
        score: Numeric (1=Red, 2=Yellow, 3=Green) or RAGColor enum

    Returns:
        EvaluationResult with RAG color and interpretation.
    """
    val = score.value if isinstance(score, RAGColor) else score

    if val not in (1, 2, 3):
        return EvaluationResult(
            gate_name="ReadinessScoreValidator",
            passed=False,
            message=f"Invalid readiness score: {val} (expected 1-3)",
            severity="ERROR",
        )

    try:
        rag = RAGColor(val)
        descriptions = {
            RAGColor.RED: "🔴 Little or no readiness; needs substantial improvement",
            RAGColor.YELLOW: "🟡 Partial readiness; needs improvement and focus",
            RAGColor.GREEN: "🟢 Substantial readiness; well established",
        }
        return EvaluationResult(
            gate_name="ReadinessScoreValidator",
            passed=True,
            message=f"Readiness score {rag.name}: {descriptions[rag]}",
            severity="INFO",
            details={"score": val, "color": rag.name}
        )
    except ValueError as e:
        return EvaluationResult(
            gate_name="ReadinessScoreValidator",
            passed=False,
            message=f"Invalid readiness score: {e}",
            severity="ERROR",
        )


# ==============================================================================
# Graph Context Evaluator (Orchestration)
# ==============================================================================

def evaluate_graph_context(
    graph_data: dict[str, Any],
) -> dict[str, EvaluationResult]:
    """Run all evaluation gates on graph context before sending to LLM.

    Args:
        graph_data: Knowledge graph data (typically from qode_graph.json)
                   Expected keys: nodes, edges, metadata

    Returns:
        Dict mapping gate names to EvaluationResult objects.
    """
    results: dict[str, EvaluationResult] = {}

    # Extract nodes by type
    nodes = graph_data.get("nodes", [])
    pillars = [n for n in nodes if n.get("node_type") == "pillar"]
    roles = [n for n in nodes if n.get("node_type") == "role"]
    tools = [n for n in nodes if n.get("node_type") == "tool"]
    activities = [n for n in nodes if n.get("node_type") == "activity"]

    # Gate 1: Validate pillar definitions
    for pillar in pillars:
        pillar_label = pillar.get("label", "Unknown")
        pillar_roles = [r.get("label") for r in roles
                       if r.get("id") in [n.get("id") for n in nodes]]
        pillar_tools = [t.get("label") for t in tools
                       if t.get("id") in [n.get("id") for n in nodes]]

        key = f"EngineeringPractice:{pillar_label}"
        results[key] = evaluate_engineering_practices(
            pillar_label,
            pillar_roles or [],
            pillar_tools or [],
        )

    # Gate 2: Check for minimum entity coverage
    min_entities = {
        "pillars": len(pillars) >= 6,
        "roles": len(roles) >= 4,
        "tools": len(tools) >= 5,
        "activities": len(activities) >= 0,  # Activities are optional
    }

    passed = all(min_entities.values())
    results["EntityCoverageValidator"] = EvaluationResult(
        gate_name="EntityCoverageValidator",
        passed=passed,
        message=f"Entity coverage: pillars={len(pillars)}, roles={len(roles)}, tools={len(tools)}, activities={len(activities)}",
        severity="INFO" if passed else "WARNING",
        details=min_entities,
    )

    return results


# ==============================================================================
# Convenience Functions for Chain Integration
# ==============================================================================

def validate_context_before_llm(graph_data: dict[str, Any]) -> bool:
    """Orchestrator: run all evaluations and log results.

    Returns True if all critical gates pass, False if any ERROR-severity gate fails.
    """
    results = evaluate_graph_context(graph_data)

    critical_failures = [
        r for r in results.values()
        if not r.passed and r.severity == "ERROR"
    ]

    for result in results.values():
        level = logging.WARNING if result.severity == "ERROR" else logging.INFO
        logger.log(level, str(result))

    if critical_failures:
        logger.error(
            "Context validation FAILED: %d critical issues. Proceeding with degraded context.",
            len(critical_failures),
        )
        return False

    logger.info("Context validation passed. Ready for LLM processing.")
    return True
