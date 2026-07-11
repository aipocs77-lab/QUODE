"""
graph_builder.py — QODE Knowledge Graph construction and persistence.

Builds a directed Neo4j graph representing structural relationships between
SDLC pillars, human roles, automation tools, and Q_Stories activities.
This graph is the backbone of the Graph-RAG retrieval layer, enabling the LLM
to reason across multi-hop connections that flat vector search cannot traverse.

Node types (stored as 'node_type' attribute):
    pillar   — one of the 9 SDLC pillars
    role     — a human role  (e.g. "Product Owner")
    tool     — an automation tool (e.g. "Jira")
    activity — a Q_Stories activity row from the Excel questionnaire

Edge relationship types (stored as 'rel' attribute):
    HAS_ROLE      : pillar   → role      (pillar owns this role)
    USES_TOOL     : pillar   → tool      (pillar uses this tool)
    PRECEDES      : pillar   → pillar    (sequential SDLC order 1→2→…→9)
    OWNS_ACTIVITY : role     → activity  (role is responsible for this activity)
    USED_IN       : tool     → activity  (tool is used in this activity)

Public API
----------
    PILLAR_DEFINITIONS : list[dict]  — authoritative structured pillar data

    QODEKnowledgeGraph
        .build_from_pillars() -> QODEKnowledgeGraph
        .add_excel_activities(excel_path) -> None
        .get_subgraph_text(seed_node_ids, hops) -> str
        .community_summaries() -> dict[str, str]
        .all_entity_labels() -> dict[str, str]
        .save(graph_path) -> None
        .load(graph_path) -> QODEKnowledgeGraph  [classmethod]
        .node_count : int
        .edge_count : int

    build_graph(excel_path=None) -> QODEKnowledgeGraph  [convenience]
    save_graph(graph, graph_path) -> None
    load_graph(graph_path) -> QODEKnowledgeGraph | None
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

from neo4j import GraphDatabase
from config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD

from . import eval_metrics

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default persistence path
# ---------------------------------------------------------------------------
DEFAULT_GRAPH_PATH = "./graph_db/qode_graph.json"

# ---------------------------------------------------------------------------
# Stop-words to exclude from single-word entity label matching
# ---------------------------------------------------------------------------
_STOP_WORDS: frozenset[str] = frozenset(
    {
        "with", "from", "that", "this", "have", "been", "their", "there",
        "about", "covers", "which", "these", "those", "after", "before",
        "management", "operations", "practices",  # too generic for QODE
    }
)

# ---------------------------------------------------------------------------
# Authoritative QODE pillar definitions — single source of truth for the graph
# Aligned with QODE_methodologies.md Section 3: Engineering Practices (9 practices)
# and Section 4: Three Propensities (Practice, Technology Usage, Collaboration)
# ---------------------------------------------------------------------------
PILLAR_DEFINITIONS: list[dict[str, Any]] = [
    {
        "id": "pillar_1",
        "label": "Requirements Engineering",
        "pillar_num": 1,
        "diagram_type": "process",
        "roles": ["IT Product Owner", "Business Analyst", "IT Lead"],
        "tools": ["Jira", "Confluence", "Azure DevOps Boards"],
        "summary": (
            "Capturing, translating, categorizing, and integrating requirements; "
            "traceability automation."
        ),
    },
    {
        "id": "pillar_2",
        "label": "Code Engineering",
        "pillar_num": 2,
        "diagram_type": "technology",
        "roles": ["Developer", "Tech Lead"],
        "tools": ["GitHub", "GitLab", "Bitbucket", "SonarQube"],
        "summary": (
            "Modularity, microservices architecture, in-line code quality, "
            "model-driven code generation, application security."
        ),
    },
    {
        "id": "pillar_3",
        "label": "Data Engineering",
        "pillar_num": 3,
        "diagram_type": "technology",
        "roles": ["Data Architect", "Data Modeler"],
        "tools": ["DBaaS", "ETL Tools", "Data Pipeline", "Apache Spark"],
        "summary": (
            "Schema optimization, unstructured data management, data migration, "
            "data security, archival automation."
        ),
    },
    {
        "id": "pillar_4",
        "label": "Quality Engineering",
        "pillar_num": 4,
        "diagram_type": "technology",
        "roles": ["Tester", "QA Team", "QA Lead"],
        "tools": ["Selenium", "JUnit", "pytest", "TestNG", "Cypress"],
        "summary": (
            "TDD/BDD, in-sprint automation, regression and performance test automation, "
            "shift-left testing."
        ),
    },
    {
        "id": "pillar_5",
        "label": "Build & Release Engineering",
        "pillar_num": 5,
        "diagram_type": "process",
        "roles": ["Operations", "Release Team", "DevOps Lead"],
        "tools": ["Jenkins", "GitHub Actions", "CircleCI", "Nexus", "JFrog"],
        "summary": (
            "CI/CD frameworks, toolchain integration, zero-downtime deployment, "
            "release pipeline reliability."
        ),
    },
    {
        "id": "pillar_6",
        "label": "Environment Engineering",
        "pillar_num": 6,
        "diagram_type": "technology",
        "roles": ["Operations", "Infrastructure Team", "Cloud Engineer"],
        "tools": ["Terraform", "Ansible", "Puppet", "CloudFormation", "Helm"],
        "summary": (
            "Infrastructure-as-code, environment provisioning automation, "
            "configuration management, containerization, cloud."
        ),
    },
    {
        "id": "pillar_7",
        "label": "Service Operations Engineering",
        "pillar_num": 7,
        "diagram_type": "process",
        "roles": ["Operations", "Production Support", "Support Lead"],
        "tools": ["ServiceNow", "Jira", "Splunk", "ELK Stack"],
        "summary": (
            "Production monitoring, automated incident detection and resolution, "
            "feedback loops to SDLC, ITSM workflows."
        ),
    },
    {
        "id": "pillar_8",
        "label": "Security Engineering",
        "pillar_num": 8,
        "diagram_type": "process",
        "roles": ["IT Security Team", "Security Engineer"],
        "tools": ["Snyk", "Checkmarx", "OWASP ZAP", "Twistlock"],
        "summary": (
            "Application, data, infrastructure, and code security; "
            "vulnerability scanning; chaos engineering for security; DevSecOps."
        ),
    },
    {
        "id": "pillar_9",
        "label": "Reliability Engineering",
        "pillar_num": 9,
        "diagram_type": "process",
        "roles": ["Ops", "SRE"],
        "tools": ["Prometheus", "Grafana", "Datadog","Splunk", "PagerDuty", "Dynatrace"],
        "summary": (
            "Monitoring-based failure detection, self-healing, chaos engineering, "
            "SLO/SLA/SLI management, Observability, Monitoring, RCA, Known Error Database, Observability."
        ),
    },
]

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _node_id(node_type: str, label: str) -> str:
    """Return a normalised, stable node ID from *node_type* and *label*."""
    slug = re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")
    return f"{node_type}_{slug}"


# ---------------------------------------------------------------------------
# QODEKnowledgeGraph
# ---------------------------------------------------------------------------


class QODEKnowledgeGraph:
    """Neo4j-backed QODE Knowledge Graph preserving the original public API."""

    def __init__(self) -> None:
        self._driver = GraphDatabase.driver(
            NEO4J_URI,
            auth=(NEO4J_USER, NEO4J_PASSWORD),
        )

    @staticmethod
    def _safe_label(node_type: str) -> str:
        return {
            "pillar": "Pillar",
            "role": "Role",
            "tool": "Tool",
            "activity": "Activity",
            "document": "Document",
            "system": "System",
            "process": "Process",
            "concept": "Concept",
            "metric": "Metric",
            "entity": "Entity",
        }.get(node_type, "Entity")

    def clear_database(self) -> None:
        """Delete all nodes and relationships from the configured Neo4j database."""
        logger.warning("Flushing Neo4j database before ingestion")
        with self._driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n").consume()
        logger.info("Neo4j database flush completed")

    def _merge_node(self, node_id: str, node_type: str, **attrs: Any) -> None:
        label = self._safe_label(node_type)
        props = {"id": node_id, "node_type": node_type, **attrs}
        query = f"""
        MERGE (n:QODENode:{label} {{id: $id}})
        SET n += $props
        """
        with self._driver.session() as session:
            session.run(query, id=node_id, props=props).consume()

    def _merge_edge(self, source_id: str, target_id: str, rel: str) -> None:
        allowed = {
            "HAS_ROLE", "USES_TOOL", "PRECEDES",
            "OWNS_ACTIVITY", "USED_IN",
        }
        relationship = rel if rel in allowed else "RELATED_TO"
        query = f"""
        MATCH (a:QODENode {{id: $source_id}})
        MATCH (b:QODENode {{id: $target_id}})
        MERGE (a)-[r:{relationship}]->(b)
        SET r.rel = $relationship
        """
        with self._driver.session() as session:
            session.run(
                query,
                source_id=source_id,
                target_id=target_id,
                relationship=relationship,
            ).consume()

    def _has_node(self, node_id: str) -> bool:
        with self._driver.session() as session:
            row = session.run(
                "MATCH (n:QODENode {id: $id}) RETURN count(n) > 0 AS found",
                id=node_id,
            ).single()
        return bool(row and row["found"])

    def build_from_pillars(self) -> "QODEKnowledgeGraph":
        pillar_ids = [p["id"] for p in PILLAR_DEFINITIONS]
        for pillar in PILLAR_DEFINITIONS:
            pid = pillar["id"]
            self._merge_node(
                pid, "pillar",
                label=pillar["label"],
                pillar_num=pillar["pillar_num"],
                diagram_type=pillar["diagram_type"],
                summary=pillar["summary"],
            )
            for role_label in pillar["roles"]:
                rid = _node_id("role", role_label)
                self._merge_node(rid, "role", label=role_label)
                self._merge_edge(pid, rid, "HAS_ROLE")
            for tool_label in pillar["tools"]:
                tid = _node_id("tool", tool_label)
                self._merge_node(tid, "tool", label=tool_label)
                self._merge_edge(pid, tid, "USES_TOOL")

        for i in range(len(pillar_ids) - 1):
            self._merge_edge(pillar_ids[i], pillar_ids[i + 1], "PRECEDES")

        logger.info(
            "Graph built from pillar definitions: %d nodes, %d edges",
            self.node_count, self.edge_count,
        )
        return self

    def add_excel_activities(self, excel_path: str | Path) -> None:
        try:
            import pandas as pd
            df_raw = pd.read_excel(
                str(excel_path), sheet_name="Q_Stories", header=3
            ).iloc[2:]
        except Exception as exc:
            logger.warning(
                "Graph builder: could not read Q_Stories from '%s': %s",
                excel_path, exc,
            )
            return

        yes_df = df_raw[df_raw.get("Yes / No", df_raw.iloc[:, 1]) == "Yes"]
        activity_count = 0
        for _, row in yes_df.iterrows():
            s_num = str(row.get("S#", "")).strip()
            if not s_num or s_num.lower() == "nan":
                continue

            act_id = f"activity_s{s_num}"
            team = str(row.get("Team / owner role", "")).strip()
            tool = str(row.get("Automation tool", "")).strip()
            inp = str(row.get("Input", "")).strip()
            out = str(row.get("Output", "")).strip()
            criticality = str(row.get("Criticality", "")).strip()
            pred = str(row.get("Predecessor 1 (incl. INIT)", "")).strip()
            label = (
                f"Activity S{s_num}: {inp} → {out}"
                if inp and out and inp.lower() != "nan" and out.lower() != "nan"
                else f"Activity S{s_num}"
            )

            self._merge_node(
                act_id, "activity",
                label=label,
                s_num=s_num,
                team=team if team.lower() != "nan" else "",
                tool=tool if tool.lower() != "nan" else "",
                criticality=criticality if criticality.lower() != "nan" else "",
                predecessor=pred if pred.lower() != "nan" else "",
            )
            activity_count += 1

            if team and team.lower() not in ("nan", ""):
                rid = _node_id("role", team)
                self._merge_node(rid, "role", label=team)
                self._merge_edge(rid, act_id, "OWNS_ACTIVITY")
            if tool and tool.lower() not in ("nan", ""):
                tid = _node_id("tool", tool)
                self._merge_node(tid, "tool", label=tool)
                self._merge_edge(tid, act_id, "USED_IN")

        logger.info(
            "Added %d activity nodes from Excel; graph now has %d nodes, %d edges",
            activity_count, self.node_count, self.edge_count,
        )

    def get_subgraph_text(
        self,
        seed_node_ids: list[str],
        hops: int = 2,
        max_activity_nodes: int = 12,
    ) -> str:
        if not seed_node_ids:
            return ""
        hops = max(1, min(int(hops), 5))
        query = f"""
        MATCH (seed:QODENode)
        WHERE seed.id IN $seed_ids
        OPTIONAL MATCH p=(seed)-[*0..{hops}]-(n:QODENode)
        WITH collect(DISTINCT n) AS nodes
        UNWIND nodes AS n
        OPTIONAL MATCH (n)-[r]->(m:QODENode)
        WHERE m IN nodes
        RETURN n.id AS id, properties(n) AS attrs,
               collect(DISTINCT {{
                   target_id: m.id,
                   target_label: m.label,
                   rel: coalesce(r.rel, type(r))
               }}) AS outgoing
        """
        with self._driver.session() as session:
            rows = [row.data() for row in session.run(query, seed_ids=seed_node_ids)]
        if not rows:
            return ""

        by_type = {}
        for row in rows:
            attrs = row["attrs"]
            by_type.setdefault(attrs.get("node_type", ""), []).append(row)

        lines = ["[QODE Knowledge Graph — Relevant Subgraph]"]
        visited_ids = {row["id"] for row in rows}
        for node_type, section_title in (
            ("pillar", "SDLC PILLARS"),
            ("role", "ROLES"),
            ("tool", "TOOLS"),
            ("activity", "ACTIVITIES"),
            ("document", "DOCUMENTS"),
            ("system", "SYSTEMS"),
            ("process", "PROCESSES"),
            ("concept", "CONCEPTS"),
            ("metric", "METRICS"),
            ("entity", "ENTITIES"),
        ):
            nodes_of_type = sorted(
                by_type.get(node_type, []), key=lambda row: row["id"]
            )
            if not nodes_of_type:
                continue
            if node_type == "activity":
                nodes_of_type = nodes_of_type[:max_activity_nodes]
            lines.append(f"\n{section_title}:")
            for row in nodes_of_type:
                attrs = row["attrs"]
                label = attrs.get("label", row["id"])
                summary = attrs.get("summary", "")
                line = f"  • {label}"
                if summary:
                    line += f"  —  {summary}"
                lines.append(line)
                for edge in row["outgoing"]:
                    if edge.get("target_id") in visited_ids:
                        lines.append(
                            f"      ──[{edge.get('rel', 'RELATED_TO')}]──▶ "
                            f"{edge.get('target_label', edge['target_id'])}"
                        )
        return "\n".join(lines)

    def evaluate_propensities(
        self, pillar_id: str
    ) -> eval_metrics.PropensityScore | None:
        if not self._has_node(pillar_id):
            logger.warning("Pillar %s not found in graph", pillar_id)
            return None
        with self._driver.session() as session:
            rows = session.run(
                """
                MATCH (:QODENode {id: $pillar_id})-[:USES_TOOL]->(t:QODENode)
                RETURN t.label AS label
                """,
                pillar_id=pillar_id,
            )
            tool_labels = [row["label"] or "" for row in rows]
        return eval_metrics.PropensityScore(
            practice=eval_metrics.PropensityLevel.MEDIUM,
            technology_usage=self._infer_technology_propensity(tool_labels),
            collaboration=eval_metrics.PropensityLevel.MEDIUM,
        )

    def _infer_technology_propensity(
        self, tool_labels: list[str]
    ) -> eval_metrics.PropensityLevel:
        coding_tools = {
            "terraform", "ansible", "jenkinsfile", "helm", "docker", "kubernetes"
        }
        config_tools = {
            "jenkins", "github actions", "circleci", "jira", "servicenow"
        }
        doc_tools = {"confluence", "word", "excel", "sharepoint"}
        lower_labels = [label.lower() for label in tool_labels]
        has_coding = any(
            any(tool in label for tool in coding_tools) for label in lower_labels
        )
        has_config = any(
            any(tool in label for tool in config_tools) for label in lower_labels
        )
        has_doc = any(
            any(tool in label for tool in doc_tools) for label in lower_labels
        )
        if has_coding:
            return eval_metrics.PropensityLevel.HIGH
        if has_config:
            return eval_metrics.PropensityLevel.MEDIUM
        return eval_metrics.PropensityLevel.LOW

    def community_summaries(self) -> dict[str, str]:
        summaries = {}
        for pillar in PILLAR_DEFINITIONS:
            pid = pillar["id"]
            if not self._has_node(pid):
                continue
            with self._driver.session() as session:
                roles = [
                    row["label"] for row in session.run(
                        """
                        MATCH (:QODENode {id: $id})-[:HAS_ROLE]->(n:QODENode)
                        RETURN n.label AS label ORDER BY label
                        """, id=pid,
                    )
                ]
                tools = [
                    row["label"] for row in session.run(
                        """
                        MATCH (:QODENode {id: $id})-[:USES_TOOL]->(n:QODENode)
                        RETURN n.label AS label ORDER BY label
                        """, id=pid,
                    )
                ]
                precedes_labels = [
                    row["label"] for row in session.run(
                        """
                        MATCH (:QODENode {id: $id})-[:PRECEDES]->(n:QODENode)
                        RETURN n.label AS label
                        """, id=pid,
                    )
                ]
                preceded_by_labels = [
                    row["label"] for row in session.run(
                        """
                        MATCH (n:QODENode)-[:PRECEDES]->(:QODENode {id: $id})
                        RETURN n.label AS label
                        """, id=pid,
                    )
                ]

            parts = [f"Pillar {pillar['pillar_num']} — {pillar['label']}"]
            parts.append(f"  Description: {pillar['summary']}")
            if preceded_by_labels:
                parts.append(f"  Comes after: {', '.join(preceded_by_labels)}")
            if precedes_labels:
                parts.append(f"  Leads into:  {', '.join(precedes_labels)}")
            if roles:
                parts.append(f"  Key roles:   {', '.join(roles)}")
            if tools:
                parts.append(f"  Tools used:  {', '.join(tools)}")
            summaries[pid] = "\n".join(parts)
        return summaries

    def all_entity_labels(self) -> dict[str, str]:
        label_map = {}
        with self._driver.session() as session:
            data = [
                row.data() for row in session.run(
                    """
                    MATCH (n:QODENode)
                    WHERE n.label IS NOT NULL
                    RETURN n.id AS id, properties(n) AS attrs
                    ORDER BY coalesce(n.pillar_num, 999), n.id
                    """
                )
            ]
        for row in data:
            nid = row["id"]
            attrs = row["attrs"]
            label = attrs.get("label", "")
            node_type = attrs.get("node_type", "")
            if not label:
                continue
            label_map[label.lower()] = nid
            if node_type == "pillar":
                pillar_num = attrs.get("pillar_num")
                if pillar_num:
                    label_map[f"pillar {pillar_num}"] = nid
                    label_map[f"p{pillar_num}"] = nid
                for word in re.findall(r"[a-z]{5,}", label.lower()):
                    if word not in _STOP_WORDS:
                        label_map.setdefault(word, nid)
            elif node_type == "tool":
                first_token = re.split(r"[\s/]", label)[0].lower()
                if len(first_token) >= 4:
                    label_map.setdefault(first_token, nid)
        return label_map

    def snapshot_data(self) -> dict[str, list[dict]]:
        with self._driver.session() as session:
            nodes = [
                row["node"] for row in session.run(
                    "MATCH (n:QODENode) RETURN properties(n) AS node"
                )
            ]
            edges = [
                row["edge"] for row in session.run(
                    """
                    MATCH (a:QODENode)-[r]->(b:QODENode)
                    RETURN {
                        source: a.id,
                        target: b.id,
                        rel: coalesce(r.rel, type(r))
                    } AS edge
                    """
                )
            ]
        return {"nodes": nodes, "edges": edges}

    def save(self, graph_path: str | Path) -> None:
        logger.info(
            "Knowledge graph is persisted in Neo4j; graph_path '%s' is ignored.",
            graph_path,
        )

    @classmethod
    def load(cls, graph_path: str | Path) -> "QODEKnowledgeGraph":
        instance = cls()
        instance._driver.verify_connectivity()
        logger.info("Knowledge graph loaded from Neo4j")
        return instance

    @property
    def node_count(self) -> int:
        with self._driver.session() as session:
            row = session.run(
                "MATCH (n:QODENode) RETURN count(n) AS count"
            ).single()
        return int(row["count"]) if row else 0

    @property
    def edge_count(self) -> int:
        with self._driver.session() as session:
            row = session.run(
                "MATCH (:QODENode)-[r]->(:QODENode) RETURN count(r) AS count"
            ).single()
        return int(row["count"]) if row else 0


def build_graph(
    excel_path: str | Path | None = None,
) -> QODEKnowledgeGraph:
    graph = QODEKnowledgeGraph().build_from_pillars()
    if excel_path is not None:
        graph.add_excel_activities(excel_path)
    return graph


def save_graph(
    graph: QODEKnowledgeGraph,
    graph_path: str | Path = DEFAULT_GRAPH_PATH,
) -> None:
    graph.save(graph_path)


def load_graph(
    graph_path: str | Path = DEFAULT_GRAPH_PATH,
) -> QODEKnowledgeGraph | None:
    try:
        return QODEKnowledgeGraph.load(graph_path)
    except Exception as exc:
        logger.error("Failed to connect to Neo4j: %s", exc)
        return None
