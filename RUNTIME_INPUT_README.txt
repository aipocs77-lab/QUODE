QODE Generic Runtime Input Package

1. Build application-only image:
   docker build -t qode-neo4j:5.0 .

2. Ensure external network exists:
   docker network create qode-network
   (skip if it already exists)

3. Put runtime files in ./input.
   Supported: PDF, DOCX, XLSX, XLSM, CSV, TXT, MD, JSON, PNG, JPG, JPEG, SVG.

4. Neo4j must be on qode-network and resolvable as 'neo4j', or set NEO4J_URI.

5. Run:
   docker compose up -d

Startup blocks API launch until runtime ingestion completes.
All readable files -> ChromaDB.
QODE Q_Stories workbook -> structured Neo4j graph.
Other files -> LLM entity/relation extraction -> Neo4j.
Then /query uses Neo4j graph retrieval + ChromaDB vector retrieval.


Neo4j reset behavior:
At every container startup ingestion, the configured Neo4j database is flushed first using:
MATCH (n) DETACH DELETE n
Then the current /app/input files are processed and the Knowledge Graph is rebuilt.
WARNING: use a dedicated Neo4j database/container because all nodes and relationships
in the configured database are deleted before ingestion.
