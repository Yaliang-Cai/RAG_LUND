RAG_LUND 后端启动指南
启动顺序：PostgreSQL → Qdrant → Neo4j → Phoenix。所有服务无需 sudo。
0. 环境
bashconda activate lightRAG
1. PostgreSQL (端口 5433)
数据目录由 PostgreSQL 16 初始化，必须用 16 启动（环境内已装好 16，不要升级到 17）。
```bash
pg_ctl -D /data/h50056787/workspaces/RAG_LUND/database/pg_data/ start
```
# 查看状态
```bash
pg_ctl -D /data/h50056787/workspaces/RAG_LUND/database/pg_data/ status
```
# 停止
```bash
pg_ctl -D /data/h50056787/workspaces/RAG_LUND/database/pg_data/ stop
```
如报 incompatible with server 或版本是 17，降回 16：
```bash
conda install -n lightRAG -c conda-forge postgresql=16
```
2. Qdrant (端口 6333)
```bash
cd /data/h50056787
./qdrant
```
Web UI: http://localhost:6333/dashboard
3. Neo4j (端口 7474 / 7687)
系统默认 Java 11 不支持，必须指定 Java 21。
```bash
cd /data/h50056787/neo4j-community-2026.02.3/bin
JAVA_HOME=/data/h50056787/jdk-21.0.10+7 ./neo4j start
# 状态 / 停止
JAVA_HOME=/data/h50056787/jdk-21.0.10+7 ./neo4j status
JAVA_HOME=/data/h50056787/jdk-21.0.10+7 ./neo4j stop
```
Web UI: http://localhost:7474
4. Phoenix (链路追踪)
注意子命令是 serve，不是 launch。
```bash
conda activate lightRAG
python -m phoenix.server.main serve
```
备注
所有命令需在对应 conda activate lightRAG 环境下执行（Qdrant 除外，无依赖）。
严禁删除 pg_data 内容，数据库数据保存于此。
机器为离线环境，conda 装包用 -c conda-forge。

5. Web 服务 (RAG-Anything，端口 9621)
前端构建 + 后端启动。所有命令在 conda activate lightRAG 下执行。
首次或前端代码更新后，需先构建前端：
```bash
cd /data/h50056787/workspaces/RAG_LUND/rag-anything/server/frontend
npm install          # 仅首次或依赖变更时需要
npm run build        # 产物输出到 ../static/dist
```
启动后端（会自动 serve 已构建的前端）：
```bash
cd /data/h50056787/workspaces/RAG_LUND/rag-anything
uvicorn server.app:app --host 0.0.0.0 --port 9621
```
访问：http://localhost:9621
备注
Web 服务依赖前面 1–4 的后端，请先确保 PostgreSQL、Qdrant、Neo4j（必要时 Phoenix）已启动。
若只改了后端代码无需重新 npm run build，直接重启 uvicorn 即可。
调试时可加 --reload，但生产/常规使用不要加。
