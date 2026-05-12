# Frontend 参考手册

## 快速启动

### 生产模式（FastAPI 托管 SPA）

```bash
# 第一次或代码变更后，构建前端
cd rag-anything/server/frontend
npm run build          # 输出到 ../static/dist/

# 启动 FastAPI（同时 serve API + SPA）
cd rag-anything
uvicorn server.app:app --host 0.0.0.0 --port 9621 --reload

# 访问
open http://localhost:9621
```

### 开发模式（Vite HMR）

```bash
# 终端 1：FastAPI（提供 API）
cd rag-anything
uvicorn server.app:app --host 0.0.0.0 --port 9621 --reload

# 终端 2：Vite dev server（HMR，自动 proxy 到 FastAPI）
cd rag-anything/server/frontend
npm run dev            # → http://localhost:5173

# 访问开发版本
open http://localhost:5173
```

### 测试

```bash
cd rag-anything/server/frontend
npm test               # vitest run（一次性）
npm run test:watch     # 监视模式
```

---

## 架构

```
rag-anything/
└── server/
    ├── app.py                    # FastAPI：所有 /api/* 路由 + 最后挂载 SPA
    ├── static/
    │   └── dist/                 # npm run build 输出（不入 git）
    └── frontend/                 # Vite 项目根
        ├── vite.config.ts        # outDir: ../static/dist，dev proxy → :9621
        └── src/
            ├── main.tsx          # 入口：主题初始化 + ReactDOM.createRoot
            ├── App.tsx           # QueryClientProvider + BrowserRouter + AppShell
            ├── types/index.ts    # 所有 TypeScript 接口（对齐后端 Pydantic）
            ├── api/              # 纯函数，每文件对应一组端点
            │   ├── client.ts     # axios 实例，统一错误处理
            │   ├── workspace.ts  # getWorkspaces / freeze / unfreeze
            │   ├── files.ts      # getFiles / getFileContent / uploadFile / deleteDocument
            │   ├── jobs.ts       # getJobs / cancelJob / retryWorkspace
            │   ├── graph.ts      # getOverview / getSubgraph / searchGraph
            │   └── query.ts      # openQueryStream（返回 fetch Response）
            ├── store/index.ts    # Zustand（workspaceId / theme / selectedFileId / pendingPageNum）
            ├── hooks/            # React Query hooks（消费 api/ 层）
            │   ├── useWorkspaces.ts
            │   ├── useFiles.ts
            │   ├── useJobs.ts    # refetchInterval 动态轮询（running → 2s，否则停止）
            │   ├── useGraph.ts
            │   └── useStreamQuery.ts   # fetch + ReadableStream SSE，非 React Query
            ├── components/
            │   ├── layout/
            │   │   ├── AppShell.tsx    # Outlet + TopNav + 全局 Job 失败通知
            │   │   ├── TopNav.tsx      # 导航 Tab + Jobs 徽章 + WorkspaceSwitcher + ThemeToggle
            │   │   ├── WorkspaceSwitcher.tsx
            │   │   └── ThemeToggle.tsx
            │   ├── chat/
            │   │   ├── ChatInput.tsx   # textarea + mode Select + Enter 发送
            │   │   ├── MessageBubble.tsx
            │   │   ├── MessageList.tsx # 自动滚动，用户上滚暂停
            │   │   ├── ReasoningTrace.tsx  # Collapsible 折叠面板
            │   │   └── CitationChip.tsx    # 点击 → navigate('/documents') + PDF 跳页
            │   ├── documents/
            │   │   ├── FileList.tsx    # 文件列表 + 删除 AlertDialog + Freeze Switch
            │   │   ├── FileUpload.tsx  # 拖拽 + 点击上传，扩展名校验
            │   │   ├── MarkdownViewer.tsx  # react-markdown + rehype-highlight + KaTeX
            │   │   └── PdfViewer.tsx   # react-pdf + 翻页/缩放工具栏，支持 initialPage 跳转
            │   ├── graph/
            │   │   ├── ForceGraph.tsx  # react-force-graph-2d，Canvas 渲染，节点颜色按类型
            │   │   ├── GraphSearch.tsx # 调 /graph/{ws}/search，highlight 结果节点
            │   │   └── NodeSheet.tsx   # 右侧抽屉，显示节点详情
            │   ├── jobs/
            │   │   ├── JobCard.tsx     # Progress + Cancel / Retry 按钮
            │   │   └── JobList.tsx     # 按 status 过滤
            │   └── ui/                 # shadcn/ui 组件（@base-ui/react 驱动）
            └── routes/                 # 懒加载页面
                ├── ChatPage.tsx
                ├── DocumentsPage.tsx
                ├── GraphPage.tsx
                └── JobsPage.tsx
```

---

## 数据流

```
浏览器
  ├── REST 请求 → TanStack Query → src/api/*  → FastAPI /files /jobs /graph /workspace
  ├── SSE 流式  → useStreamQuery → fetch POST → FastAPI /query/stream
  ├── 全局状态  → Zustand store（workspaceId / theme / pendingPageNum / lastSeenJobStatuses）
  └── KG 渲染  → react-force-graph-2d ← /graph/{ws}/overview + /graph/{ws}/subgraph
```

---

## 关键设计决策

| 决策 | 说明 |
|------|------|
| Tailwind v4 | CSS-first 配置（`@theme` in `index.css`），无 `tailwind.config.ts` |
| shadcn/ui v4 | 使用 `@base-ui/react`（非 Radix），Trigger 无需 `asChild` |
| React Router v7 | library mode，`<BrowserRouter>` + `<Routes>` API 与 v6 兼容 |
| SSE 用 fetch | `POST /query/stream` 不支持 `EventSource`（仅 GET），用 `fetch + ReadableStream` |
| Citation jump | 点击引用 chip → Zustand `pendingPageNum` → navigate → PdfViewer `initialPage` |
| Job 通知 | AppShell 层 diff 上一轮 job 状态，`running→failed` 触发 Sonner toast |
| SPA mount 顺序 | `app.mount("/", StaticFiles(..., html=True))` **必须在所有 API 路由之后** |
| `static/dist/` | 不入 git，CI/开发前需先 `npm run build` |
