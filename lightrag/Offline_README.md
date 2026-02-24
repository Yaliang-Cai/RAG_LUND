# LightRAG Ollama 离线/内网环境全流程部署指南 (Ubuntu)
本指南专门针对在 Ubuntu 离线或受限网络环境下，如何通过手动注入 GGUF 模型、修复 Tiktoken SSL 报错以及微调 LightRAG 参数，实现本地化 RAG 系统的完整部署。
## 1. 项目初始化与环境准备
```bash
# 1. 克隆项目仓库
```bash
# 1. 克隆项目仓库
git clone [https://github.com/Jobfromearth/LightRAG.git](https://github.com/Jobfromearth/LightRAG.git)
cd LightRAG

# 2. 创建并激活 Conda 虚拟环境
conda create -n lightRAG python=3.10
conda activate lightRAG

# 3. 安装依赖
pip install -e .

# 4. 获取演示文档 (A Christmas Carol)
# 如果无法通过 curl 下载，请手动获取文本并保存为项目根目录下的 book.txt
curl [https://raw.githubusercontent.com/gusye1234/nano-graphrag/main/tests/mock_data.txt](https://raw.githubusercontent.com/gusye1234/nano-graphrag/main/tests/mock_data.txt) > ./book.txt
```


## 2. 下获取演示文档 (A Christmas Carol)
download the demo document of "A Christmas Carol" by Charles Dickens
```bash
curl https://raw.githubusercontent.com/gusye1234/nano-graphrag/main/tests/mock_data.txt > ./book.txt
```
## 3. 离线注入模型 (手动注册 GGUF)
LightRAG 需要利用LLM和Embeding模型来完成文档索引和知识库查询工作。在初始化LightRAG的时候需要把阶段，需要把LLM和Embedding的操作函数注入到对象中：
在无法执行 ollama pull 的环境下，需手动从 Hugging Face 下载模型并注册。

此处由于ubuntu offline 无法使用ollama pull，手动从huggingface上clone gguf格式模型，再手动配置Modelfile，再将模型注册到Ollama，再调用

# 3.1 嵌入模型，例如nomic-embed-text

从ollama处获得：https://ollama.com/library/nomic-embed-text:137m-v1.5-fp16

从hugging face处拉取：https://huggingface.co/nomic-ai/nomic-embed-text-v1.5-GGUF
```bash
wget https://huggingface.co/nomic-ai/nomic-embed-text-v1.5-GGUF/resolve/main/nomic-embed-text-v1.5.f16.gguf
```
修改`Modelfile.nomic`
```
nano Modelfile.nomic
```
粘贴以下内容
```
# 指向你下载好的本地文件
FROM ./nomic-embed-text-v1.5.f16.gguf

# Embedding 模型通常不需要复杂的 TEMPLATE 和 SYSTEM
# 但可以设置一些基础参数
PARAMETER num_ctx 8192
```

将模型注册到 Ollama
```
ollama create nomic-embed-text -f Modelfile.nomic
```
验证是否成功：
```
ollama list
```
你应该能看到 nomic-embed-text 出现在列表中。
验证 Embedding 是否工作：
```
curl http://localhost:11434/api/embeddings -d '{
  "model": "nomic-embed-text",
  "prompt": "hello world"
}'
```
运行emb模型:
```
ollama run nomic-embed-text
```

# 3.2 llm模型，例如Qwen2
从Ollama处获得：https://ollama.com/library/qwen2:7b

从hugging face处拉取：https://huggingface.co/Qwen/Qwen2-7B-Instruct-GGUF
```bash
wget https://huggingface.co/Qwen/Qwen2-7B-Instruct-GGUF/resolve/main/qwen2-7b-instruct-q4_k_m.gguf
```

修改`Modelfile`
```
nano Modelfile
```

将以下内容粘贴进去
```
# 1. 确认路径正确
FROM ./qwen2-7b-instruct-q4_k_m.gguf

# 2. 设置参数
PARAMETER stop "<|im_start|>"
PARAMETER stop "<|im_end|>"
PARAMETER num_ctx 32768

# 3. 系统提示词
SYSTEM """You are a helpful assistant."""

# 4. 对话模板 (支持工具调用)
TEMPLATE """{{ if .Messages }}
{{- if or .System .Tools }}<|im_start|>system
{{ .System }}
{{- if .Tools }}

# Tools

You are provided with function signatures within <tools></tools> XML tags:
<tools>{{- range .Tools }}
{"type": "function", "function": {{ .Function }}}{{- end }}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>
{{- end }}<|im_end|>
{{ end }}
{{- range $i, $_ := .Messages }}
{{- $last := eq (len (slice $.Messages $i)) 1 -}}
{{- if eq .Role "user" }}<|im_start|>user
{{ .Content }}<|im_end|>
{{ else if eq .Role "assistant" }}<|im_start|>assistant
{{ if .Content }}{{ .Content }}
{{- else if .ToolCalls }}<tool_call>
{{ range .ToolCalls }}{"name": "{{ .Function.Name }}", "arguments": {{ .Function.Arguments }}}
{{ end }}</tool_call>
{{- end }}{{ if not $last }}<|im_end|>
{{ end }}
{{- else if eq .Role "tool" }}<|im_start|>tool
<tool_response>
{{ .Content }}
</tool_response><|im_end|>
{{ end }}
{{- if and (ne .Role "assistant") $last }}<|im_start|>assistant
{{ end }}
{{- end }}
{{- else }}
{{- if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{ if .Prompt }}<|im_start|>user
{{ .Prompt }}<|im_end|>
{{ end }}<|im_start|>assistant
{{ end }}{{ .Response }}{{ if .Response }}<|im_end|>{{ end }}"""
```

创建Ollama模型：
```
ollama create -f Modelfile qwen2m
```
验证模型：
```
ollama list
```
你应该能看到 qwen2m 出现在列表中。

运行模型：
```
ollama run qwen2m
```

# 3.3 最终的文件结构图示
现在你的 LightRAG 文件夹应该是这样的：
```
LightRAG/
├── qwen2-7b-instruct-q4_k_m.gguf      <-- Qwen2 模型文件
├── Modelfile                          <-- Qwen2 的配置文件
├── nomic-embed-text-v1.5.f16.gguf    <-- Nomic 模型文件
├── Modelfile.nomic                    <-- Nomic 的配置文件
└── ...
```
# 4. 解决 Tiktoken 离线 SSL 报错
SSL 证书验证失败，LightRAG 在启动时需要下载一个名为 tiktoken 的编码文件（用于计算 Token 数量）。由于你的网络环境（可能是在公司内网或使用了代理）无法通过 SSL 验证连接到 OpenAI 的存储服务器，导致下载中断。

手动下载文件： 在浏览器中打开以下链接，并将文件下载到你的 Ubuntu 机器上（比如放到 `~/workspaces/LightRAG/tokens/` 文件夹下）：[ o200k_base.tiktoken 下载链接](https://www.google.com/url?sa=E&source=gmail&q=https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken)

通过模拟 tiktoken 库内部的缓存匹配机制，彻底欺骗程序，让它以为已经下载好了文件，此处参考：https://blog.csdn.net/chenxin0215/article/details/150491393

创建确定的缓存目录
```
mkdir -p /home/h50056787/workspaces/LightRAG/tiktoken_cache
```
将下载的文件重命名为“哈希值”文件名
```
mv /home/h50056787/workspaces/LightRAG/tokens/o200k_base.tiktoken /home/h50056787/workspaces/LightRAG/tiktoken_cache/fb374d419588a4632f3f557e76b4b70aebbca790
```
在 Python 脚本中配置环境，修改 `examples/lightrag_ollama_demo.py`
```python
import os
# 指向刚才创建的那个包含哈希文件名文件的目录
os.environ["TIKTOKEN_CACHE_DIR"] = "/home/h50056787/workspaces/LightRAG/tiktoken_cache"
```
文件结构确认
```
/home/h50056787/workspaces/LightRAG/
└── tiktoken_cache/
    └── fb374d419588a4632f3f557e76b4b70aebbca790  <-- 这是重命名后的原 o200k 文件
```


# 5. 增加上下文大小
为了使 LightRAG 正常工作，上下文大小至少需要 32k tokens。默认情况下，Ollama 模型的上下文大小为 8k。您可以通过以下两种方式之一来实现：

在 Modelfile 中增加 num_ctx 参数
编辑 `Modelfile`，添加以下行：(前面粘贴板已经包含)
```
PARAMETER num_ctx 32768
```

# 6. 使用 llm_model_kwargs 参数来配置 Ollama，修改 `examples/lightrag_ollama_demo.py` 如下
```
rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=ollama_model_complete,
        llm_model_name="qwen2m",
        #llm_model_name=os.getenv("LLM_MODEL", "qwen2.5-coder:7b"),
        summary_max_tokens=8192,
        llm_model_kwargs={
            "host": os.getenv("LLM_BINDING_HOST", "http://localhost:11434"),
            "options": {"num_ctx": 8192},
            "timeout": int(os.getenv("TIMEOUT", "300")),
        },
        # Note: ollama_embed is decorated with @wrap_embedding_func_with_attrs,
        # which wraps it in an EmbeddingFunc. Using .func accesses the original
        # unwrapped function to avoid double wrapping when we create our own
        # EmbeddingFunc with custom configuration (embedding_dim, max_token_size).
        embedding_func=EmbeddingFunc(
            embedding_dim=int(os.getenv("EMBEDDING_DIM", "768")),
            max_token_size=int(os.getenv("MAX_EMBED_TOKENS", "8192")),
            func=partial(
                ollama_embed.func,  # Access the unwrapped function to avoid double EmbeddingFunc wrapping
                embed_model=os.getenv("EMBEDDING_MODEL", "nomic-embed-text"),
                host=os.getenv("EMBEDDING_BINDING_HOST", "http://localhost:11434"),
            ),
        ),
    )
```
此处注意：
```
llm_model_name="qwen2m",
embedding_dim=int(os.getenv("EMBEDDING_DIM", "768")),
embed_model=os.getenv("EMBEDDING_MODEL", "nomic-embed-text"),
```

# 7. 运行脚本

```
cd /home/h50056787/workspaces/LightRAG
python examples/lightrag_ollama_demo.py
```
检查日志: log文档为：`/LightRAG/lightrag_ollama_demo.log`
